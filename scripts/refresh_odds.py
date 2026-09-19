"""
オッズだけを取り直して、妙味と買い目を更新する。

朝の `predict_day.py` が保存した能力推定を読み込むので、学習をやり直さない。
1回の更新は数十秒で終わる。

    # 1回だけ更新
    python scripts/refresh_odds.py

    # 16:40 まで動き続け、各レースの発走15分前に更新する
    python scripts/refresh_odds.py --loop --until 16:40

締切直前のオッズは朝と大きく変わることがあり、妙味（市場との乖離）は
そこで判定が変わる。能力側は変わらないので「当てにいく」の買い目はほぼ動かず、
「妙味を狙う」の買い目が入れ替わる。

netkeiba への配慮:
    全レースのオッズを毎回取りに行くと負荷が高いので、
    発走が近いレースだけを対象にする。連勝式の全オッズを取るのは
    さらに絞り込んだ直近のレースだけ。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table                            # noqa: E402
from src.netkeiba_live import fetch_all_odds                # noqa: E402
from src.value import add_value_columns                     # noqa: E402
from src.confidence import ConfidenceGrader                 # noqa: E402
from src.webexport import build_payload, write_site         # noqa: E402
from src.static_report import write as write_static         # noqa: E402

JST = timezone(timedelta(hours=9))
COMBO_TYPES = ("馬連", "馬単", "3連複", "3連単")


def now_jst():
    return datetime.now(JST)


def parse_post(post_time, day):
    """'15:40' を当日の日時にする。"""
    try:
        h, m = str(post_time).split(":")
        return datetime(day.year, day.month, day.day, int(h), int(m), tzinfo=JST)
    except (ValueError, AttributeError):
        return None


def load_state(path):
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    card = load_table(st["card"])
    card["race_id"] = card["race_id"].astype(str)
    return st, card


def targets(card, day, lead_min, window_min):
    """
    発走まで lead_min 分を切り、かつ window_min 分以内のレースを返す。
    既に発走したレースは対象外。
    """
    t = now_jst()
    rows = []
    for rid, g in card.groupby("race_id", sort=False):
        post = parse_post(g["post_time"].iloc[0], day)
        if post is None:
            continue
        mins = (post - t).total_seconds() / 60.0
        if -2 <= mins <= window_min:
            rows.append((rid, mins, post))
    rows.sort(key=lambda x: x[1])
    return rows


def refresh(state_path, lead_min=15, window_min=45, sleep=0.6, verbose=True):
    """発走が近いレースのオッズを取り直し、ページを書き直す。"""
    st, card = load_state(state_path)
    day = datetime.fromisoformat(st["date"]).date()

    cache_path = os.path.join(os.path.dirname(state_path), "odds_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    upcoming = targets(card, day, lead_min, window_min)
    if verbose:
        print(f"{now_jst():%H:%M} 対象 {len(upcoming)}レース", flush=True)

    updated = []
    for i, (rid, mins, post) in enumerate(upcoming):
        # 直近2レースは連勝式まで、それ以外は単勝だけ取る
        full = i < 2 or mins <= lead_min
        types = (1, 2, 3, 4, 5, 6, 7, 8) if full else (1,)
        want = ("単勝",) + COMBO_TYPES if full else ("単勝",)
        try:
            tbl = fetch_all_odds(rid, types=types, sleep=sleep, want=want)
        except Exception as e:
            if verbose:
                print(f"  {rid} 失敗: {e}", flush=True)
            continue
        if not tbl:
            continue
        entry = cache.setdefault(rid, {})
        if tbl.get("単勝"):
            entry["単勝"] = {str(k): v for k, v in tbl["単勝"].items()}
        for bt in COMBO_TYPES:
            if tbl.get(bt):
                entry[bt] = {"|".join(map(str, k)): v for k, v in tbl[bt].items()}
        entry["at"] = now_jst().strftime("%H:%M")
        updated.append((rid, mins, full))
        if verbose:
            kinds = "/".join(k for k in ("単勝",) + COMBO_TYPES if k in entry)
            print(f"  {rid} 発走まで{mins:.0f}分  {kinds}", flush=True)

    if not updated:
        return None

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    # 最新オッズを反映して作り直す
    d = card.copy()
    win_map = {}
    for rid, entry in cache.items():
        for no, o in (entry.get("単勝") or {}).items():
            win_map[(rid, int(no))] = float(o)
    if win_map:
        keys = list(zip(d["race_id"], d["horse_no"].astype(int)))
        d["odds_prev_win"] = [win_map.get(k, o) for k, o in
                              zip(keys, d["odds_prev_win"])]

    d = add_value_columns(d)
    d["p_win"] = d["p_blend"]

    odds_tables = {}
    for rid, entry in cache.items():
        t = {}
        for bt in COMBO_TYPES:
            if entry.get(bt):
                t[bt] = {tuple(int(x) for x in k.split("|")): float(v)
                         for k, v in entry[bt].items()}
        if t:
            odds_tables[rid] = t

    grader = ConfidenceGrader()
    grader.thresholds = st.get("thresholds")
    payload = build_payload(d, grader=grader, lam2=st["lam2"], lam3=st["lam3"],
                            min_grade=st.get("min_grade", "B"),
                            odds_tables=odds_tables,
                            meta={"date": st["date"], "venues": "中央競馬",
                                  "updated": now_jst().strftime("%H:%M")})
    out = st.get("out", "docs/today.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_site(payload, "webapp/template.html", out)
    static = out.replace(".html", "_static.html")
    write_static(payload, static, title=f"予想 {st['date']} {now_jst():%H:%M}更新")
    if verbose:
        print(f"  → 更新 {len(payload['races'])}レース分を書き出し", flush=True)
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="docs/data/state.json")
    ap.add_argument("--lead", type=float, default=15,
                    help="発走の何分前に本更新するか")
    ap.add_argument("--window", type=float, default=45,
                    help="この分数以内のレースを対象にする")
    ap.add_argument("--loop", action="store_true", help="時計を見ながら動き続ける")
    ap.add_argument("--until", default="16:45", help="ループを終える時刻 (JST)")
    ap.add_argument("--interval", type=float, default=5,
                    help="ループの確認間隔（分）")
    ap.add_argument("--commit", action="store_true",
                    help="更新のたびに git commit / push する")
    a = ap.parse_args()

    if not os.path.exists(a.state):
        print(f"状態ファイルがありません: {a.state}")
        print("先に scripts/predict_day.py を実行してください。")
        return 0

    if not a.loop:
        refresh(a.state, a.lead, a.window)
        return 0

    h, m = map(int, a.until.split(":"))
    t = now_jst()
    end = datetime(t.year, t.month, t.day, h, m, tzinfo=JST)
    print(f"{t:%H:%M} から {end:%H:%M} まで、{a.interval:.0f}分ごとに確認します",
          flush=True)

    while now_jst() < end:
        try:
            got = refresh(a.state, a.lead, a.window)
            if got and a.commit:
                os.system(
                    'git add -A docs/ && '
                    'git diff --staged --quiet || '
                    '(git commit -q -m "オッズ更新 $(TZ=Asia/Tokyo date +%H:%M)" && '
                    'git push -q origin HEAD 2>/dev/null)')
        except Exception as e:
            print(f"  更新中のエラー: {e}", flush=True)
        left = (end - now_jst()).total_seconds()
        if left <= 0:
            break
        time.sleep(min(a.interval * 60, left))

    print(f"{now_jst():%H:%M} 終了", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
