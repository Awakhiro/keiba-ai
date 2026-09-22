"""
これから走るレースを予想する。

    python scripts/predict_day.py --date 2026-09-20
    python scripts/predict_day.py --date 2026-09-20 --seed-race-id 202609040811

過去データ（data/races.*）で学習し、指定日の出馬表に対して予想を出す。
出力は docs/next.html。
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists                # noqa: E402
from src.netkeiba import to_model_schema               # noqa: E402
from src.netkeiba_live import fetch_race_card, apply_manual_odds  # noqa: E402
from src.netkeiba_result import attach_results                   # noqa: E402
from src.predict import KeibaPredictor                 # noqa: E402
from src.modelstore import load_or_train, describe     # noqa: E402
from src.features import build_features                # noqa: E402
from src.backtest import walk_forward, evaluate        # noqa: E402
from src.exotics import fit_lambdas                    # noqa: E402
from src.webexport import build_payload, write_site, freeze_finished  # noqa: E402
from src.static_report import write as write_static     # noqa: E402


def drop_scratched(entries):
    """
    出走取消とみられる馬を除く。

    発売が始まっているのに、その馬だけ単勝オッズが無い場合は取消とみなす。
    そのまま残すと中立値で埋められ、取消馬を買い目に入れてしまう。
    前日で発売前（レースの大半にオッズが無い）なら、判定せずに残す。
    """
    d = entries.copy()
    has = d["odds_prev_win"].notna()
    share = has.groupby(d["race_id"]).transform("mean")
    scratched = (~has) & (share >= 0.5)
    if scratched.any():
        rows = d[scratched]
        for rid, g in rows.groupby("race_id"):
            name = g.iloc[0].get("venue", "")
            no = g.iloc[0].get("race_no", "")
            print(f"  出走取消とみなして除外: {name}{no}R "
                  f"{', '.join(str(int(x)) for x in g['horse_no'])}番")
        d = d[~scratched].copy()
        d["field_size"] = d.groupby("race_id")["horse_no"].transform("size")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="対象日 (既定: 日本時間の今日)")
    ap.add_argument("--seed-race-id", default=None,
                    help="一覧が取れない場合、その日のレースIDを1つ渡す")
    ap.add_argument("--odds-file", default=None, help="手入力オッズのテキスト")
    ap.add_argument("--min-grade", default="B")
    ap.add_argument("--out", default="docs/today.html")
    ap.add_argument("--archive", action="store_true",
                    help="日付つきの控えを docs/archive/ に残す")
    ap.add_argument("--retrain", action="store_true",
                    help="学習データが変わっていなくても学習し直す")
    ap.add_argument("--save-state", default="docs/data/state.json",
                    help="能力推定を保存する。オッズだけ更新する際に再学習を省ける")
    a = ap.parse_args()

    if a.date:
        from src.netkeiba_live import parse_date
        target = parse_date(a.date)
    else:
        # GitHub のランナーは UTC なので、日本時間の「今日」を使う
        target = datetime.now(JST).date()
    print(f"対象日: {target}（日本時間 {datetime.now(JST):%Y-%m-%d %H:%M}）")

    history = to_model_schema(load_table("data/races"))
    print(f"学習データ {len(history):,}行 / {history['race_id'].nunique():,}レース")

    print("\n出馬表を取得します")
    try:
        entries, odds_tables = fetch_race_card(target, seed_race_id=a.seed_race_id)
    except RuntimeError as e:
        # 開催が無い日は正常終了させる（ワークフローを赤くしない）
        print(f"取得できませんでした: {e}")
        print("開催が無い日か、まだ出馬表が公開されていません。")
        return
    if a.odds_file and os.path.exists(a.odds_file):
        entries = apply_manual_odds(entries, open(a.odds_file, encoding="utf-8").read())

    # 既に発走したレースがあれば結果を取り込む（途中から実行した場合）
    entries, got, _pays = attach_results(entries, sleep=1.0, verbose=False)
    if got:
        print(f"発走済みレースの結果を取得: {len(got)}レース")

    entries["odds_prev_win"] = pd.to_numeric(entries["odds_prev_win"], errors="coerce")
    entries = drop_scratched(entries)
    has_odds = entries["odds_prev_win"].notna()

    # 発売前でオッズが無い馬は、中立な値（全馬同じ支持率）で埋める。
    # 本命・中穴・穴の能力推定はオッズを使わないので影響を受けない。
    if not has_odds.all():
        entries["odds_prev_win"] = entries["odds_prev_win"].fillna(
            entries.groupby("race_id")["horse_no"].transform("size").astype(float))

    print()
    p, trained = load_or_train(history, force=a.retrain)
    if trained or p.grader is None:
        # 自信度の基準と順位割引は、学習したときだけ作り直す
        feat = build_features(history)
        _pred, _conf = walk_forward(feat, n_folds=3, verbose=False)
        _s, _o, grader, _d = evaluate(_pred, _conf)
        lam2, lam3 = fit_lambdas(_pred)
        p.grader = grader
        p.lam2, p.lam3 = lam2, lam3
        from src.modelstore import save as save_model
        save_model(p, history)      # 基準を含めて保存し直す
    else:
        grader = p.grader
        lam2 = getattr(p, "lam2", 0.81)
        lam3 = getattr(p, "lam3", 0.65)
    out = p.predict_day(history, entries)
    have = ["本命"] + (["中穴"] if "p_top3_mid" in out.columns else []) \
                    + (["穴"] if "p_top3_long" in out.columns else [])
    print(f"使えるモデル: {' / '.join(have)}")
    if len(have) < 3:
        print("  中穴・穴が欠けています。学習データか払戻データが不足している可能性があります。")

    payload = build_payload(out, grader=grader, lam2=lam2, lam3=lam3,
                            min_grade=a.min_grade, odds_tables=odds_tables,
                            meta={"date": target.isoformat(),
                                  "venues": "・".join(sorted(entries["venue"].unique()))})
    # 途中から実行し直した場合も、発走済みのレースは固定済みの予想を使う
    if a.save_state:
        payload = freeze_finished(payload, os.path.dirname(a.save_state),
                                  odds_tables=odds_tables)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_site(payload, "webapp/template.html", a.out)
    static_path = a.out.replace(".html", "_static.html")
    write_static(payload, static_path, title=f"予想 {target}")

    # 日付つきの控えを残しておく（あとで的中を振り返れる）
    if a.archive:
        arc = os.path.join(os.path.dirname(a.out) or ".", "archive")
        os.makedirs(arc, exist_ok=True)
        stamp = datetime.now(JST).strftime("%H%M")
        write_static(payload, os.path.join(arc, f"{target}_{stamp}.html"),
                     title=f"予想 {target} {stamp[:2]}:{stamp[2:]}")

    print(f"\n書き出し: {a.out}")
    print(f"          {static_path}（JavaScript 不要。こちらが確実に開けます）")
    # 能力推定を保存しておく。以降はオッズを取り直すだけで妙味を再計算できる。
    if a.save_state:
        import json
        from src.store import save_table
        os.makedirs(os.path.dirname(a.save_state) or ".", exist_ok=True)
        stem = os.path.splitext(a.save_state)[0] + "_card"
        cols = [c for c in out.columns if c in (
            "race_id", "date", "venue", "race_no", "race_name", "surface",
            "distance", "turn", "class_level", "field_size", "post_time",
            "horse_id", "horse_name", "horse_no", "frame_no", "age", "sex",
            "weight_carried", "jockey_id", "jockey_name", "trainer_id",
            "odds_prev_win", "p_top3", "p_win_pure",
            "p_top3_mid", "p_win_mid", "p_top3_long", "p_win_long")]
        save_table(out[cols], stem)

        # 取得したオッズもキャッシュに残す。
        # これを保存しないと、発走が近くないレースは後の更新で
        # オッズが空になり「配当は推定」に落ちてしまう。
        from scripts.refresh_odds import ODDS_FORMAT_VERSION
        cache = {"_version": ODDS_FORMAT_VERSION}
        stamp = datetime.now(JST).strftime("%H:%M")
        for rid, tables in (odds_tables or {}).items():
            entry = {"at": stamp}
            for bt, tbl in tables.items():
                if bt == "単勝":
                    entry["単勝"] = {str(k): float(v) for k, v in tbl.items()}
                elif bt in ("馬連", "馬単", "3連複", "3連単"):
                    entry[bt] = {"|".join(map(str, k)): float(v)
                                 for k, v in tbl.items()}
            if len(entry) > 1:
                cache[str(rid)] = entry
        # 単勝しか無いレースも、単勝だけは残しておく
        for rid, g in entries.groupby("race_id"):
            key = str(rid)
            if key in cache or g["odds_prev_win"].isna().all():
                continue
            cache[key] = {"at": stamp,
                          "単勝": {str(int(n)): float(o) for n, o in
                                  zip(g["horse_no"], g["odds_prev_win"])
                                  if pd.notna(o)}}
        cache_path = os.path.join(os.path.dirname(a.save_state), "odds_cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print(f"オッズを保存: {len(cache)-1}レース分")

        with open(a.save_state, "w", encoding="utf-8") as f:
            json.dump({"date": str(target), "lam2": lam2, "lam3": lam3,
                       "thresholds": grader.thresholds,
                       "min_grade": a.min_grade, "card": stem,
                       "out": a.out}, f, ensure_ascii=False)
        print(f"能力推定を保存: {a.save_state}")

    print(f"{len(payload['races'])}レースを書き出しました")
    if not has_odds.all():
        print("オッズが欠けたレースがあります。「当てにいく」（モデルA）で見てください。")


if __name__ == "__main__":
    main()
