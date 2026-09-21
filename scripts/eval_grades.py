"""
中穴・穴モデルの自信度を、どの測り方で作るのが良いかを検証する。

    python scripts/eval_grades.py
    python scripts/eval_grades.py --save     # 選んだ測り方をページに反映する

先に scripts/build_models.py を実行して data/eval を作っておくこと（学習はしない）。

手順:
    1. 各レースで、そのモデルの買い目をページと同じ条件で作る
    2. 測り方の候補ごとにスコアを出し、上位10% を S、25% までを A、50% までを B とする
    3. 検証期間の前半で「格付けごとに的中率・回収率がきれいに分かれる測り方」を選ぶ
    4. 後半で、選んだ測り方が本当に効くかを答え合わせする

前半で良くても後半で崩れる測り方は、条件を探した副作用でしかない。
後半でも S と C の差が保たれたものだけを採用する。
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists                    # noqa: E402
from src.exotics import market_win_probs                    # noqa: E402
from src.confidence import model_confidence, GRADE_METHODS  # noqa: E402
from scripts.eval_filters import pick_bets, estimate_odds, truth_of, KEY  # noqa: E402

IN_DIR = "data/eval"
CONFIG_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "src", "grade_config.json")

# ページと同じ買い方
RULES = {
    "base":   {"label": "本命", "min_odds": 1.0,  "min_comb": 2.0},
    "pop6":   {"label": "中穴", "min_odds": 10.0, "min_comb": 3.0},
    "odds20": {"label": "穴",   "min_odds": 10.0, "min_comb": 5.0},
}
QUANTS = (0.50, 0.75, 0.90)       # B以上・A以上・S以上


def collect(pred, pays, model, rule, lam2, lam3, odds_cache):
    """レースごとに、買い目の結果と自信度の候補を集める。"""
    pay_map = pays.set_index("race_id").to_dict("index")
    col = f"p_top3_{model}"
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        if col not in g.columns or g[col].isna().all():
            continue
        t = truth_of(g)
        if t is None:
            continue
        s = np.clip(g[col].to_numpy(float), 1e-9, None)
        s = s / s.sum()
        numbers = g["horse_no"].astype(int).to_numpy()
        market = market_win_probs(g["odds_prev_win"].to_numpy(float))
        got = pick_bets(s, market, numbers, odds_cache.get(str(race_id), {}), 5,
                        rule["min_odds"], None, rule["min_comb"], lam2, lam3)
        if got is None:
            continue
        bt, summ, bets = got
        real = pay_map.get(race_id, {}).get(f"payout_{KEY[bt]}")
        if real is None or not np.isfinite(real):
            continue
        combos = [[int(numbers[i]) for i in c] for c in bets["combo"]]
        want = t[bt]
        hit = any((sorted(c) if bt in ("馬連", "3連複") else c) == want for c in combos)
        conf = model_confidence(g, col, rec=summ)
        rows.append({"race_id": race_id, "date": g["date"].iloc[0],
                     "stake": 100.0 * len(bets),
                     "payout": float(real) if hit else 0.0, "hit": int(hit), **conf})
    return pd.DataFrame(rows)


def grade_table(df, method, thresholds=None):
    """格付けごとの成績。thresholds を渡さなければ df 自身の分位点で切る。"""
    sc = df[method].astype(float)
    ok = sc.notna()
    d = df[ok].copy()
    if d.empty:
        return None, None
    if thresholds is None:
        thresholds = [float(sc[ok].quantile(q)) for q in QUANTS]
    b, a, s = thresholds
    d["grade"] = np.select([d[method] >= s, d[method] >= a, d[method] >= b],
                           ["S", "A", "B"], default="C")
    rows = {}
    for gr in ("S", "A", "B", "C"):
        x = d[d["grade"] == gr]
        if x.empty:
            continue
        rows[gr] = {"n": len(x), "hit": x["hit"].mean() * 100,
                    "roi": x["payout"].sum() / x["stake"].sum() * 100}
    return rows, thresholds


def separation(rows):
    """S・A と C の差。回収率の差を主に、的中率の差を従に見る。"""
    if not rows or "C" not in rows:
        return -1e9
    top = [rows[g] for g in ("S", "A") if g in rows]
    if not top:
        return -1e9
    n = sum(r["n"] for r in top)
    roi = sum(r["roi"] * r["n"] for r in top) / n
    hit = sum(r["hit"] * r["n"] for r in top) / n
    return (roi - rows["C"]["roi"]) + 0.5 * (hit - rows["C"]["hit"])


def fmt(rows):
    return "  ".join(f"{g}:{rows[g]['n']:>4}R 的中{rows[g]['hit']:>5.1f}% 回収{rows[g]['roi']:>6.1f}%"
                     for g in ("S", "A", "B", "C") if g in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=IN_DIR)
    ap.add_argument("--models", default="pop6,odds20")
    ap.add_argument("--save", action="store_true",
                    help="後半でも効いた測り方をページ用の設定として保存する")
    a = ap.parse_args()

    meta_p = os.path.join(a.dir, "meta.json")
    if not os.path.exists(meta_p):
        print(f"{meta_p} がありません。先に scripts/build_models.py を実行してください。")
        return
    with open(meta_p, encoding="utf-8") as f:
        meta = json.load(f)
    pred = load_table(os.path.join(a.dir, "predictions"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return
    lam2, lam3 = meta["lam2"], meta["lam3"]
    print(f"保存済みの予測を使います（{meta['races']:,}レース）")
    print("配当を推定しています…", flush=True)
    odds_cache = estimate_odds(pred, lam2, lam3)

    config = {}
    for model in [m.strip() for m in a.models.split(",")]:
        if model not in RULES:
            print(f"\n{model}: 買い方の設定がありません"); continue
        rule = RULES[model]
        df = collect(pred, pays, model, rule, lam2, lam3, odds_cache)
        if df.empty:
            print(f"\n{model}: 対象なし"); continue
        # CSV 経由だと日付が文字列になるので、日付に直してから前半・後半に分ける
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        dates = np.sort(df["date"].unique())
        cut = dates[len(dates) // 2]
        tune, conf = df[df["date"] < cut], df[df["date"] >= cut]
        print(f"\n{'='*78}")
        print(f"【{rule['label']}（{model}）】 前半 {len(tune)}R で選び、後半 {len(conf)}R で確かめる")
        base_roi = df["payout"].sum() / df["stake"].sum() * 100
        print(f"  格付けなしの成績: 的中 {df['hit'].mean()*100:.1f}% / 回収 {base_roi:.1f}%")

        scored = []
        for m, desc in GRADE_METHODS.items():
            rows, th = grade_table(tune, m)
            if rows is None:
                continue
            scored.append((separation(rows), m, desc, rows, th))
        scored.sort(reverse=True)

        print("\n  ── 前半での成績（S・A と C の差が大きい順）──")
        for sep, m, desc, rows, th in scored:
            print(f"  {desc:<26} 差 {sep:>7.1f}")
            print(f"      {fmt(rows)}")

        # 上位3つを後半で答え合わせ（しきい値は前半で決めたものをそのまま使う）
        print("\n  ── 後半での答え合わせ（前半のしきい値のまま）──")
        chosen = None
        for sep, m, desc, rows, th in scored[:3]:
            r2, _ = grade_table(conf, m, th)
            sep2 = separation(r2)
            ok = sep > 0 and sep2 > 0
            print(f"  {desc:<26} 前半 {sep:>6.1f} → 後半 {sep2:>6.1f}  {'★保たれた' if ok else '崩れた'}")
            if r2:
                print(f"      {fmt(r2)}")
            if ok and chosen is None:
                chosen = (m, desc)

        if chosen is None:
            print(f"\n  → {rule['label']}は、どの測り方でも後半まで効きませんでした。"
                  f"格付けを付けない方が誤解がありません。")
            continue
        m, desc = chosen
        # 採用する場合のしきい値は全期間で決め直す
        full_rows, full_th = grade_table(df, m)
        print(f"\n  → 採用: {desc}")
        print(f"     全期間: {fmt(full_rows)}")
        config[model] = {"label": rule["label"], "method": m, "desc": desc,
                         "thresholds": full_th,
                         "summary": {g: {k: round(v, 1) for k, v in r.items()}
                                     for g, r in full_rows.items()}}

    if a.save:
        if config:
            with open(CONFIG_OUT, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=1)
            print(f"\n設定を保存しました: {CONFIG_OUT}")
            print("「publish」でリポジトリに送ると、ページに反映されます。")
        else:
            print("\n後半まで効いた測り方が無かったので、設定は保存しません。")
    elif config:
        print("\nページに反映するには --save を付けて実行し、その後「publish」してください。")


if __name__ == "__main__":
    main()
