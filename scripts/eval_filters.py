"""
保存済みの予測を読んで、買い目の条件を変えながら検証する。学習はしない。

    python scripts/eval_filters.py
    python scripts/eval_filters.py --models base,pop8,odds40 --min-odds 1,5,10,20

先に scripts/build_models.py を実行しておくこと。
学習を挟まないので、条件を変えた検証が数十秒で終わる。

試せる条件:
    --min-odds   1点あたりの配当の下限。これを割る組は買わない
    --max-odds   配当の上限。当たらない大穴を避ける
    --min-comb   合成オッズの下限。1 ÷ Σ(1/各組のオッズ)。画面と同じ定義
    --points     1レースあたりの点数

「5倍以下は買わない」のような制限が効くのかを、実際の払戻で確かめる。
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists                    # noqa: E402
from src.exotics import build_bets, bet_summary, market_win_probs  # noqa: E402
import src.recommend as R                                   # noqa: E402

IN_DIR = "data/eval"
KEY = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}


def truth_of(g):
    fin = g["finish_pos"].to_numpy(dtype=float)
    try:
        i = [int(np.where(fin == k)[0][0]) for k in (1, 2, 3)]
    except IndexError:
        return None
    no = g["horse_no"].to_numpy()
    a, b, c = (int(no[x]) for x in i)
    return {"馬連": sorted([a, b]), "馬単": [a, b],
            "3連複": sorted([a, b, c]), "3連単": [a, b, c]}


def pick_bets(probs, market, numbers, odds_by_type, k,
              min_odds, max_odds, min_comb, lam2, lam3, min_ev=None):
    """
    券種ごとに候補を作り、条件を満たす中から当たりやすい順に k 点選ぶ。
    合成オッズの下限を満たす券種の中から、最も当たりやすいものを採用する（画面と同じ選び方）。
    """
    best = None
    for bt in ("馬連", "馬単", "3連複", "3連単"):
        tbl = odds_by_type.get(bt)
        if not tbl:
            continue
        pos = {int(n): i for i, n in enumerate(numbers)}
        actual = {}
        for combo, o in tbl.items():
            try:
                actual[tuple(pos[int(x)] for x in combo)] = float(o)
            except (KeyError, ValueError, TypeError):
                continue
        if not actual:
            continue
        bets = build_bets(probs, market, bt, max_points=k * 6,
                          strategy="hit", actual_odds=actual,
                          lam2=lam2, lam3=lam3)
        if bets.empty:
            continue
        # 1点あたりの配当で足切り
        m = bets["odds"] >= min_odds
        if max_odds:
            m &= bets["odds"] <= max_odds
        bets = bets[m]
        if len(bets) < k:
            continue
        bets = bets.head(k)
        s = bet_summary(bets)
        if min_comb and s["合成オッズ"] < min_comb:
            continue
        if min_ev and s["期待回収率"] < min_ev:
            continue
        # 期待回収率のしきい値。届かない券種は候補から外す
        if min_ev and s["期待回収率"] < min_ev:
            continue
        if best is None or s["的中確率"] > best[1]["的中確率"]:
            best = (bt, s, bets)
    return best


def run(pred, pays, model, k, min_odds, max_odds, min_comb, lam2, lam3,
        odds_cache, min_ev=None):
    pay_map = pays.set_index("race_id").to_dict("index")
    col_p = f"p_top3_{model}"
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        if col_p not in g.columns or g[col_p].isna().all():
            continue
        t = truth_of(g)
        if t is None:
            continue
        s = np.clip(g[col_p].to_numpy(float), 1e-9, None)
        s = s / s.sum()
        numbers = g["horse_no"].astype(int).to_numpy()
        market = market_win_probs(g["odds_prev_win"].to_numpy(float))

        got = pick_bets(s, market, numbers, odds_cache.get(str(race_id), {}),
                        k, min_odds, max_odds, min_comb, lam2, lam3, min_ev)
        if got is None:
            continue
        bt, summary, bets = got
        real = pay_map.get(race_id, {}).get(f"payout_{KEY[bt]}")
        if real is None or not np.isfinite(real):
            continue
        want = t[bt]
        combos = [[int(numbers[i]) for i in c] for c in bets["combo"]]
        hit = any((sorted(c) if bt in ("馬連", "3連複") else c) == want
                  for c in combos)
        rows.append({"race_id": race_id, "券種": bt, "点数": len(bets),
                     "stake": 100.0 * len(bets),
                     "payout": float(real) if hit else 0.0, "hit": int(hit)})
    return pd.DataFrame(rows)


def estimate_odds(pred, lam2, lam3):
    """
    連勝式の配当を、単勝オッズから推定して用意する。
    実際の全組み合わせオッズは過去に遡って取れないので、
    条件比較の目的ではこれで十分（同じ推定を全条件で使うため公平）。
    """
    from src.exotics import estimated_payouts
    cache = {}
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        nos = g["horse_no"].astype(int).to_numpy()
        q = market_win_probs(g["odds_prev_win"].to_numpy(float))
        t = {}
        for bt in ("馬連", "馬単", "3連複", "3連単"):
            tbl = estimated_payouts(q, bt, lam2, lam3)
            t[bt] = {tuple(int(nos[i]) for i in combo): o
                     for combo, o in tbl.items()}
        cache[str(race_id)] = t
    return cache


def report(df, label):
    if df is None or df.empty:
        print(f"  {label:<34} 対象なし")
        return None
    s, p = df["stake"].to_numpy(float), df["payout"].to_numpy(float)
    roi = p.sum() / s.sum() * 100
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(s), size=(2000, len(s)))
    lo = float(np.percentile(p[i].sum(1) / s[i].sum(1) * 100, 2.5))
    mix = df["券種"].value_counts().head(2).to_dict()
    print(f"  {label:<34}{len(df):>5}R  的中 {df['hit'].mean()*100:>5.1f}%  "
          f"回収 {roi:>6.1f}%  下限 {lo:>5.0f}%  {mix}")
    return {"label": label, "roi": roi, "lo": lo, "n": len(df),
            "hit": df["hit"].mean() * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=IN_DIR)
    ap.add_argument("--models", default="base,pop8,odds40")
    ap.add_argument("--min-odds", default="1,3,5,10,20",
                    help="1点あたりの配当の下限（カンマ区切り）")
    ap.add_argument("--max-odds", default="", help="配当の上限")
    ap.add_argument("--min-comb", default="0", help="合成オッズの下限（例 2,3,5）。0 で条件なし")
    ap.add_argument("--points", default="5")
    ap.add_argument("--min-ev", default="",
                    help="期待回収率のしきい値（例 1.0,1.2）。届かないレースは買わない")
    a = ap.parse_args()

    meta_p = os.path.join(a.dir, "meta.json")
    if not os.path.exists(meta_p):
        print(f"{meta_p} がありません。"
              "先に scripts/build_models.py を実行してください。")
        return
    with open(meta_p, encoding="utf-8") as f:
        meta = json.load(f)
    pred = load_table(os.path.join(a.dir, "predictions"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return

    lam2, lam3 = meta["lam2"], meta["lam3"]
    print(f"保存済みの予測を使います（{meta['races']:,}レース / "
          f"{meta['created']} 作成）")
    print(f"学習に使ったレース数: {meta['models']}\n")

    print("配当を推定しています…", flush=True)
    odds_cache = estimate_odds(pred, lam2, lam3)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    mins = [float(x) for x in a.min_odds.split(",") if x.strip()]
    maxs = [float(x) for x in a.max_odds.split(",")] if a.max_odds else [None]
    combs = [float(x) for x in a.min_comb.split(",")]
    pts = [int(x) for x in a.points.split(",")]
    evs = [float(x) for x in a.min_ev.split(",")] if a.min_ev else [None]

    best = []
    for model in models:
        if f"p_top3_{model}" not in pred.columns:
            print(f"\n{model}: 保存されていません")
            continue
        print(f"\n── {model} ──")
        for k, mn, mx, mc, ev in itertools.product(pts, mins, maxs, combs, evs):
            lbl = f"{k}点" + (f" 配当{mn:.0f}倍以上" if mn > 1 else " 配当制限なし")
            if ev:
                lbl += f" 期待値{ev*100:.0f}%以上"
            if mx:
                lbl += f"{mx:.0f}倍以下"
            if mc:
                lbl += f" 合成{mc:g}倍以上"
            r = report(run(pred, pays, model, k, mn, mx, mc, lam2, lam3,
                           odds_cache, ev), lbl)
            if r:
                r["model"] = model
                best.append(r)

    if best:
        top = sorted(best, key=lambda x: -x["roi"])[:5]
        print(f"\n{'='*72}")
        print("回収率の上位5件")
        for r in top:
            print(f"  {r['model']:<8}{r['label']:<34}"
                  f"回収 {r['roi']:>6.1f}%  下限 {r['lo']:>5.0f}%  {r['n']}R")
        print("\n  下限が100%を超えていなければ、その数字はまだ運の範囲です。")
        print("  条件を多数試して一番良いものを選ぶこと自体が過学習を生みます。")


if __name__ == "__main__":
    main()
