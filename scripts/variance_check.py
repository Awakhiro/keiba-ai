"""
回収率がまぐれかどうかを調べる。

高い回収率が「数本の高配当」で作られていないかを、
  ・的中1本ごとの払戻の分布（最大の1本が全体の何割を占めるか）
  ・ブートストラップ法による回収率の信頼区間
  ・最大の的中を除いたときの回収率
から確認する。信頼区間の下限が100%を割るなら、その数字はまだ信用できない。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists                # noqa: E402
from src.netkeiba import to_model_schema                # noqa: E402
from src.features import build_features                 # noqa: E402
from src.backtest import holdout, evaluate, _win_probs_from_model  # noqa: E402
from src.exotics import build_bets, market_win_probs, fit_lambdas  # noqa: E402

KEY = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}
POINTS = {"馬連": 6, "馬単": 8, "3連複": 10, "3連単": 16}


def per_race_returns(pred, detail, pays, bet_type, model, grades, lam2, lam3):
    """レースごとの回収（1点100円あたり）を並べて返す。"""
    grade_map = detail["grade"].to_dict()
    pay_map = pays.set_index("race_id").to_dict("index")
    n_pts = POINTS[bet_type]
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        if grade_map.get(race_id) not in grades:
            continue
        g = g.reset_index(drop=True)
        fin = g["finish_pos"].to_numpy()
        try:
            t = [int(np.where(fin == k)[0][0]) for k in (1, 2, 3)]
        except IndexError:
            continue
        real = pay_map.get(race_id, {}).get(f"payout_{KEY[bet_type]}")
        if real is None or not np.isfinite(real):
            continue
        truth = {"馬連": tuple(sorted(t[:2])), "馬単": tuple(t[:2]),
                 "3連複": tuple(sorted(t)), "3連単": tuple(t)}[bet_type]
        s = _win_probs_from_model(g, model)
        bets = build_bets(s, market_win_probs(g["odds_prev_win"].to_numpy()),
                          bet_type, max_points=n_pts,
                          strategy="hit" if model == "A" else "ev",
                          lam2=lam2, lam3=lam3)
        if bets.empty:
            continue
        hit = truth in set(bets["combo"])
        rows.append({"race_id": race_id, "hit": int(hit),
                     "stake": 100 * len(bets),
                     "payout": float(real) if hit else 0.0})
    return pd.DataFrame(rows)


def report(df, label, n_boot=5000, seed=0):
    if df.empty:
        print(f"{label}: データなし")
        return
    stake, payout = df["stake"].to_numpy(float), df["payout"].to_numpy(float)
    total = payout.sum() / stake.sum() * 100
    hits = payout[payout > 0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    boots = payout[idx].sum(1) / stake[idx].sum(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n── {label} ──")
    print(f"  レース数 {len(df)} / 的中 {len(hits)}本 ({len(hits)/len(df)*100:.1f}%)")
    print(f"  回収率 {total:.1f}%   95%信頼区間 {lo:.0f}〜{hi:.0f}%")
    if len(hits):
        share = hits.max() / payout.sum() * 100
        print(f"  最高配当 {hits.max():,.0f}円（払戻総額の {share:.0f}%）"
              f" / 中央値 {np.median(hits):,.0f}円")
        if len(hits) > 1:
            order = np.argsort(payout)[::-1]
            for drop in (1, 3):
                if len(hits) > drop:
                    keep = np.ones(len(df), bool); keep[order[:drop]] = False
                    r = payout[keep].sum() / stake[keep].sum() * 100
                    print(f"  上位{drop}本を除くと {r:.1f}%")
    verdict = ("信頼区間の下限が100%を超えています。再現性を別期間でも確認する価値があります。"
               if lo > 100 else
               "信頼区間が100%をまたぐので、この回収率はまだ運の範囲です。")
    print(f"  → {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-08-01")
    ap.add_argument("--bets", default="3連単,馬連,3連複")
    ap.add_argument("--models", default="A")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return

    feat = build_features(data)
    pred, conf, _ = holdout(feat, a.test_start, verbose=False)
    _s, _o, _grader, detail = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)

    for bt in a.bets.split(","):
        for m in a.models.split(","):
            for label, grades in [("S級のみ", {"S"}), ("A級以上", {"S", "A"})]:
                df = per_race_returns(pred, detail, pays, bt.strip(), m.strip(),
                                      grades, lam2, lam3)
                report(df, f"{bt.strip()} モデル{m.strip()} {label}")


if __name__ == "__main__":
    main()
