"""
レースごとの推奨買い目が実際に機能するか検証する。

    python scripts/eval_recommend.py --test-start 2026-01-01

実際の払戻で回収率を出し、ブートストラップ法で信頼区間を付ける。
券種を固定した場合とも比べ、「レースごとに選ぶ」ことに意味があるかを見る。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists              # noqa: E402
from src.netkeiba import to_model_schema              # noqa: E402
from src.features import build_features               # noqa: E402
from src.backtest import holdout, evaluate            # noqa: E402
from src.exotics import fit_lambdas                   # noqa: E402
from src.recommend import recommend_for_race, BET_TYPES  # noqa: E402

KEY = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}


def truth_of(g):
    fin = g["finish_pos"].to_numpy()
    try:
        i = [int(np.where(fin == k)[0][0]) for k in (1, 2, 3)]
    except IndexError:
        return None
    no = g["horse_no"].to_numpy()
    a, b, c = (int(no[x]) for x in i)
    return {"馬連": sorted([a, b]), "馬単": [a, b],
            "3連複": sorted([a, b, c]), "3連単": [a, b, c]}


def run(pred, pays, detail, mode, lam2, lam3, grades=None, fixed=None):
    """fixed に券種を渡すとその券種だけで買い目を作る。None ならレースごとに選ぶ。"""
    pay_map = pays.set_index("race_id").to_dict("index")
    grade_map = detail["grade"].to_dict()
    types = (fixed,) if fixed else BET_TYPES
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        if grades and grade_map.get(race_id) not in grades:
            continue
        g = g.reset_index(drop=True)
        t = truth_of(g)
        if t is None:
            continue
        rec = recommend_for_race(g, mode=mode, lam2=lam2, lam3=lam3,
                                 bet_types=types)
        if rec.get("見送り"):
            rows.append({"race_id": race_id, "券種": "見送り", "点数": 0,
                         "stake": 0.0, "payout": 0.0, "hit": 0})
            continue
        bt = rec["券種"]
        real = pay_map.get(race_id, {}).get(f"payout_{KEY[bt]}")
        if real is None or not np.isfinite(real):
            continue
        want = t[bt]
        hit = any((sorted(c) if bt in ("馬連", "3連複") else c) == want
                  for c in rec["買い目"])
        rows.append({"race_id": race_id, "券種": bt, "点数": rec["点数"],
                     "stake": 100.0 * rec["点数"],
                     "payout": float(real) if hit else 0.0, "hit": int(hit)})
    return pd.DataFrame(rows, columns=["race_id", "券種", "点数", "stake", "payout", "hit"])


def report(df, label, n_boot=4000, seed=0):
    if df.empty:
        print(f"\n{label}: 対象レースがありません")
        return
    bet = df[df["券種"] != "見送り"]
    skip = len(df) - len(bet)
    if bet.empty:
        print(f"\n{label}: 全て見送り（{skip}レース）")
        return
    stake, payout = bet["stake"].to_numpy(), bet["payout"].to_numpy()
    roi = payout.sum() / stake.sum() * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bet), size=(n_boot, len(bet)))
    boots = payout[idx].sum(1) / stake[idx].sum(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n── {label} ──")
    print(f"  購入 {len(bet)}レース（見送り {skip}） / 平均 {bet['点数'].mean():.1f}点")
    print(f"  的中率 {bet['hit'].mean()*100:.1f}%  回収率 {roi:.1f}%  "
          f"95%区間 {lo:.0f}〜{hi:.0f}%")
    mix = bet["券種"].value_counts().to_dict()
    print(f"  券種の内訳: {mix}")
    print(f"  → {'下限が100%超' if lo > 100 else 'まだ運の範囲'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--grades", default="S,A")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return

    pred, conf, _ = holdout(build_features(data), a.test_start)
    _s, _o, _g, detail = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)
    grades = set(a.grades.split(",")) if a.grades else None
    print(f"対象グレード: {sorted(grades) if grades else '全て'}")

    for mode, name in (("hit", "的中率優先"), ("ev", "期待値優先")):
        report(run(pred, pays, detail, mode, lam2, lam3, grades),
               f"{name} レースごとに券種を選ぶ")
        for bt in BET_TYPES:
            report(run(pred, pays, detail, mode, lam2, lam3, grades, fixed=bt),
                   f"{name} {bt}に固定")


if __name__ == "__main__":
    main()
