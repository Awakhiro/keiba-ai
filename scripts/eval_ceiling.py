"""
「毎回当たった方のモデルを選べたら何%になるか」を測る。

目標は、レースごとに的中率モデルと妙味モデルのどちらかを選んで
全体の回収率を100%超にすること。そこで最初に上限を測る。

    完全予知  … 結果を見てから良い方を選んだ場合（到達できない理想値）
    実際      … 事前の指標で選んだ場合
    単独      … どちらか一方だけを使い続けた場合

完全予知が100%を割るなら、選択精度をどれだけ上げても目標には届かない。
その場合は選択の問題ではなく、買い目そのものを変える必要がある。

完全予知が十分高ければ、あとは「どちらが当たるか」を事前に見分ける指標の問題になる。
このスクリプトは、その見分けに使えそうな特徴を洗い出して、
どれだけ予測できるかも測る。

    python scripts/eval_ceiling.py --test-start 2026-01-01
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists                   # noqa: E402
from src.netkeiba import to_model_schema                   # noqa: E402
from src.features import build_features                    # noqa: E402
from src.backtest import holdout, evaluate                 # noqa: E402
from src.exotics import fit_lambdas                        # noqa: E402
from src.recommend import recommend_for_race               # noqa: E402
from src.confidence import value_metrics, value_score      # noqa: E402

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


def collect(pred, pays, lam2, lam3):
    """レースごとに、両モデルの結果を並べて記録する。"""
    pay_map = pays.set_index("race_id").to_dict("index")
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        t = truth_of(g)
        if t is None:
            continue
        rec = {}
        ok = True
        for mode, name in (("hit", "的中率"), ("ev", "妙味")):
            r = recommend_for_race(g, mode=mode, lam2=lam2, lam3=lam3)
            if r.get("見送り"):
                ok = False
                break
            bt = r["券種"]
            real = pay_map.get(race_id, {}).get(f"payout_{KEY[bt]}")
            if real is None or not np.isfinite(real):
                ok = False
                break
            want = t[bt]
            hit = any((sorted(c) if bt in ("馬連", "3連複") else c) == want
                      for c in r["買い目"])
            rec[name] = {"券種": bt, "点数": r["点数"],
                         "stake": 100.0 * r["点数"],
                         "payout": float(real) if hit else 0.0,
                         "hit": int(hit),
                         "p": r["的中確率"], "ret": r["期待回収率"],
                         "odds": r["合成オッズ"]}
        if not ok:
            continue
        rows.append({
            "race_id": race_id,
            **{f"{k}_{f}": v[f] for k, v in rec.items()
               for f in ("券種", "stake", "payout", "hit", "p", "ret", "odds")},
        })
    return pd.DataFrame(rows)


def roi(stake, payout):
    s, p = np.asarray(stake, float), np.asarray(payout, float)
    return p.sum() / s.sum() * 100 if s.sum() else 0.0


def boot_lo(stake, payout, n=3000, seed=0):
    s, p = np.asarray(stake, float), np.asarray(payout, float)
    if len(s) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), size=(n, len(s)))
    return float(np.percentile(p[i].sum(1) / s[i].sum(1) * 100, 2.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return

    pred, conf, _ = holdout(build_features(data), a.test_start)
    _s, _o, _g, detail = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)

    print("\n各レースで両モデルの買い目を作っています…", flush=True)
    d = collect(pred, pays, lam2, lam3)
    if d.empty:
        print("対象レースがありません。"); return
    n = len(d)

    # ---------- 上限と下限 ----------
    best_pay = np.maximum(d["的中率_payout"], d["妙味_payout"])
    best_stake = np.where(d["的中率_payout"] >= d["妙味_payout"],
                          d["的中率_stake"], d["妙味_stake"])
    worst_pay = np.minimum(d["的中率_payout"], d["妙味_payout"])
    worst_stake = np.where(d["的中率_payout"] <= d["妙味_payout"],
                           d["的中率_stake"], d["妙味_stake"])
    coin = np.random.default_rng(0).integers(0, 2, n).astype(bool)
    coin_pay = np.where(coin, d["的中率_payout"], d["妙味_payout"])
    coin_stake = np.where(coin, d["的中率_stake"], d["妙味_stake"])

    print(f"\n対象 {n}レース")
    print("=" * 56)
    print(f"{'完全予知（毎回 当たった方）':<28} {roi(best_stake, best_pay):>7.1f}%"
          f"   ← これが上限")
    print(f"{'的中率モデルだけ':<28} {roi(d['的中率_stake'], d['的中率_payout']):>7.1f}%")
    print(f"{'妙味モデルだけ':<28} {roi(d['妙味_stake'], d['妙味_payout']):>7.1f}%")
    print(f"{'ランダムに選ぶ':<28} {roi(coin_stake, coin_pay):>7.1f}%")
    print(f"{'最悪（毎回 外れた方）':<28} {roi(worst_stake, worst_pay):>7.1f}%")
    print("=" * 56)

    ceiling = roi(best_stake, best_pay)
    if ceiling < 100:
        print("\n完全予知でも100%に届きません。")
        print("結果を知っていても勝てないということは、選択の問題ではありません。")
        print("買い目そのもの（券種・点数・選び方）を変える必要があります。")
    else:
        print(f"\n完全予知なら {ceiling:.0f}% です。選択精度を上げる余地があります。")

    # ---------- どちらが当たったか ----------
    both = (d["的中率_hit"] == 1) & (d["妙味_hit"] == 1)
    only_h = (d["的中率_hit"] == 1) & (d["妙味_hit"] == 0)
    only_v = (d["的中率_hit"] == 0) & (d["妙味_hit"] == 1)
    none = (d["的中率_hit"] == 0) & (d["妙味_hit"] == 0)
    print(f"\n【的中の内訳】")
    print(f"  両方的中 {both.sum():>4} ({both.mean()*100:>4.1f}%)   "
          f"的中率だけ {only_h.sum():>4} ({only_h.mean()*100:>4.1f}%)")
    print(f"  妙味だけ {only_v.sum():>4} ({only_v.mean()*100:>4.1f}%)   "
          f"両方外れ {none.sum():>4} ({none.mean()*100:>4.1f}%)")
    print(f"\n  妙味でしか取れないレースが {only_v.mean()*100:.1f}% あります。"
          f"ここを見分けられれば伸びます。")
    if only_v.sum():
        print(f"  そのレースの妙味側の平均払戻: "
              f"{d.loc[only_v, '妙味_payout'].mean():,.0f}円")

    # ---------- 見分けられるか ----------
    vm = value_metrics(pred)
    vm["value_score"] = value_score(vm)
    feat = (conf.set_index("race_id")
            .join(vm.set_index("race_id"), how="inner")
            .join(d.set_index("race_id"), how="inner"))
    feat["妙味が有利"] = (feat["妙味_payout"] - feat["妙味_stake"] >
                        feat["的中率_payout"] - feat["的中率_stake"]).astype(int)

    cols = ["conf_score", "value_score", "a_top1_p", "a_margin", "a_entropy",
            "b_top3_share", "model_agree", "max_edge", "top3_edge",
            "fav_overrated", "rank_disagree", "market_entropy",
            "的中率_p", "妙味_p", "妙味_ret", "妙味_odds"]
    cols = [c for c in cols if c in feat.columns]
    print(f"\n【妙味が有利だったレースと、そうでないレースの違い】")
    print(f"{'指標':<16}{'妙味有利':>10}{'的中率有利':>11}{'差':>9}")
    diffs = []
    for c in cols:
        a_ = feat.loc[feat["妙味が有利"] == 1, c].astype(float)
        b_ = feat.loc[feat["妙味が有利"] == 0, c].astype(float)
        if a_.empty or b_.empty:
            continue
        sd = feat[c].astype(float).std()
        gap = (a_.mean() - b_.mean()) / sd if sd > 1e-9 else 0.0
        diffs.append((abs(gap), c, a_.mean(), b_.mean(), gap))
    for _, c, am, bm, gap in sorted(diffs, reverse=True)[:10]:
        print(f"{c:<16}{am:>10.3f}{bm:>11.3f}{gap:>9.2f}")
    print("\n  差は標準偏差で割った値。0.2を超えると見分けに使える見込みがあります。")
    print(f"  最大でも {max(x[0] for x in diffs):.2f} でした。" if diffs else "")


if __name__ == "__main__":
    main()
