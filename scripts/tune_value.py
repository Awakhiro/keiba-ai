"""
妙味馬の抽出条件を探索する。

    python scripts/tune_value.py --test-start 2026-01-01

「能力の下限」「乖離の下限」「オッズ上限」「市場への寄せ具合」を総当たりし、
複勝と単勝の回収率を出す。点数が少ない設定は偶然で跳ねるので除外し、
上位の設定にはブートストラップ法の信頼区間を付ける。

324通りを試して一番良いものを選ぶ行為そのものが過学習を生む。
（コインを324枚投げれば、どれかは連続で表が出る）
そこで検証期間を前半と後半に割り、**前半で決めた条件を後半で確かめる**。
後半でも通用した条件だけが、使う価値のある条件。
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table                      # noqa: E402
from src.netkeiba import to_model_schema              # noqa: E402
from src.features import build_features               # noqa: E402
from src.backtest import holdout                      # noqa: E402
from src.value import (                               # noqa: E402
    add_value_columns, pick_value_horses, evaluate_value_picks, bootstrap_roi,
)


def sweep(pred, grid, min_picks=60):
    rows = []
    for w, min_top3, min_edge, max_odds, gap in itertools.product(*grid):
        d = add_value_columns(pred, w=w)
        picks = pick_value_horses(d, min_top3=min_top3, min_edge=min_edge,
                                  max_odds=max_odds, min_pop_gap=gap)
        if len(picks) < min_picks:
            continue
        r = evaluate_value_picks(picks)
        r.update({"寄せ": w, "能力下限": min_top3, "乖離下限": min_edge,
                  "odds上限": max_odds, "人気差": gap})
        rows.append(r)
    return pd.DataFrame(rows)


def baseline(pred):
    """比較用: 各レースで能力1位の馬を単純に買った場合。"""
    idx = pred.groupby("race_id")["p_top3"].idxmax()
    b = pred.loc[idx]
    n = len(b)
    out = {"点数": n,
           "複勝的中率": round(float(b["is_top3"].mean()) * 100, 1),
           "単勝的中率": round(float(b["is_win"].mean()) * 100, 1)}
    if "payout_place" in b.columns:
        out["複勝回収率"] = round(float((b["is_top3"] * b["payout_place"]).mean()), 1)
    if "payout_win" in b.columns:
        out["単勝回収率"] = round(float((b["is_win"] * b["payout_win"]).mean()), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--min-picks", type=int, default=60)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--split", type=float, default=0.6,
                    help="検証期間のうち、条件探索に使う割合（残りは答え合わせ）")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    feat = build_features(data)
    pred, _conf, _m = holdout(feat, a.test_start)

    # 期間を前半（条件探索）と後半（答え合わせ）に割る
    dates = np.sort(pred["date"].unique())
    cut = dates[int(len(dates) * a.split)]
    tune = pred[pred["date"] < cut]
    conf = pred[pred["date"] >= cut]
    print(f"\n条件探索 {pd.Timestamp(dates[0]).date()}〜{pd.Timestamp(cut).date()} "
          f"{tune['race_id'].nunique()}レース / "
          f"答え合わせ {pd.Timestamp(cut).date()}〜{pd.Timestamp(dates[-1]).date()} "
          f"{conf['race_id'].nunique()}レース")

    print("\n【比較の基準】能力1位の馬を全レースで買った場合（答え合わせ期間）")
    for k, v in baseline(conf).items():
        print(f"  {k}: {v}")

    grid = (
        [0.35, 0.45, 0.60],          # 市場への寄せ具合 w
        [0.20, 0.28, 0.35],          # 能力の下限（複勝圏確率）
        [1.15, 1.30, 1.50, 1.80],    # 乖離の下限
        [20.0, 40.0, 80.0],          # オッズ上限
        [0, 1, 2],                   # 人気順の差
    )
    print(f"\n{np.prod([len(g) for g in grid])}通りを前半で試します…")
    res = sweep(tune, grid, min_picks=a.min_picks)
    if res.empty:
        print(f"条件を満たす設定がありません（点数 {a.min_picks} 以上）。"
              "--min-picks を下げるか、検証期間を広げてください。")
        return

    cols = ["寄せ", "能力下限", "乖離下限", "odds上限", "人気差", "点数",
            "平均オッズ", "複勝的中率", "複勝回収率", "単勝的中率", "単勝回収率"]
    cols = [c for c in cols if c in res.columns]

    for key in ("複勝回収率", "単勝回収率"):
        if key not in res.columns:
            continue
        print(f"\n【前半での{key} 上位{a.top}件】")
        top = res.sort_values(key, ascending=False).head(a.top)
        print(top[cols].to_string(index=False))

        pay = "payout_place" if key.startswith("複勝") else "payout_win"
        hit = "is_top3" if key.startswith("複勝") else "is_win"
        print(f"\n  ── 後半での答え合わせ（{key}）──")
        print("  前半で良かった条件が後半でも通用するかどうか。ここが本番。")
        survived = 0
        for _, r in top.head(5).iterrows():
            d2 = add_value_columns(conf, w=r["寄せ"])
            p2 = pick_value_horses(d2, min_top3=r["能力下限"], min_edge=r["乖離下限"],
                                   max_odds=r["odds上限"], min_pop_gap=int(r["人気差"]))
            label = (f"寄せ{r['寄せ']} 能力{r['能力下限']} 乖離{r['乖離下限']} "
                     f"odds≤{r['odds上限']:.0f} 人気差{int(r['人気差'])}")
            if len(p2) < 10:
                print(f"    {label}: 後半は{len(p2)}点しかなく判定不能")
                continue
            ci = bootstrap_roi(p2, payout_col=pay, hit_col=hit)
            e2 = evaluate_value_picks(p2)
            ok = ci and ci["下限"] > 100
            survived += bool(ok)
            print(f"    {label}")
            print(f"      前半 {r[key]}%  →  後半 {ci['回収率']}%  "
                  f"({len(p2)}点 / 95%区間 {ci['下限']}〜{ci['上限']}%)"
                  f"  {'★通用した' if ok else '→ 崩れた'}")
        if survived == 0:
            print("    → 後半まで生き残った条件はありません。"
                  "前半の好成績は条件を探した副作用です。")

    os.makedirs("docs", exist_ok=True)
    res.sort_values("複勝回収率" if "複勝回収率" in res.columns else "点数",
                    ascending=False).to_csv("docs/value_sweep.csv", index=False)
    print("\n全結果: docs/value_sweep.csv")


if __name__ == "__main__":
    main()
