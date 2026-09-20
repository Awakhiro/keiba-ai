"""
人気薄が絡んだレースだけで学習したモデルを試す。

    python scripts/eval_upset.py --test-start 2026-01-01

いまの妙味モデルは「市場との乖離」で穴を探していたが、検証では
乖離が大きいレースほど妙味が無いという逆の結果が出た。
そこで市場との比較をやめ、**穴が来たレースだけを学習に使う**形に変える。

比べる条件:
    人気 … 2着以内に N番人気以下が来たレース
    配当 … 馬連が N倍以上だったレース（既存の高配当特化）

どちらも「荒れるレースでの好走馬の特徴」を学ぶが、
人気で切る方が馬連の的中に直結する。

見るべきは単独の回収率より、**的中率モデルとの補完性**。
「こちらだけ当たったレース」が多いほど、組み合わせたときの伸びしろが大きい。
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
from src.models import RaceProbModel                       # noqa: E402
from src.backtest import holdout, evaluate                 # noqa: E402
from src.exotics import fit_lambdas                        # noqa: E402
from src.value import add_value_columns                    # noqa: E402
from src.recommend import recommend_for_race               # noqa: E402

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


def settle(g, rec, pay_map, race_id):
    if rec is None or rec.get("見送り"):
        return None
    t = truth_of(g)
    if t is None:
        return None
    bt = rec["券種"]
    real = pay_map.get(race_id, {}).get(f"payout_{KEY[bt]}")
    if real is None or not np.isfinite(real):
        return None
    want = t[bt]
    hit = any((sorted(c) if bt in ("馬連", "3連複") else c) == want
              for c in rec["買い目"])
    return {"stake": 100.0 * rec["点数"],
            "payout": float(real) if hit else 0.0, "hit": int(hit)}


def roi(df):
    return df["payout"].sum() / df["stake"].sum() * 100 if len(df) else 0.0


def boot(df, n=3000, seed=0):
    if not len(df):
        return (0.0, 0.0)
    s, p = df["stake"].to_numpy(float), df["payout"].to_numpy(float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), size=(n, len(s)))
    b = p[i].sum(1) / s[i].sum(1) * 100
    return tuple(np.percentile(b, [2.5, 97.5]))


def upset_races(feat, min_pop, max_place=2):
    """
    max_place 着以内に min_pop 番人気以下が来たレースを選ぶ。
    人気は確定オッズの順位から求める（過去レースなので取得済み）。
    """
    hit = set()
    for race_id, g in feat.groupby("race_id", sort=False):
        pop = None
        if "popularity_final" in g.columns:
            v = g["popularity_final"]
            # 全馬が同じ値なら壊れているので使わない（1着から順に振られるはず）
            if v.notna().all() and v.nunique() > 1:
                pop = v.to_numpy()
        if pop is None:
            o = g["odds_win_final"] if "odds_win_final" in g.columns \
                else g["odds_prev_win"]
            pop = o.rank(method="min").to_numpy()
        fin = g["finish_pos"].to_numpy()
        m = (fin <= max_place) & ~np.isnan(fin)
        if m.any() and np.nanmax(pop[m]) >= min_pop:
            hit.add(race_id)
    return hit


def build_and_run(feat, pays, test_start, races, label, lam2, lam3, pay_map):
    """指定したレース集合だけで学習し、検証期間で買ってみる。"""
    train = feat[(feat["date"] < pd.Timestamp(test_start))
                 & (feat["race_id"].isin(races))]
    test = feat[feat["date"] >= pd.Timestamp(test_start)]
    n_tr = train["race_id"].nunique()
    if n_tr < 400 or test.empty:
        print(f"  {label:<26} 学習レース {n_tr} 件で不足")
        return None

    ma = RaceProbModel("A", "is_top3", use_odds=False, expected_sum=3.0)
    mb = RaceProbModel("B", "is_win", use_odds=False, expected_sum=1.0)
    ma.fit(train); mb.fit(train)

    t = test.copy()
    t["p_top3"] = ma.predict(t)
    t["p_win_pure"] = mb.predict(t)
    t = add_value_columns(t)
    t["p_win"] = t["p_blend"]

    rows = []
    for race_id, g in t.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        r = settle(g, recommend_for_race(g, mode="hit", lam2=lam2, lam3=lam3),
                   pay_map, race_id)
        if r:
            rows.append({"race_id": race_id, **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    lo, hi = boot(df)
    print(f"  {label:<26} 学習{n_tr:>5}R  {len(df):>5}レース  "
          f"的中 {df['hit'].mean()*100:>5.1f}%  回収 {roi(df):>6.1f}%  "
          f"区間 {lo:>4.0f}〜{hi:>4.0f}%")
    return df


def complement(base, other, label):
    """的中率モデルとの補完性。"""
    if other is None or other.empty:
        return None
    b = base.set_index("race_id")
    o = other.set_index("race_id")
    common = b.index.intersection(o.index)
    if len(common) < 50:
        return None
    b, o = b.loc[common], o.loc[common]
    only_o = int(((o["hit"] == 1) & (b["hit"] == 0)).sum())
    only_b = int(((b["hit"] == 1) & (o["hit"] == 0)).sum())
    take = (o["payout"] - o["stake"]) > (b["payout"] - b["stake"])
    pay = np.where(take, o["payout"], b["payout"])
    st = np.where(take, o["stake"], b["stake"])
    ceil = pay.sum() / st.sum() * 100
    rng = np.random.default_rng(0)
    coin = rng.integers(0, 2, len(common)).astype(bool)
    rnd = (np.where(coin, o["payout"], b["payout"]).sum()
           / np.where(coin, o["stake"], b["stake"]).sum() * 100)
    print(f"  {label:<26} 上限 {ceil:>6.1f}%  こちらだけ的中 {only_o:>4}件  "
          f"的中率だけ {only_b:>4}件  ランダム {rnd:>5.1f}%")
    return {"label": label, "ceiling": ceil, "only": only_o, "random": rnd}


# ------------------------------------------------------------------ 配当の性質
def payout_profile(results, base, label_order=None):
    """
    的中したときの配当の分布を比べる。

    回収率が同じでも「安いのを高確率で当てる」のと
    「高いのを低確率で当てる」のでは性質がまったく違う。
    どちらが穴狙いなのかは、この分布で決まる。
    """
    print(f"\n{'='*78}")
    print("【的中したときの配当】どちらが穴狙いか")
    print(f"{'モデル':<14}{'的中率':>7}{'平均':>9}{'中央値':>9}{'最高':>10}"
          f"{'1000円未満':>11}{'1万円以上':>10}")
    rows = {"的中率モデル": base}
    rows.update(results)
    for k, df in rows.items():
        if df is None or df.empty:
            continue
        won = df.loc[df["hit"] == 1, "payout"]
        if won.empty:
            continue
        # 1点あたりの配当ではなく、実際の払戻額で見る
        cheap = (won < 1000).mean() * 100
        rich = (won >= 10000).mean() * 100
        print(f"{k:<14}{df['hit'].mean()*100:>6.1f}%{won.mean():>9,.0f}"
              f"{won.median():>9,.0f}{won.max():>10,.0f}"
              f"{cheap:>10.0f}%{rich:>9.0f}%")

    print("\n  平均より中央値を見てください。平均は少数の高配当に引っ張られます。")
    print("  中央値が高いほど、普段から高めの配当を取りにいくモデルです。")

    # 収支の安定性
    print(f"\n{'モデル':<14}{'最大連敗':>9}{'収支のぶれ':>12}{'利益の8割を作った的中数':>22}")
    for k, df in rows.items():
        if df is None or df.empty:
            continue
        prof = (df["payout"] - df["stake"]).to_numpy()
        # 最大連敗
        miss = (df["hit"] == 0).to_numpy()
        run = best = 0
        for m in miss:
            run = run + 1 if m else 0
            best = max(best, run)
        # 利益の8割を作った的中数
        won = np.sort(prof[prof > 0])[::-1]
        need = 0
        if won.sum() > 0:
            cum = np.cumsum(won) / won.sum()
            need = int(np.searchsorted(cum, 0.8) + 1)
        print(f"{k:<14}{best:>8}回{prof.std():>11,.0f}"
              f"{need:>15}/{int((df['hit']==1).sum())}件")
    print("\n  少数の的中が利益の大半を作っているモデルほど、"
          "実際に続けるのが難しくなります。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return
    pay_map = pays.set_index("race_id").to_dict("index")

    feat = build_features(data)
    pred, conf, _ = holdout(feat, a.test_start)
    _s, _o, _g, _d = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)

    # 基準
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        r = settle(g, recommend_for_race(g, mode="hit", lam2=lam2, lam3=lam3),
                   pay_map, race_id)
        if r:
            rows.append({"race_id": race_id, **r})
    base = pd.DataFrame(rows)
    lo, hi = boot(base)
    print(f"\n【基準】的中率モデル  {len(base)}レース  "
          f"的中 {base['hit'].mean()*100:.1f}%  回収 {roi(base):.1f}%  "
          f"区間 {lo:.0f}〜{hi:.0f}%")

    print("\n【人気で切る】2着以内に N番人気以下が来たレースだけで学習")
    results = {}
    for pop in (4, 5, 6, 8):
        races = upset_races(feat, pop, max_place=2)
        df = build_and_run(feat, pays, a.test_start, races,
                           f"2着内に{pop}番人気以下", lam2, lam3, pay_map)
        if df is not None:
            results[f"人気{pop}"] = df

    print("\n【3着以内で切る】より緩い条件")
    for pop in (5, 6):
        races = upset_races(feat, pop, max_place=3)
        df = build_and_run(feat, pays, a.test_start, races,
                           f"3着内に{pop}番人気以下", lam2, lam3, pay_map)
        if df is not None:
            results[f"3着{pop}"] = df

    print("\n【配当で切る】既存の高配当特化との比較")
    for th in (20, 40):
        hot = {rid for rid, v in pay_map.items()
               if v.get("payout_umaren") and v["payout_umaren"] / 100.0 >= th}
        df = build_and_run(feat, pays, a.test_start, hot,
                           f"馬連{th}倍以上", lam2, lam3, pay_map)
        if df is not None:
            results[f"配当{th}"] = df

    print(f"\n{'='*74}")
    print("【的中率モデルとの補完性】組み合わせたときの上限")
    comps = []
    for k, df in results.items():
        c = complement(base, df, k)
        if c:
            comps.append(c)
    if comps:
        best_c = max(comps, key=lambda x: x["ceiling"])
        best_o = max(comps, key=lambda x: x["only"])
        print(f"\n  上限が最大: {best_c['label']} ({best_c['ceiling']:.1f}%)")
        print(f"  補完件数が最大: {best_o['label']} ({best_o['only']}件)")
        print("  実用上は補完件数の多い方が選択指標を作りやすく、数字も安定します。")

    payout_profile(results, base)


if __name__ == "__main__":
    main()
