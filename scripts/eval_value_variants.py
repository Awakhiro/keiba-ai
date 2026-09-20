"""
妙味モデルの方針を3つ試して比べる。

    A 外れ予測型
        「的中率モデルが外れそうなレース」を学習で当てにいく。
        外れる理由は市場との食い違いではなく、能力推定の不確かさにある、
        という考え方。外れそうと予測したレースで高配当を狙う。

    B 高配当学習型
        馬連が一定倍率以上だったレースだけを学習に使い、
        「荒れるレースでの好走馬」に特化したモデルを作る。

    C 合成オッズ条件型
        的中率モデルと同じ作りのまま、合成オッズの下限だけ引き上げる。
        当てにいく姿勢は変えず、配当の見合う買い目だけを選ぶ。

いずれも「的中率モデルが外した60%のレースで当てる」ことを狙う。
最後に、各方針を使い分けた場合の回収率を出す。

    python scripts/eval_value_variants.py --test-start 2026-01-01
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
from src.schema import feature_columns                     # noqa: E402
from src.backtest import holdout, evaluate                 # noqa: E402
from src.exotics import fit_lambdas                        # noqa: E402
from src.recommend import recommend, recommend_for_race    # noqa: E402
import src.recommend as R                                  # noqa: E402

KEY = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False


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
    """買い目の結果を確定させる。"""
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
    return {"券種": bt, "stake": 100.0 * rec["点数"],
            "payout": float(real) if hit else 0.0, "hit": int(hit)}


def roi(df):
    if df is None or len(df) == 0:
        return 0.0
    return df["payout"].sum() / df["stake"].sum() * 100


def boot(df, n=3000, seed=0):
    if df is None or len(df) == 0:
        return (0.0, 0.0)
    s, p = df["stake"].to_numpy(float), df["payout"].to_numpy(float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), size=(n, len(s)))
    b = p[i].sum(1) / s[i].sum(1) * 100
    return tuple(np.percentile(b, [2.5, 97.5]))


def show(df, label):
    if df is None or len(df) == 0:
        print(f"  {label:<26} 対象なし")
        return 0.0
    lo, hi = boot(df)
    print(f"  {label:<26} {len(df):>5}レース  的中 {df['hit'].mean()*100:>5.1f}%  "
          f"回収 {roi(df):>6.1f}%  区間 {lo:>5.0f}〜{hi:>5.0f}%")
    return roi(df)


# ------------------------------------------------------------------ 方針A
def build_miss_model(pred, pays, lam2, lam3, train_frac=0.5):
    """
    的中率モデルが外れるレースを予測する。
    検証期間の前半で「外れたか」を学習し、後半で使う。
    """
    pay_map = pays.set_index("race_id").to_dict("index")
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        rec = recommend_for_race(g, mode="hit", lam2=lam2, lam3=lam3)
        r = settle(g, rec, pay_map, race_id)
        if r is None:
            continue
        rows.append({"race_id": race_id, "date": g["date"].iloc[0],
                     "miss": 1 - r["hit"]})
    lab = pd.DataFrame(rows)
    if lab.empty:
        return None, None

    # レース単位の特徴量（出走馬の統計を要約）
    num = [c for c in feature_columns(pred, use_odds=True)]
    agg = (pred.groupby("race_id")[num]
           .agg(["mean", "std", "max"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    X = lab.merge(agg, on="race_id", how="left").sort_values("date")

    cut = X["date"].quantile(train_frac)
    tr, te = X[X["date"] < cut], X[X["date"] >= cut]
    cols = [c for c in X.columns
            if c not in ("race_id", "date", "miss") and X[c].dtype.kind in "if"]
    if len(tr) < 200 or len(te) < 100:
        return None, None

    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=31, min_child_samples=40, verbose=-1)
        m.fit(tr[cols].fillna(0), tr["miss"])
        p = m.predict_proba(te[cols].fillna(0))[:, 1]
    else:
        m = HistGradientBoostingClassifier(max_iter=300)
        m.fit(tr[cols].fillna(0).to_numpy(), tr["miss"].to_numpy())
        p = m.predict_proba(te[cols].fillna(0).to_numpy())[:, 1]

    te = te.copy()
    te["miss_prob"] = p
    auc = _auc(te["miss"].to_numpy(), p)
    print(f"  外れ予測の精度 AUC={auc:.3f}"
          f"（0.5なら予測できていない / 0.6超で使える見込み）")
    return te.set_index("race_id")["miss_prob"], cut


def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ------------------------------------------------------------------ 方針C
def run_with_min_odds(pred, pays, lam2, lam3, min_odds, races=None):
    """合成オッズの下限を変えて、的中率モデルと同じ作りで買う。"""
    pay_map = pays.set_index("race_id").to_dict("index")
    old = R.MIN_COMBINED_ODDS
    R.MIN_COMBINED_ODDS = min_odds
    rows = []
    try:
        for race_id, g in pred.groupby("race_id", sort=False):
            if races is not None and race_id not in races:
                continue
            g = g.reset_index(drop=True)
            rec = recommend_for_race(g, mode="hit", lam2=lam2, lam3=lam3)
            r = settle(g, rec, pay_map, race_id)
            if r:
                rows.append({"race_id": race_id, **r})
    finally:
        R.MIN_COMBINED_ODDS = old
    return pd.DataFrame(rows)


def run_mode(pred, pays, lam2, lam3, mode, races=None):
    pay_map = pays.set_index("race_id").to_dict("index")
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        if races is not None and race_id not in races:
            continue
        g = g.reset_index(drop=True)
        rec = recommend_for_race(g, mode=mode, lam2=lam2, lam3=lam3)
        r = settle(g, rec, pay_map, race_id)
        if r:
            rows.append({"race_id": race_id, **r})
    return pd.DataFrame(rows)


def run_high_odds_model(data, pays, test_start, min_odds, lam2, lam3):
    """
    馬連が min_odds 倍以上だったレースだけを学習に使い、
    「荒れるレースでの好走馬」に特化したモデルで買う。
    """
    from src.models import RaceProbModel
    from src.value import add_value_columns

    pay_map = pays.set_index("race_id").to_dict("index")
    # 馬連配当が高かったレースを抽出（払戻は円なので 100 で割って倍率に）
    hot = {rid for rid, v in pay_map.items()
           if v.get("payout_umaren") and v["payout_umaren"] / 100.0 >= min_odds}
    feat = build_features(data)
    train = feat[(feat["date"] < pd.Timestamp(test_start))
                 & (feat["race_id"].isin(hot))]
    test = feat[feat["date"] >= pd.Timestamp(test_start)]
    if len(train) < 3000 or test.empty:
        return None

    print(f"    学習に使うレース: {train['race_id'].nunique():,}"
          f"（全体の一部のみ）", flush=True)
    ma = RaceProbModel("A_荒れ特化", "is_top3", use_odds=False, expected_sum=3.0)
    mb = RaceProbModel("B_荒れ特化", "is_win", use_odds=False, expected_sum=1.0)
    ma.fit(train); mb.fit(train)

    t = test.copy()
    t["p_top3"] = ma.predict(t)
    t["p_win_pure"] = mb.predict(t)
    t = add_value_columns(t)
    t["p_win"] = t["p_blend"]

    rows = []
    for race_id, g in t.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        rec = recommend_for_race(g, mode="hit", lam2=lam2, lam3=lam3)
        r = settle(g, rec, pay_map, race_id)
        if r:
            rows.append({"race_id": race_id, **r})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 補完性の評価
def pair_ceiling(base: pd.DataFrame, other: pd.DataFrame, label: str,
                 quiet=False):
    """
    的中率モデルと相方を組み合わせたときの上限を出す。

    単独の成績が低くても、当たるレースが重ならなければ
    「毎回 良い方を選べた場合」の値は高くなる。そこを測る。
    """
    if base is None or other is None or base.empty or other.empty:
        return None
    b = base.set_index("race_id")
    o = other.set_index("race_id")
    common = b.index.intersection(o.index)
    if len(common) < 50:
        return None
    b, o = b.loc[common], o.loc[common]

    # 各レースで利益の大きい方を選べた場合
    prof_b = b["payout"] - b["stake"]
    prof_o = o["payout"] - o["stake"]
    take_o = (prof_o > prof_b).to_numpy()
    pay = np.where(take_o, o["payout"], b["payout"])
    stake = np.where(take_o, o["stake"], b["stake"])
    ceiling = pay.sum() / stake.sum() * 100

    # ランダムに選んだ場合（選択精度ゼロの目安）
    rng = np.random.default_rng(0)
    coin = rng.integers(0, 2, len(common)).astype(bool)
    rnd = (np.where(coin, o["payout"], b["payout"]).sum()
           / np.where(coin, o["stake"], b["stake"]).sum() * 100)

    only_o = ((o["hit"] == 1) & (b["hit"] == 0)).sum()
    only_b = ((b["hit"] == 1) & (o["hit"] == 0)).sum()
    both = ((b["hit"] == 1) & (o["hit"] == 1)).sum()

    if not quiet:
        print(f"  {label:<24} 上限 {ceiling:>6.1f}%  "
              f"（相方だけ的中 {only_o:>4} / 的中率だけ {only_b:>4} / 両方 {both:>3}）"
              f"  ランダム {rnd:>5.1f}%")
    return {"label": label, "ceiling": ceiling, "random": rnd,
            "only_other": int(only_o), "n": len(common)}


def compare_ceilings(pred, pays, data, lam2, lam3, test_start, high_odds):
    """すべての候補について、的中率モデルと組んだときの上限を並べる。"""
    base = run_mode(pred, pays, lam2, lam3, "hit")
    print(f"\n{'='*70}")
    print("【組み合わせたときの上限】各レースで利益の大きい方を選べた場合")
    print(f"  的中率モデル単独は {roi(base):.1f}%")
    print()

    rows = []
    r = pair_ceiling(base, run_mode(pred, pays, lam2, lam3, "ev"),
                     "いまの妙味モデル")
    if r: rows.append(r)

    for mo in (4.0, 5.0, 7.0, 10.0):
        r = pair_ceiling(base, run_with_min_odds(pred, pays, lam2, lam3, mo),
                         f"C: 合成オッズ{mo:>4.1f}倍")
        if r: rows.append(r)

    for th in (high_odds, high_odds * 2):
        df = run_high_odds_model(data, pays, test_start, th, lam2, lam3)
        r = pair_ceiling(base, df, f"B: 馬連{th:>3.0f}倍以上で学習")
        if r: rows.append(r)

    if rows:
        best = max(rows, key=lambda x: x["ceiling"])
        print(f"\n  上限が最も高いのは「{best['label'].strip()}」で {best['ceiling']:.1f}%")
        print(f"  ただしランダム選択だと {best['random']:.1f}% なので、")
        print(f"  上限と ランダム の差 {best['ceiling'] - best['random']:.1f} "
              f"ポイントが、選択精度で取りにいける幅です。")
        print("  この幅が小さい相方は、いくら見分けても意味がありません。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--high-odds", type=float, default=20.0,
                    help="方針B で学習に使う馬連配当の下限（倍）")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return

    pred, conf, _ = holdout(build_features(data), a.test_start)
    _s, _o, _g, detail = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)

    print("\n【基準】")
    base = run_mode(pred, pays, lam2, lam3, "hit")
    show(base, "的中率モデル")
    show(run_mode(pred, pays, lam2, lam3, "ev"), "いまの妙味モデル")

    # -------- 方針C: 合成オッズの下限を変える --------
    print("\n【方針C】合成オッズの下限を上げる（作りは的中率モデルのまま）")
    for mo in (2.6, 4.0, 5.0, 7.0, 10.0, 15.0):
        df = run_with_min_odds(pred, pays, lam2, lam3, mo)
        show(df, f"下限 {mo:>4.1f}倍")

    # -------- 方針A: 外れ予測 --------
    print("\n【方針A】的中率モデルが外れそうなレースを予測して、そこで妙味を使う")
    miss_prob, cut = build_miss_model(pred, pays, lam2, lam3)
    if miss_prob is None:
        print("  データ不足で評価できません")
    else:
        late = pred[pred["date"] >= cut]
        base_late = run_mode(late, pays, lam2, lam3, "hit")
        show(base_late, "的中率モデル（後半のみ）")
        for q in (0.3, 0.5, 0.7):
            th = miss_prob.quantile(q)
            risky = set(miss_prob[miss_prob >= th].index)
            safe = set(miss_prob[miss_prob < th].index)
            hit_part = run_mode(late, pays, lam2, lam3, "hit", races=safe)
            ev_part = run_mode(late, pays, lam2, lam3, "ev", races=risky)
            mix = pd.concat([hit_part, ev_part], ignore_index=True)
            show(mix, f"外れ確率 上位{int((1-q)*100)}%で妙味")

    # -------- 方針B: 高配当レースだけで学習 --------
    print("\n【方針B】馬連が一定倍率以上だったレースだけで学習し直す")
    print("  荒れたレースでの好走馬に特化したモデルを作り、その予測で買う。")
    for th in (a.high_odds, a.high_odds * 2):
        df = run_high_odds_model(data, pays, a.test_start, th, lam2, lam3)
        if df is None:
            print(f"  馬連{th:>4.0f}倍以上: 学習データ不足")
            continue
        show(df, f"馬連{th:>4.0f}倍以上で学習")

    # -------- 組み合わせたときの上限 --------
    compare_ceilings(pred, pays, data, lam2, lam3, a.test_start, a.high_odds)


if __name__ == "__main__":
    main()
