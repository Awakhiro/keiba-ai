"""
的中率モデルと高配当特化モデルの、どちらを使うかを学習で決める。

    python scripts/eval_selector.py --test-start 2026-01-01

考え方:
    2つのモデルは当たるレースが大きく異なる。毎回良い方を選べれば 120% 近くまで届く。
    そこで「このレースはどちらが勝つか」を、レースの性質から予測する。

    検証期間を前半と後半に割り、前半で学習して後半で試す。
    前半で良くても後半で崩れるなら、それは条件を探した副作用でしかない。

出力の見方:
    選択精度   … どちらが勝つかを当てられた割合。50%ならコイン投げと同じ
    回収率     … 実際に選んで買った場合。単独モデルを上回るかが焦点
    区間下限   … これが単独モデルを上回らないなら、差はまだ運の範囲
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
from src.models import RaceProbModel                       # noqa: E402
from src.backtest import holdout, evaluate                 # noqa: E402
from src.exotics import fit_lambdas                        # noqa: E402
from src.value import add_value_columns                    # noqa: E402
from src.recommend import recommend_for_race               # noqa: E402
from src.confidence import (                               # noqa: E402
    race_confidence, confidence_score, value_metrics, value_score,
)

KEY = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False


# ------------------------------------------------------------------ 共通
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
    return {"券種": bt, "点数": rec["点数"], "stake": 100.0 * rec["点数"],
            "payout": float(real) if hit else 0.0, "hit": int(hit),
            "p": rec["的中確率"], "odds": rec["合成オッズ"],
            "ret": rec["期待回収率"]}


def roi(stake, payout):
    s, p = np.asarray(stake, float), np.asarray(payout, float)
    return p.sum() / s.sum() * 100 if s.sum() else 0.0


def boot(stake, payout, n=3000, seed=0):
    s, p = np.asarray(stake, float), np.asarray(payout, float)
    if len(s) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), size=(n, len(s)))
    b = p[i].sum(1) / s[i].sum(1) * 100
    return tuple(np.percentile(b, [2.5, 97.5]))


def line(label, stake, payout, hit=None, extra=""):
    lo, hi = boot(stake, payout)
    h = f"的中 {np.mean(hit)*100:>5.1f}%  " if hit is not None else ""
    print(f"  {label:<28}{len(stake):>5}レース  {h}"
          f"回収 {roi(stake, payout):>6.1f}%  区間 {lo:>5.0f}〜{hi:>5.0f}%{extra}")
    return roi(stake, payout), lo


# ------------------------------------------------------------------ 高配当特化モデル
def high_odds_predictions(data, pays, test_start, min_odds):
    """馬連が min_odds 倍以上だったレースだけで学習したモデルの予測。"""
    pay_map = pays.set_index("race_id").to_dict("index")
    hot = {rid for rid, v in pay_map.items()
           if v.get("payout_umaren") and v["payout_umaren"] / 100.0 >= min_odds}
    feat = build_features(data)
    train = feat[(feat["date"] < pd.Timestamp(test_start))
                 & (feat["race_id"].isin(hot))]
    test = feat[feat["date"] >= pd.Timestamp(test_start)]
    if len(train) < 3000 or test.empty:
        return None
    print(f"  高配当特化モデルの学習: {train['race_id'].nunique():,}レース", flush=True)

    ma = RaceProbModel("A_高配当", "is_top3", use_odds=False, expected_sum=3.0)
    mb = RaceProbModel("B_高配当", "is_win", use_odds=False, expected_sum=1.0)
    ma.fit(train); mb.fit(train)

    t = test.copy()
    t["p_top3"] = ma.predict(t)
    t["p_win_pure"] = mb.predict(t)
    t = add_value_columns(t)
    t["p_win"] = t["p_blend"]
    return t


# ------------------------------------------------------------------ 選択指標
def race_features(pred_main, pred_alt, conf, vm):
    """
    レース単位の特徴量を作る。
    2つのモデルの「見立ての違い」を数値にするのが要点。
    """
    rows = []
    alt_by_race = {r: g for r, g in pred_alt.groupby("race_id", sort=False)}
    for race_id, g in pred_main.groupby("race_id", sort=False):
        h = alt_by_race.get(race_id)
        if h is None:
            continue
        g = g.sort_values("horse_no").reset_index(drop=True)
        h = h.sort_values("horse_no").reset_index(drop=True)
        if len(g) != len(h):
            continue

        pa = np.clip(g["p_top3"].to_numpy(float), 1e-9, None); pa /= pa.sum()
        pb = np.clip(h["p_top3"].to_numpy(float), 1e-9, None); pb /= pb.sum()
        q = np.clip(g["q_market"].to_numpy(float), 1e-9, None) \
            if "q_market" in g.columns else None
        if q is None:
            o = np.clip(g["odds_prev_win"].to_numpy(float), 1.01, None)
            q = (1 / o) / (1 / o).sum()
        q = q / q.sum()

        oa, ob, oq = np.argsort(pa)[::-1], np.argsort(pb)[::-1], np.argsort(q)[::-1]
        rows.append({
            "race_id": race_id,
            "n": len(g),
            # 2モデルの一致度
            "same_top1": int(oa[0] == ob[0]),
            "same_top3": len(set(oa[:3]) & set(ob[:3])),
            "kl": float(np.sum(pa * np.log(pa / pb))),
            # それぞれの自信
            "main_top1": float(pa[oa[0]]),
            "alt_top1": float(pb[ob[0]]),
            "main_margin": float(pa[oa[0]] - pa[oa[1]]) if len(pa) > 1 else 0.0,
            "alt_margin": float(pb[ob[0]] - pb[ob[1]]) if len(pb) > 1 else 0.0,
            # 市場との関係
            "main_vs_market": int(oa[0] == oq[0]),
            "alt_vs_market": int(ob[0] == oq[0]),
            "fav_p_main": float(pa[oq[0]]),
            "fav_p_alt": float(pb[oq[0]]),
            "market_top1": float(q[oq[0]]),
            "market_ent": float(-(q * np.log(q)).sum() / np.log(len(q))),
            # レースの条件
            "distance": float(g["distance"].iloc[0]),
            "class_level": float(g["class_level"].iloc[0]),
            "is_turf": int(str(g["surface"].iloc[0]) == "芝"),
        })
    d = pd.DataFrame(rows)
    d = d.merge(conf[["race_id", "conf_score", "a_entropy", "b_top3_share",
                      "model_agree"]], on="race_id", how="left")
    d = d.merge(vm[["race_id", "value_score", "max_edge", "rank_disagree",
                    "fav_overrated"]], on="race_id", how="left")
    return d


def fit_selector(tr_X, tr_y, te_X):
    cols = [c for c in tr_X.columns if tr_X[c].dtype.kind in "if"]
    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04,
                               num_leaves=24, min_child_samples=50,
                               subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(tr_X[cols].fillna(0), tr_y)
        p = m.predict_proba(te_X[cols].fillna(0))[:, 1]
        imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    else:
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05)
        m.fit(tr_X[cols].fillna(0).to_numpy(), tr_y.to_numpy())
        p = m.predict_proba(te_X[cols].fillna(0).to_numpy())[:, 1]
        imp = None
    return p, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--high-odds", type=float, default=40.0)
    ap.add_argument("--split", type=float, default=0.5,
                    help="検証期間のうち、選択指標の学習に使う割合")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return
    pay_map = pays.set_index("race_id").to_dict("index")

    pred, conf, _ = holdout(build_features(data), a.test_start)
    _s, _o, _g, _d = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)

    alt = high_odds_predictions(data, pays, a.test_start, a.high_odds)
    if alt is None:
        print("高配当特化モデルを作れませんでした。"); return

    # 両モデルの買い目と結果
    print("\n両モデルの買い目を作っています…", flush=True)
    alt_by_race = {r: g for r, g in alt.groupby("race_id", sort=False)}
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        h = alt_by_race.get(race_id)
        if h is None:
            continue
        g = g.reset_index(drop=True); h = h.reset_index(drop=True)
        ra = settle(g, recommend_for_race(g, mode="hit", lam2=lam2, lam3=lam3),
                    pay_map, race_id)
        rb = settle(h, recommend_for_race(h, mode="hit", lam2=lam2, lam3=lam3),
                    pay_map, race_id)
        if ra is None or rb is None:
            continue
        rows.append({"race_id": race_id, "date": g["date"].iloc[0],
                     **{f"main_{k}": v for k, v in ra.items()},
                     **{f"alt_{k}": v for k, v in rb.items()}})
    res = pd.DataFrame(rows)
    if res.empty:
        print("対象レースがありません。"); return

    # 正解ラベル: どちらが利益を出したか
    res["alt_better"] = ((res["alt_payout"] - res["alt_stake"]) >
                         (res["main_payout"] - res["main_stake"])).astype(int)

    vm = value_metrics(pred); vm["value_score"] = value_score(vm)
    rf = race_features(pred, alt, conf, vm)
    X = res.merge(rf, on="race_id", how="left").sort_values("date")

    cut = X["date"].quantile(a.split)
    tr = X[X["date"] < cut]
    te = X[X["date"] >= cut]
    print(f"\n選択指標の学習 {len(tr)}レース / 検証 {len(te)}レース")
    if len(tr) < 120 or len(te) < 120:
        print("データ不足です。"); return

    drop = {"race_id", "date", "alt_better"} | {
        c for c in X.columns if c.startswith(("main_", "alt_"))
        and not c.endswith(("_p", "_odds", "_ret"))}
    cols = [c for c in X.columns if c not in drop and X[c].dtype.kind in "if"]
    p, imp = fit_selector(tr[cols], tr["alt_better"], te[cols])

    print(f"\n{'='*66}")
    print(f"【後半 {len(te)}レースでの結果】")
    m_roi, _ = line("的中率モデルだけ", te["main_stake"], te["main_payout"],
                    te["main_hit"])
    a_roi, _ = line("高配当特化モデルだけ", te["alt_stake"], te["alt_payout"],
                    te["alt_hit"])

    best_pay = np.maximum(te["main_payout"], te["alt_payout"])
    best_st = np.where(te["alt_payout"] > te["main_payout"],
                       te["alt_stake"], te["main_stake"])
    line("毎回 当たった方（上限）", best_st, best_pay)

    # 予測値の分布を見せる（どのあたりにしきい値を置けるか）
    print(f"\n  高配当側を選ぶ確率の分布: "
          f"最小 {p.min():.2f} / 中央 {np.median(p):.2f} / 最大 {p.max():.2f}")
    print(f"  正解の割合（高配当側が良かったレース）: "
          f"{te['alt_better'].mean()*100:.1f}%")

    print()
    # 固定のしきい値に加えて、上位何%を選ぶかでも見る
    qs = [np.quantile(p, 1 - r) for r in (0.1, 0.2, 0.3, 0.5)]
    for th in sorted(set([0.4, 0.5, 0.6] + [round(float(q), 3) for q in qs])):
        pick_alt = p >= th
        st = np.where(pick_alt, te["alt_stake"], te["main_stake"])
        py = np.where(pick_alt, te["alt_payout"], te["main_payout"])
        hh = np.where(pick_alt, te["alt_hit"], te["main_hit"])
        acc = (pick_alt == te["alt_better"].to_numpy()).mean()
        if pick_alt.sum() == 0:
            continue
        r, lo = line(f"学習で選ぶ（しきい値{th:.2f}）", st, py, hh,
                     extra=f"  選択精度 {acc*100:.1f}%  高配当側 {pick_alt.sum()}件")
        if r > max(m_roi, a_roi) and lo > max(m_roi, a_roi):
            print(f"      → 単独を上回り、区間下限も上回っています。有望です。")
        elif r > max(m_roi, a_roi):
            print(f"      → 上回っていますが、区間下限は単独を下回ります。")

    if imp is not None:
        print("\n【選択に効いた指標 上位10】")
        for k, v in imp.head(10).items():
            print(f"  {k:<20}{v:>7.0f}")


if __name__ == "__main__":
    main()
