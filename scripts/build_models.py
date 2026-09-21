"""
検証に使うモデルを一度に全部学習して保存する。

    python scripts/build_models.py --test-start 2026-01-01

条件を変えて何度も検証したいとき、そのたびに学習し直すのは無駄。
ここで一度だけ学習し、予測結果まで含めて保存しておけば、
あとは条件を変えて読み込むだけで済む。

作るモデル:
    base    … 全レースで学習（本命）
    pop4/5/6/8  … 2着以内に N番人気以下が来たレースで学習（中穴の候補）
    odds20/40   … 馬連が N倍以上だったレースで学習（穴の候補）

保存先 data/eval/ :
    predictions.parquet  各モデルの確率を横に並べたもの
    meta.json            学習の条件と、データの指紋
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, save_table, exists          # noqa: E402
from src.netkeiba import to_model_schema                      # noqa: E402
from src.features import build_features                       # noqa: E402
from src.models import RaceProbModel, make_models             # noqa: E402
from src.backtest import evaluate, walk_forward               # noqa: E402
from src.exotics import fit_lambdas                           # noqa: E402
from src.value import add_value_columns                       # noqa: E402
from src.modelstore import data_fingerprint                   # noqa: E402
from src.predict import KeibaPredictor                        # noqa: E402

OUT_DIR = "data/eval"


def upset_races(feat, min_pop, max_place=2):
    """max_place 着以内に min_pop 番人気以下が来たレース。"""
    out = set()
    for race_id, g in feat.groupby("race_id", sort=False):
        pop = None
        if "popularity_final" in g.columns:
            v = g["popularity_final"]
            if v.notna().all() and v.nunique() > 1:
                pop = v.to_numpy()
        if pop is None:
            col = ("odds_win_final" if "odds_win_final" in g.columns
                   else "odds_prev_win")
            if col not in g.columns:
                continue
            pop = g[col].rank(method="min").to_numpy()
        fin = g["finish_pos"].to_numpy(dtype=float)
        m = (fin <= max_place) & ~np.isnan(fin)
        if m.any() and np.nanmax(pop[m]) >= min_pop:
            out.add(str(race_id))
    return out


def high_odds_races(pays, min_odds):
    return {str(r) for r, v in pays.set_index("race_id").to_dict("index").items()
            if v.get("payout_umaren") and v["payout_umaren"] / 100.0 >= min_odds}


def fit_subset(train, race_ids, label, min_races=400):
    if race_ids is not None:
        sub = train[train["race_id"].astype(str).isin(race_ids)]
    else:
        sub = train
    n = sub["race_id"].nunique()
    if n < min_races:
        print(f"  {label:<10} 対象 {n}レースで不足", flush=True)
        return None, n
    ma = RaceProbModel(f"A_{label}", "is_top3", use_odds=False, expected_sum=3.0)
    mb = RaceProbModel(f"B_{label}", "is_win", use_odds=False, expected_sum=1.0)
    ma.fit(sub)
    mb.fit(sub)
    print(f"  {label:<10} {n:>6,}レースで学習", flush=True)
    return (ma, mb), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="保存済みがあっても学習し直す")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    fp = data_fingerprint(data) + f"_{a.test_start}"

    meta_p = os.path.join(a.out, "meta.json")
    if not a.force and os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") == fp:
            print("保存済みの予測があり、データも変わっていません。")
            print(f"  {old['races']:,}レース / 作成 {old['created']}")
            print("学習し直すには --force を付けてください。")
            return

    print(f"データ {len(data):,}行 / {data['race_id'].nunique():,}レース")
    feat = build_features(data)
    cut = pd.Timestamp(a.test_start)
    train = feat[feat["date"] < cut]
    test = feat[feat["date"] >= cut].copy()
    print(f"学習 {train['race_id'].nunique():,}レース / "
          f"検証 {test['race_id'].nunique():,}レース\n")

    # ---- 学習対象の一覧 ----
    targets = {"base": None}
    for pop in (4, 5, 6, 8):
        targets[f"pop{pop}"] = upset_races(train, pop, max_place=2)
    if len(pays):
        for th in (20, 40):
            targets[f"odds{th}"] = high_odds_races(pays, th)

    print("モデルを学習します")
    counts = {}
    for label, races in targets.items():
        pair, n = fit_subset(train, races, label)
        counts[label] = n
        if pair is None:
            continue
        ma, mb = pair
        test[f"p_top3_{label}"] = ma.predict(test)
        test[f"p_win_{label}"] = mb.predict(test)

    # 市場情報（妙味の計算に使う列）を一度だけ作る
    test["p_top3"] = test["p_top3_base"]
    test["p_win_pure"] = test["p_win_base"]
    test = add_value_columns(test)
    test["p_win"] = test["p_blend"]

    # 自信度の基準と順位割引
    print("\n自信度の基準を作っています…", flush=True)
    pred_wf, conf_wf = walk_forward(feat[feat["date"] < cut], n_folds=3,
                                    verbose=False)
    _s, _o, grader, _d = evaluate(pred_wf, conf_wf)
    lam2, lam3 = fit_lambdas(pred_wf)

    keep = [c for c in test.columns
            if c in ("race_id", "date", "venue", "race_no", "surface", "distance",
                     "class_level", "field_size", "horse_id", "horse_no",
                     "frame_no", "horse_name", "odds_prev_win", "odds_win_final",
                     "popularity_final", "finish_pos", "post_time",
                     "q_market", "p_blend", "p_win", "p_top3", "p_win_pure",
                     "edge_blend", "pop_gap")
            or c.startswith(("p_top3_", "p_win_"))]
    os.makedirs(a.out, exist_ok=True)
    save_table(test[keep], os.path.join(a.out, "predictions"))

    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump({
            "fingerprint": fp,
            "test_start": a.test_start,
            "races": int(test["race_id"].nunique()),
            "models": {k: int(v) for k, v in counts.items()},
            "lam2": float(lam2), "lam3": float(lam3),
            "thresholds": grader.thresholds,
            "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }, f, ensure_ascii=False, indent=1)

    made = [k for k in targets if f"p_top3_{k}" in test.columns]
    print(f"\n保存しました: {a.out}")
    print(f"  検証 {test['race_id'].nunique():,}レース / モデル {len(made)}種類")
    print(f"  {', '.join(made)}")
    print("\n以降は scripts/eval_filters.py で、学習せずに条件を変えて検証できます。")


if __name__ == "__main__":
    main()
