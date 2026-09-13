"""
ウォークフォワード・バックテスト。

競馬データは時系列なので、ランダムな交差検証は必ず過大評価になる。
「過去N期間で学習 → 次の1期間で検証 → 窓をずらす」を繰り返す。

評価指標:
  - 単勝的中率 / 複勝的中率
  - 単勝回収率 / 複勝回収率(100%を超えれば理論上プラス。控除率20%の壁がある)
  - 自信度グレード別の成績  ← ここが本命。全レース買うのではなく選別する前提。
"""

import numpy as np
import pandas as pd

from .models import make_models
from .confidence import race_confidence, confidence_score, ConfidenceGrader


def walk_forward(feat: pd.DataFrame, n_folds=5, min_train_ratio=0.4, verbose=True):
    """
    feat: build_features() の出力(is_win/is_top3 と payout 列を含む)
    戻り値: (予測が付与された検証データ, レース単位の自信度テーブル)
    """
    dates = np.sort(feat["date"].unique())
    start = int(len(dates) * min_train_ratio)
    bounds = np.linspace(start, len(dates), n_folds + 1).astype(int)

    preds = []
    for i in range(n_folds):
        tr_end, te_end = dates[bounds[i] - 1], dates[bounds[i + 1] - 1]
        train = feat[feat["date"] <= tr_end]
        test = feat[(feat["date"] > tr_end) & (feat["date"] <= te_end)]
        if len(test) == 0 or len(train) < 1000:
            continue

        model_a, model_b = make_models()
        model_a.fit(train)
        model_b.fit(train)

        t = test.copy()
        t["p_top3"] = model_a.predict(t)
        t["p_win"] = model_b.predict(t)
        t["fold"] = i
        preds.append(t)

        if verbose:
            print(f"  fold{i}: train={len(train):6d}件 / test={len(test):6d}件 "
                  f"({pd.Timestamp(tr_end).date()} → {pd.Timestamp(te_end).date()})")

    if not preds:
        raise ValueError("検証データが作れませんでした。データ量を増やしてください。")

    out = pd.concat(preds).sort_values(["date", "race_id", "horse_no"])
    conf = race_confidence(out)
    conf["conf_score"] = confidence_score(conf)
    return out, conf


def evaluate(pred: pd.DataFrame, conf: pd.DataFrame):
    """モデル別・グレード別の成績を集計する。"""
    # 各レースの本命馬(モデルA=複勝軸 / モデルB=単勝軸)
    idx_a = pred.groupby("race_id")["p_top3"].idxmax()
    idx_b = pred.groupby("race_id")["p_win"].idxmax()
    pick_a = pred.loc[idx_a].set_index("race_id")
    pick_b = pred.loc[idx_b].set_index("race_id")

    c = conf.set_index("race_id").copy()
    c["A_複勝的中"] = pick_a["is_top3"]
    c["A_単勝的中"] = pick_a["is_win"]
    c["B_単勝的中"] = pick_b["is_win"]
    c["B_複勝的中"] = pick_b["is_top3"]

    if "payout_win" in pred.columns:
        c["A_複勝払戻"] = pick_a["is_top3"] * pick_a.get("payout_place", 0)
        c["B_単勝払戻"] = pick_b["is_win"] * pick_b.get("payout_win", 0)
        c["B_期待値"] = pick_b["p_win"] * pick_b["odds_prev_win"]

    grader = ConfidenceGrader().fit(c["conf_score"], c["A_複勝的中"])
    c["grade"] = grader.transform(c["conf_score"])

    agg = {
        "レース数": ("A_複勝的中", "size"),
        "A複勝的中率": ("A_複勝的中", "mean"),
        "A単勝的中率": ("A_単勝的中", "mean"),
        "B単勝的中率": ("B_単勝的中", "mean"),
        "B複勝的中率": ("B_複勝的中", "mean"),
    }
    if "B_単勝払戻" in c.columns:
        agg["A複勝回収率"] = ("A_複勝払戻", "mean")
        agg["B単勝回収率"] = ("B_単勝払戻", "mean")

    summary = c.groupby("grade").agg(**agg)
    summary = summary.reindex(["S", "A", "B", "C"]).dropna(how="all")
    for col in summary.columns:
        if col.endswith("的中率"):
            summary[col] = (summary[col] * 100).round(1)   # 割合 → %
        elif col.endswith("回収率"):
            # 払戻は100円賭けたときの円。100で割ると回収率(%)。
            summary[col] = (summary[col] / 100 * 100).round(1)
    overall = pd.Series({
        "レース数": len(c),
        "A複勝的中率": round(c["A_複勝的中"].mean() * 100, 1),
        "A単勝的中率": round(c["A_単勝的中"].mean() * 100, 1),
        "B単勝的中率": round(c["B_単勝的中"].mean() * 100, 1),
        "B複勝的中率": round(c["B_複勝的中"].mean() * 100, 1),
    })
    return summary, overall, grader, c


def ev_filter_report(pred: pd.DataFrame, thresholds=(1.0, 1.1, 1.2, 1.3)):
    """期待値フィルタごとの単勝成績(モデルB)。"""
    if "payout_win" not in pred.columns:
        return None
    d = pred.copy()
    d["ev"] = d["p_win"] * d["odds_prev_win"]
    rows = []
    for th in thresholds:
        s = d[d["ev"] >= th]
        if len(s) == 0:
            continue
        rows.append({
            "期待値しきい値": th,
            "購入点数": len(s),
            "的中率%": round(s["is_win"].mean() * 100, 1),
            "回収率%": round((s["is_win"] * s["payout_win"]).mean(), 1),
            "平均オッズ": round(s["odds_prev_win"].mean(), 1),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 連勝式評価
def _win_probs_from_model(g, model):
    """モデルAは複勝圏確率なので、勝率相当の強さに変換して使う。"""
    if model == "A":
        s = g["p_top3"].to_numpy(dtype=float)
    else:
        s = g["p_win"].to_numpy(dtype=float)
    s = np.clip(s, 1e-9, None)
    return s / s.sum()


def evaluate_exotics(pred, graded, bet_types=("馬連", "馬単", "3連複", "3連単"),
                     points=(6, 8, 10, 16), lam2=None, lam3=None):
    """
    券種別・自信度グレード別に的中率と推定回収率を出す。
    graded: evaluate() が返す detail（grade 列つき、race_id インデックス）
    """
    from .exotics import build_bets, market_win_probs, LAMBDA2_DEFAULT, LAMBDA3_DEFAULT

    lam2 = lam2 or LAMBDA2_DEFAULT
    lam3 = lam3 or LAMBDA3_DEFAULT
    grade_map = graded["grade"].to_dict()
    pts = dict(zip(bet_types, points))

    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        fin = g["finish_pos"].to_numpy()
        try:
            t1, t2, t3 = (int(np.where(fin == k)[0][0]) for k in (1, 2, 3))
        except IndexError:
            continue
        market = market_win_probs(g["odds_prev_win"].to_numpy())
        grade = grade_map.get(race_id, "C")

        truth = {
            "馬連": tuple(sorted((t1, t2))),
            "馬単": (t1, t2),
            "3連複": tuple(sorted((t1, t2, t3))),
            "3連単": (t1, t2, t3),
        }

        for model in ("A", "B"):
            s = _win_probs_from_model(g, model)
            strategy = "hit" if model == "A" else "ev"
            for bt in bet_types:
                bets = build_bets(s, market, bt, max_points=pts[bt],
                                  strategy=strategy, lam2=lam2, lam3=lam3)
                if bets.empty:
                    continue
                hit_row = bets[bets["combo"] == truth[bt]]
                hit = len(hit_row) > 0
                ret = float(hit_row["odds"].iloc[0]) if hit else 0.0
                rows.append({
                    "grade": grade, "model": model, "券種": bt,
                    "点数": len(bets), "的中": int(hit),
                    "回収": ret * 100 / len(bets),   # 1点100円の総賭け金あたり
                })

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    out = (
        d.groupby(["券種", "model", "grade"])
        .agg(レース数=("的中", "size"), 的中率=("的中", "mean"),
             推定回収率=("回収", "mean"), 平均点数=("点数", "mean"))
        .reset_index()
    )
    out["的中率"] = (out["的中率"] * 100).round(1)
    out["推定回収率"] = out["推定回収率"].round(1)
    out["平均点数"] = out["平均点数"].round(1)
    order = {"S": 0, "A": 1, "B": 2, "C": 3}
    out["_o"] = out["grade"].map(order)
    return out.sort_values(["券種", "model", "_o"]).drop(columns="_o").reset_index(drop=True)


def evaluate_exotics_real(pred, graded, pays, bet_types=("馬連", "馬単", "3連複", "3連単"),
                          points=(6, 8, 10, 16), lam2=None, lam3=None):
    """
    実際の払戻データを使った連勝式の検証。推定配当ではなく実配当で回収率を出す。
    pays: netkeiba から取得した払戻DataFrame（race_id, payout_umaren, ... を含む）
    """
    from .exotics import build_bets, market_win_probs, LAMBDA2_DEFAULT, LAMBDA3_DEFAULT

    lam2 = lam2 or LAMBDA2_DEFAULT
    lam3 = lam3 or LAMBDA3_DEFAULT
    grade_map = graded["grade"].to_dict()
    pts = dict(zip(bet_types, points))
    key = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}
    pay_map = pays.set_index("race_id").to_dict("index") if len(pays) else {}

    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        fin = g["finish_pos"].to_numpy()
        try:
            t1, t2, t3 = (int(np.where(fin == k)[0][0]) for k in (1, 2, 3))
        except IndexError:
            continue
        market = market_win_probs(g["odds_prev_win"].to_numpy())
        grade = grade_map.get(race_id, "C")
        pr = pay_map.get(race_id, {})
        truth = {"馬連": tuple(sorted((t1, t2))), "馬単": (t1, t2),
                 "3連複": tuple(sorted((t1, t2, t3))), "3連単": (t1, t2, t3)}

        for model in ("A", "B"):
            s = _win_probs_from_model(g, model)
            strategy = "hit" if model == "A" else "ev"
            for bt in bet_types:
                real = pr.get(f"payout_{key[bt]}")
                if real is None or not np.isfinite(real):
                    continue
                bets = build_bets(s, market, bt, max_points=pts[bt],
                                  strategy=strategy, lam2=lam2, lam3=lam3)
                if bets.empty:
                    continue
                hit = truth[bt] in set(bets["combo"])
                n = len(bets)
                rows.append({"grade": grade, "model": model, "券種": bt,
                             "点数": n, "的中": int(hit),
                             "回収": (real / (100.0 * n) * 100.0) if hit else 0.0})

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    out = (d.groupby(["券種", "model", "grade"])
             .agg(レース数=("的中", "size"), 的中率=("的中", "mean"),
                  回収率=("回収", "mean"), 点数=("点数", "mean")).reset_index())
    out["的中率"] = (out["的中率"] * 100).round(1)
    out["回収率"] = out["回収率"].round(1)
    order = {"S": 0, "A": 1, "B": 2, "C": 3}
    out["_o"] = out["grade"].map(order)
    return out.sort_values(["券種", "model", "_o"]).drop(columns="_o").reset_index(drop=True)


def holdout(feat: pd.DataFrame, test_start, test_end=None, verbose=True):
    """
    期間を固定して検証する（例: 学習=2026年7月まで / 検証=8〜9月）。
    walk_forward と同じ形式で (予測つき検証データ, 自信度テーブル) を返す。
    """
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end) if test_end else feat["date"].max()
    train = feat[feat["date"] < test_start]
    test = feat[(feat["date"] >= test_start) & (feat["date"] <= test_end)]
    if len(train) < 2000 or len(test) == 0:
        raise ValueError(f"データ不足: 学習{len(train)}行 / 検証{len(test)}行")

    model_a, model_b = make_models()
    model_a.fit(train)
    model_b.fit(train)

    t = test.copy()
    t["p_top3"] = model_a.predict(t)
    t["p_win"] = model_b.predict(t)
    if verbose:
        print(f"学習 {train['date'].min().date()}〜{train['date'].max().date()} "
              f"{train['race_id'].nunique()}レース / "
              f"検証 {test['date'].min().date()}〜{test['date'].max().date()} "
              f"{test['race_id'].nunique()}レース")

    conf = race_confidence(t)
    conf["conf_score"] = confidence_score(conf)
    return t, conf, (model_a, model_b)
