"""
レース自信度(confidence)の算出。

「どの馬が来るか」より「そもそもこのレースは読めるのか」を判定する層。
以下の4つの観点を合成する。

  1. 突出度   : 本命馬の確率そのもの
  2. 差       : 1位と2位の確率差(2強混戦なら低い)
  3. 混戦度   : 正規化エントロピー(全馬が横並びなら高い = 読めない)
  4. 一致度   : オッズ非考慮モデルとオッズ考慮モデルの本命が一致するか
                (市場と独自評価が一致 = 堅い / 不一致 = 妙味はあるが不安定)

生スコアは意味を持たないので、バックテストで「スコア帯ごとの実測的中率」を
学習してキャリブレートし、S/A/B/C のグレードに落とす。
"""

import numpy as np
import pandas as pd


def _entropy_norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    p = p / p.sum()
    n = len(p)
    if n <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(n))


def race_confidence(df: pd.DataFrame, pa_col="p_top3", pb_col="p_win") -> pd.DataFrame:
    """レース単位の自信度指標を計算する。"""
    rows = []
    for race_id, g in df.groupby("race_id", sort=False):
        pa = g[pa_col].to_numpy(dtype=float)
        pb = g[pb_col].to_numpy(dtype=float)
        n = len(g)

        pa_s = np.sort(pa)[::-1]
        pb_s = np.sort(pb)[::-1]

        top_a = g.iloc[int(np.argmax(pa))]
        top_b = g.iloc[int(np.argmax(pb))]
        agree = int(top_a["horse_id"] == top_b["horse_id"])

        ent_a = _entropy_norm(pa)
        ent_b = _entropy_norm(pb)

        rows.append({
            "race_id": race_id,
            "field_size": n,
            "a_top1_p": pa_s[0],
            "a_margin": pa_s[0] - (pa_s[1] if n > 1 else 0.0),
            "a_entropy": ent_a,
            "b_top1_p": pb_s[0],
            "b_margin": pb_s[0] - (pb_s[1] if n > 1 else 0.0),
            "b_top3_share": pb_s[:3].sum(),
            "b_entropy": ent_b,
            "model_agree": agree,
            "a_pick_horse": top_a["horse_id"],
            "b_pick_horse": top_b["horse_id"],
        })
    return pd.DataFrame(rows)


def confidence_score(conf: pd.DataFrame) -> pd.Series:
    """0〜100 の生スコア。重みは経験則ベース(バックテストで調整可)。"""
    def z(s):
        s = s.astype(float)
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    raw = (
        1.1 * z(conf["a_top1_p"])
        + 0.9 * z(conf["a_margin"])
        + 0.7 * z(conf["b_top1_p"])
        + 0.6 * z(conf["b_margin"])
        + 0.5 * z(conf["b_top3_share"])
        - 0.8 * z(conf["a_entropy"])
        - 0.4 * z(conf["b_entropy"])
        + 0.45 * conf["model_agree"]
    )
    # 0-100 に写像
    r = raw.rank(pct=True) * 100
    return r


class ConfidenceGrader:
    """バックテスト結果からスコア帯ごとの実測的中率を学習し、グレードを付ける。"""

    GRADES = ["C", "B", "A", "S"]

    def __init__(self, quantiles=(0.50, 0.75, 0.90)):
        self.quantiles = quantiles
        self.thresholds = None
        self.stats = None

    def fit(self, scores: pd.Series, hit_flags: pd.Series):
        self.thresholds = [float(scores.quantile(q)) for q in self.quantiles]
        g = self.transform(scores)
        self.stats = (
            pd.DataFrame({"grade": g, "hit": hit_flags.to_numpy(dtype=float)})
            .groupby("grade")["hit"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "実測的中率", "count": "レース数"})
        )
        return self

    def transform(self, scores: pd.Series) -> pd.Series:
        if self.thresholds is None:
            # 未学習時は固定閾値
            self.thresholds = [50.0, 75.0, 90.0]
        t = self.thresholds
        cond = [scores >= t[2], scores >= t[1], scores >= t[0]]
        out = np.select(cond, ["S", "A", "B"], default="C")
        return pd.Series(out, index=scores.index)


def pick_confident_races(conf: pd.DataFrame, grader: ConfidenceGrader,
                         min_grade="A", top_n=None) -> pd.DataFrame:
    """勝負レースを抽出する。"""
    conf = conf.copy()
    conf["conf_score"] = confidence_score(conf)
    conf["grade"] = grader.transform(conf["conf_score"])
    order = {"S": 3, "A": 2, "B": 1, "C": 0}
    conf["_o"] = conf["grade"].map(order)
    sel = conf[conf["_o"] >= order[min_grade]].sort_values("conf_score", ascending=False)
    if top_n:
        sel = sel.head(top_n)
    return sel.drop(columns=["_o"])


# ------------------------------------------------------------------ 妙味の格付け
def value_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """
    レースごとに「妙味がありそうか」を測る指標を出す。

    的中率の格付けが「読みやすいか」を測るのに対し、こちらは
    「市場の評価とモデルの評価がどれだけ食い違っているか」を測る。

    実力が拮抗していて人気馬が飛びそうなレースは、的中率では低評価になるが
    市場との乖離は大きくなりやすい。その取りこぼしを拾うための指標。
    """
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        p = np.clip(g["p_win_pure"].to_numpy(dtype=float)
                    if "p_win_pure" in g.columns
                    else g["p_win"].to_numpy(dtype=float), 1e-9, None)
        p = p / p.sum()
        q = np.clip(g["q_market"].to_numpy(dtype=float), 1e-9, None) \
            if "q_market" in g.columns else None
        if q is None:
            o = np.clip(g["odds_prev_win"].to_numpy(dtype=float), 1.01, None)
            q = (1.0 / o) / (1.0 / o).sum()
        q = q / q.sum()

        edge = p / q
        order_p = np.argsort(p)[::-1]
        order_q = np.argsort(q)[::-1]

        # 能力上位3頭のうち、市場評価が低い馬の乖離
        top_edges = sorted(edge[order_p[:3]], reverse=True)
        # 市場の1番人気をモデルがどう見ているか（1未満なら過大評価とみなす）
        fav_edge = float(edge[order_q[0]])
        # モデルと市場の順位の食い違い
        rank_p = np.empty(len(p), int); rank_p[order_p] = np.arange(len(p))
        rank_q = np.empty(len(q), int); rank_q[order_q] = np.arange(len(q))
        disagree = float(np.abs(rank_p - rank_q)[order_q[:5]].mean())

        rows.append({
            "race_id": race_id,
            "max_edge": float(max(edge)),
            "top3_edge": float(np.mean(top_edges)) if top_edges else 1.0,
            "fav_edge": fav_edge,
            "fav_overrated": float(max(0.0, 1.0 - fav_edge)),
            "rank_disagree": disagree,
            "market_entropy": _entropy_norm(q),
        })
    return pd.DataFrame(rows)


def value_score(vm: pd.DataFrame) -> pd.Series:
    """0〜100 の妙味スコア。大きいほど市場との食い違いが大きい。"""
    def z(s):
        s = s.astype(float)
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    raw = (
        1.0 * z(vm["top3_edge"])          # 能力上位が市場に評価されていない
        + 0.8 * z(vm["fav_overrated"])    # 1番人気が過大評価されている
        + 0.6 * z(vm["rank_disagree"])    # 上位人気の順序がモデルと食い違う
        + 0.4 * z(vm["max_edge"])         # 突出した乖離がある
        + 0.3 * z(vm["market_entropy"])   # 市場も割れている＝配当が付く
    )
    return raw.rank(pct=True) * 100


def choose_mode(conf_score, val_score, margin=12.0):
    """
    そのレースを「当てにいく」か「妙味を狙う」かを決める。

    どちらのスコアが高いかで選ぶが、差が小さいときは当てにいく方を既定にする。
    妙味狙いは当たらない期間が長くなるため、迷ったら堅い方を選ぶという考え方。
    """
    c = np.asarray(conf_score, dtype=float)
    v = np.asarray(val_score, dtype=float)
    return np.where(v - c >= margin, "妙味", "的中率")


# ------------------------------------------------------------------ モデル別の格付け
# ------------------------------------------------------------------ モデル別の自信度
# 穴寄りのモデルでは「自信がある」の意味が本命と同じとは限らない。
# そこで測り方を複数用意し、検証で一番効くものを選ぶ（scripts/eval_grades.py）。
# どれもレース単位の絶対的な値で、日によって尺度が変わらないものに限っている
# （当日の予想でも同じしきい値で格付けできるように）。
GRADE_METHODS = {
    "top1":     "本命馬の確率",
    "margin":   "1位と2位の確率の差",
    "conc":     "上位3頭への確率の集中",
    "calm":     "混戦でないこと（1 − 正規化エントロピー）",
    "hitp":     "選んだ買い目の的中確率",
    "ev":       "選んだ買い目の期待回収率",
    "disagree": "本命モデルとの食い違い（全馬の順位の相関）",
    "edge":     "推している上位3頭の市場との乖離",
}


def model_confidence(g, prob_col, rec=None, base_col="p_top3_base"):
    """
    1レース分について、各測り方の値を返す。
    g: 出走馬の表, prob_col: そのモデルの複勝圏確率の列, rec: そのモデルの推奨
    """
    p = np.clip(g[prob_col].to_numpy(dtype=float), 1e-9, None)
    p = p / p.sum()
    order = np.argsort(p)[::-1]
    n = len(p)
    out = {
        "top1": float(p[order[0]]),
        "margin": float(p[order[0]] - p[order[1]]) if n > 1 else 0.0,
        "conc": float(p[order[:3]].sum()),
        "calm": float(1.0 - _entropy_norm(p)),
    }
    if rec is not None and not rec.get("見送り", rec.get("skip", False)):
        out["hitp"] = float(rec.get("的中確率", rec.get("p", np.nan)))
        out["ev"] = float(rec.get("期待回収率", rec.get("ret", np.nan)))
    else:
        out["hitp"] = np.nan
        out["ev"] = np.nan

    if base_col in g.columns and base_col != prob_col and n >= 3:
        # 全馬の順位がどれだけ食い違うか（1 − 順位相関）。0 なら同じ見立て、
        # 大きいほど本命モデルと違う馬を推している＝独自の情報を持っている。
        # 上位3頭の入れ替わり数だと 0〜3 の4段階にしかならず格付けが粗くなるので、
        # 連続的な値にしている。
        b = g[base_col].to_numpy(dtype=float)
        ra = pd.Series(p).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        c = np.corrcoef(ra, rb)[0, 1] if np.std(ra) > 0 and np.std(rb) > 0 else 1.0
        out["disagree"] = float(1.0 - c)
    else:
        out["disagree"] = np.nan

    if "odds_prev_win" in g.columns:
        o = np.clip(pd.to_numeric(g["odds_prev_win"], errors="coerce")
                    .fillna(n).to_numpy(dtype=float), 1.01, None)
        q = (1 / o) / (1 / o).sum()
        out["edge"] = float(np.mean(p[order[:3]] / q[order[:3]]))
    else:
        out["edge"] = np.nan
    return out


def grade_from_score(score, thresholds):
    """しきい値 [B以上, A以上, S以上] で格付けする。"""
    if score is None or not np.isfinite(score) or not thresholds:
        return None
    b, a, s = thresholds
    return "S" if score >= s else "A" if score >= a else "B" if score >= b else "C"


def load_grade_config(path=None):
    """モデル別の格付け設定（検証で選んだ測り方としきい値）を読む。無ければ空。"""
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "grade_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
