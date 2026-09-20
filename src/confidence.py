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
def model_confidence(pred: pd.DataFrame, prob_col="p_top3") -> pd.DataFrame:
    """
    任意のモデルの確率分布から、そのモデル自身の自信度を出す。

    的中率モデル専用だった race_confidence を、どのモデルにも使えるようにしたもの。
    「そのモデルから見て、このレースは読みやすいか」を測る。
    """
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        p = np.clip(g[prob_col].to_numpy(dtype=float), 1e-9, None)
        p = p / p.sum()
        n = len(p)
        srt = np.sort(p)[::-1]
        rows.append({
            "race_id": race_id,
            "field_size": n,
            "top1_p": float(srt[0]),
            "margin": float(srt[0] - srt[1]) if n > 1 else 0.0,
            "top3_share": float(srt[:3].sum()),
            "entropy": _entropy_norm(p),
        })
    return pd.DataFrame(rows)


def model_score(cm: pd.DataFrame) -> pd.Series:
    """モデル別自信度を 0〜100 にする。"""
    def z(s):
        s = s.astype(float)
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    raw = (1.2 * z(cm["top1_p"]) + 1.0 * z(cm["margin"])
           + 0.6 * z(cm["top3_share"]) - 0.9 * z(cm["entropy"]))
    return raw.rank(pct=True) * 100


def grade_from_score(score: pd.Series, thresholds=(50.0, 75.0, 90.0)) -> pd.Series:
    """スコアを S/A/B/C に変換する。"""
    t = thresholds
    cond = [score >= t[2], score >= t[1], score >= t[0]]
    return pd.Series(np.select(cond, ["S", "A", "B"], default="C"),
                     index=score.index)
