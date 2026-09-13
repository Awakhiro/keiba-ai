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
