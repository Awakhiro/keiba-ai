"""
妙味（市場の過小評価）を見つける。

考え方:
    1. オッズを一切見ずに「着に絡む力」を推定する        → p（能力）
    2. オッズから市場が考えている確率を復元する          → q（市場）
    3. p が q を上回っている馬が、人気が想定より低い馬    → 乖離 = p / q

旧モデルBの失敗:
    オッズを特徴量として勾配ブースティングに入れていたため、
    推定勝率がオッズをなぞってしまい「勝率 × オッズ」がどの馬もほぼ横並びになった。
    差が出るのはモデル誤差の大きい人気薄だけなので、期待値順に並べると
    実力ではなくノイズの大きい馬から選ばれてしまっていた。

今回の作り:
    能力の推定にオッズを使わない。オッズは「比較対象」としてだけ使う。
    ただし能力推定は市場より精度が劣る場面も多いので、そのまま信じ切らず、
    対数プーリングで市場側に縮める（blend）。w が小さいほど市場寄り。

        log p_final = w * log p + (1-w) * log q    （レース内で再正規化）

    人気薄ほど推定が甘くなるため、次の3つの歯止めを必ず併用する:
        ・能力の下限（そもそも着に絡む力があること）
        ・乖離の上限（極端な乖離はモデルの誤差とみなす）
        ・オッズの上限（大穴は推定が当てにならない）
"""

import numpy as np
import pandas as pd

# JRA の控除率。市場確率を復元するときに剥がす。
TAKEOUT_WIN = 0.20
TAKEOUT_PLACE = 0.20


def market_probs(odds, takeout=TAKEOUT_WIN):
    """
    単勝オッズから市場の勝率を復元する。
    1/オッズ の合計は (1-控除率) に近くなるので、合計1に正規化する。
    """
    o = np.clip(np.asarray(odds, dtype=float), 1.01, None)
    inv = 1.0 / o
    s = inv.sum()
    return inv / s if s > 0 else np.full(len(o), 1.0 / max(len(o), 1))


def blend_probs(p_model, q_market, w=0.45):
    """
    対数プーリングでモデルと市場を混ぜる。
    w=1 なら完全にモデル、w=0 なら完全に市場。
    """
    p = np.clip(np.asarray(p_model, dtype=float), 1e-6, 1.0)
    q = np.clip(np.asarray(q_market, dtype=float), 1e-6, 1.0)
    log_mix = w * np.log(p) + (1.0 - w) * np.log(q)
    mix = np.exp(log_mix - log_mix.max())
    return mix / mix.sum()


def add_value_columns(df: pd.DataFrame, w=0.45,
                      p_col="p_win_pure", top3_col="p_top3",
                      odds_col="odds_prev_win") -> pd.DataFrame:
    """
    レースごとに市場確率・混合確率・乖離を計算して列を足す。

    追加される列:
        q_market     市場の推定勝率
        p_blend      能力と市場を混ぜた勝率（買い目の確率計算に使う）
        edge         能力 / 市場。1より大きいほど人気が想定より低い
        edge_blend   混合後 / 市場。実際に使う控えめな乖離
        pop_model    能力による人気順（1が本命）
        pop_market   オッズによる人気順
        pop_gap      市場の人気順 - 能力の人気順。プラスなら過小評価
        ev_win       混合確率 × 単勝オッズ
    """
    d = df.copy()
    out = []
    for _, g in d.groupby("race_id", sort=False):
        g = g.copy()
        q = market_probs(g[odds_col].to_numpy())
        p = np.clip(g[p_col].to_numpy(dtype=float), 1e-6, None)
        p = p / p.sum()
        b = blend_probs(p, q, w=w)

        g["q_market"] = q
        g["p_blend"] = b
        g["edge"] = p / q
        g["edge_blend"] = b / q
        g["pop_model"] = (-p).argsort().argsort() + 1
        g["pop_market"] = (-q).argsort().argsort() + 1
        g["pop_gap"] = g["pop_market"] - g["pop_model"]
        g["ev_win"] = b * g[odds_col].to_numpy(dtype=float)
        out.append(g)
    return pd.concat(out).loc[d.index]


def pick_value_horses(df: pd.DataFrame, min_top3=0.28, min_edge=1.25,
                      max_odds=40.0, min_pop_gap=1, max_per_race=2) -> pd.DataFrame:
    """
    妙味馬を選ぶ。条件は「着に絡む力があること」が先、「割安であること」が後。

    min_top3     複勝圏に来る確率の下限。実力の足切り
    min_edge     能力 / 市場 の下限。これを超えたら過小評価とみなす
    max_odds     これより人気薄は推定が当てにならないので除外
    min_pop_gap  市場の人気順が能力の人気順より何枚下か
    max_per_race 1レースから拾う上限
    """
    d = df[
        (df["p_top3"] >= min_top3)
        & (df["edge_blend"] >= min_edge)
        & (df["odds_prev_win"] <= max_odds)
        & (df["pop_gap"] >= min_pop_gap)
    ].copy()
    if d.empty:
        return d
    d = d.sort_values(["race_id", "edge_blend"], ascending=[True, False])
    return d.groupby("race_id", sort=False).head(max_per_race).reset_index(drop=True)


def evaluate_value_picks(picks: pd.DataFrame) -> dict:
    """妙味馬の単勝・複勝成績。着順と払戻がある検証データ用。"""
    if picks.empty:
        return {"点数": 0}
    n = len(picks)
    res = {
        "点数": n,
        "対象レース": int(picks["race_id"].nunique()),
        "平均オッズ": round(float(picks["odds_prev_win"].mean()), 1),
        "平均乖離": round(float(picks["edge_blend"].mean()), 2),
        "単勝的中率": round(float(picks["is_win"].mean()) * 100, 1),
        "複勝的中率": round(float(picks["is_top3"].mean()) * 100, 1),
    }
    if "payout_win" in picks.columns:
        res["単勝回収率"] = round(float((picks["is_win"] * picks["payout_win"]).sum())
                              / (100 * n) * 100, 1)
    if "payout_place" in picks.columns:
        res["複勝回収率"] = round(float((picks["is_top3"] * picks["payout_place"]).sum())
                              / (100 * n) * 100, 1)
    return res


def bootstrap_roi(picks: pd.DataFrame, payout_col="payout_place",
                  hit_col="is_top3", n_boot=4000, seed=0):
    """回収率の95%信頼区間。下限が100%を割るならまだ運の範囲。"""
    if picks.empty or payout_col not in picks.columns:
        return None
    pay = (picks[hit_col] * picks[payout_col]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pay), size=(n_boot, len(pay)))
    boots = pay[idx].mean(1) / 100 * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"回収率": round(float(pay.mean()), 1),
            "下限": round(float(lo), 1), "上限": round(float(hi), 1)}
