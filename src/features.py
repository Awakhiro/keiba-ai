"""
特徴量生成。

原則:
  すべての集計特徴量は「そのレースより前のデータのみ」から作る。
  実装は cumsum - 自分自身 / cumcount によるベクトル化(全データを date 昇順に整列済み前提)。
"""

import numpy as np
import pandas as pd

from .timeform import add_time_features

DIST_BUCKET_EDGES = [0, 1400, 1800, 2200, 9999]
DIST_BUCKET_LABELS = [0, 1, 2, 3]


# ---------------------------------------------------------------- 内部ヘルパ
def _past_mean(df, keys, col):
    """keys でグループ化した、自分より前の行だけの平均と件数を返す。"""
    v = df[col].fillna(0.0).to_numpy(dtype=float)
    valid = df[col].notna().to_numpy(dtype=float)
    tmp = pd.DataFrame({"_v": v, "_n": valid})
    g = tmp.groupby([df[k] for k in keys], sort=False)
    csum = g["_v"].cumsum().to_numpy() - v
    ccnt = g["_n"].cumsum().to_numpy() - valid
    mean = np.where(ccnt > 0, csum / np.maximum(ccnt, 1e-9), np.nan)
    return mean, ccnt


def _shifted(df, keys, col, lag):
    return df.groupby(keys, sort=False)[col].shift(lag)


def _as_text(series):
    """
    どんな型でも .str が使える形にする。
    pandas 3 系では全てNaNのfloat列に astype(str) をかけてもfloatのままで、
    .str アクセサが AttributeError になるため。
    """
    return series.astype("string")


def _usable(series):
    """その列に中身があるか（全部空なら特徴量を作らない）。"""
    if series is None:
        return False
    s = _as_text(series)
    return s.notna().any() and (s.str.strip() != "").any() and \
        (~s.str.lower().isin(["nan", "none", "<na>"])).any()


def _smooth_rate(mean, cnt, prior, prior_weight=15.0):
    """出走数が少ない馬・騎手の勝率を事前分布に縮小(ベイズ平滑化)。"""
    mean = np.where(np.isnan(mean), prior, mean)
    return (mean * cnt + prior * prior_weight) / (cnt + prior_weight)


# ---------------------------------------------------------------- 本体
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    raw: 1行 = 1レース1頭。date 昇順である必要はない(内部でソート)。
    戻り値: 特徴量を付与した DataFrame。
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    # 同日内の順序を race_id で固定し、決定的にする
    df = df.sort_values(["date", "race_id", "horse_no"]).reset_index(drop=True)

    # 予想対象のレースは着順が未確定なので欠損になる。
    # pd.NA や文字列が混ざると列が object 型になり、そこから作る
    # prev1_finish などが学習時（float）と型が食い違ってしまう。
    df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")
    df["is_win"] = (df["finish_pos"] == 1).astype(float)
    df["is_top3"] = (df["finish_pos"] <= 3).astype(float)

    # ---- レース属性 ----
    df["dist_bucket"] = pd.cut(
        df["distance"], bins=DIST_BUCKET_EDGES, labels=DIST_BUCKET_LABELS
    ).astype(int)
    df["is_turf"] = (df["surface"] == "芝").astype(int)
    df["is_right"] = (df["turn"] == "右").astype(int)
    df["distance_z"] = (df["distance"] - 1800) / 400.0
    df["field_size"] = df.groupby("race_id")["horse_id"].transform("size")

    # ---- 馬番・枠 ----
    df["horse_no_rel"] = df["horse_no"] / df["field_size"]
    df["is_inner"] = (df["frame_no"] <= 3).astype(int)
    df["is_outer"] = (df["frame_no"] >= 7).astype(int)

    # ---- 基本属性 ----
    df["is_male"] = (df["sex"] == "牡").astype(int)
    df["is_filly"] = (df["sex"] == "牝").astype(int)
    df["weight_carried_rel"] = df["weight_carried"] - df.groupby("race_id")[
        "weight_carried"
    ].transform("mean")

    # ---- 出走間隔 ----
    prev_date = _shifted(df, "horse_id", "date", 1)
    df["days_since_last"] = (df["date"] - prev_date).dt.days
    df["is_layoff"] = (df["days_since_last"] > 120).fillna(False).astype(int)
    df["is_short_rest"] = (df["days_since_last"] <= 14).fillna(False).astype(int)
    df["days_since_last"] = df["days_since_last"].fillna(365).clip(0, 400)

    # ---- 前走〜3走前の結果(過去情報なので使用可) ----
    for lag in (1, 2, 3):
        df[f"prev{lag}_finish"] = _shifted(df, "horse_id", "finish_pos", lag)
        df[f"prev{lag}_field"] = _shifted(df, "horse_id", "field_size", lag)
        df[f"prev{lag}_finish_rel"] = df[f"prev{lag}_finish"] / df[f"prev{lag}_field"]
        df[f"prev{lag}_class"] = _shifted(df, "horse_id", "class_level", lag)
        if "last3f" in df.columns:
            df[f"prev{lag}_last3f"] = _shifted(df, "horse_id", "last3f", lag)

    df["class_change"] = df["class_level"] - df["prev1_class"]
    df["prev_distance"] = _shifted(df, "horse_id", "distance", 1)
    df["distance_change"] = (df["distance"] - df["prev_distance"]) / 400.0
    df["surface_change"] = (
        df["is_turf"] != _shifted(df, "horse_id", "is_turf", 1)
    ).astype(int)

    # ---- 馬の通算成績(過去のみ) ----
    win_mean, starts = _past_mean(df, ["horse_id"], "is_win")
    top3_mean, _ = _past_mean(df, ["horse_id"], "is_top3")
    df["horse_starts"] = starts
    df["horse_win_rate"] = _smooth_rate(win_mean, starts, 0.08, 8)
    df["horse_top3_rate"] = _smooth_rate(top3_mean, starts, 0.24, 8)
    df["is_debut"] = (starts == 0).astype(int)

    # ---- 条件別成績 ----
    for name, keys in [
        ("dist", ["horse_id", "dist_bucket"]),
        ("surf", ["horse_id", "is_turf"]),
        ("venue", ["horse_id", "venue"]),
    ]:
        m, c = _past_mean(df, keys, "is_top3")
        df[f"horse_{name}_top3_rate"] = _smooth_rate(m, c, 0.24, 6)
        df[f"horse_{name}_starts"] = c

    # ---- 騎手 ----
    jw, jc = _past_mean(df, ["jockey_id"], "is_win")
    jt, _ = _past_mean(df, ["jockey_id"], "is_top3")
    df["jockey_win_rate"] = _smooth_rate(jw, jc, 0.08, 60)
    df["jockey_top3_rate"] = _smooth_rate(jt, jc, 0.24, 60)
    df["jockey_starts"] = jc
    jvw, jvc = _past_mean(df, ["jockey_id", "venue"], "is_win")
    df["jockey_venue_win_rate"] = _smooth_rate(jvw, jvc, 0.08, 30)

    # 乗り替わり
    prev_jockey = _shifted(df, "horse_id", "jockey_id", 1)
    df["jockey_change"] = (df["jockey_id"] != prev_jockey).fillna(True).astype(int)

    # ---- 調教師 ----
    tw, tc = _past_mean(df, ["trainer_id"], "is_win")
    tt, _ = _past_mean(df, ["trainer_id"], "is_top3")
    df["trainer_win_rate"] = _smooth_rate(tw, tc, 0.08, 60)
    df["trainer_top3_rate"] = _smooth_rate(tt, tc, 0.24, 60)

    # ---- 血統(種牡馬 × 条件) 種牡馬IDがある場合のみ ----
    if "sire_id" in df.columns and df["sire_id"].notna().any():
        sw, sc = _past_mean(df, ["sire_id"], "is_top3")
        df["sire_top3_rate"] = _smooth_rate(sw, sc, 0.24, 100)
        ssw, ssc = _past_mean(df, ["sire_id", "is_turf"], "is_top3")
        df["sire_surface_top3_rate"] = _smooth_rate(ssw, ssc, 0.24, 60)
        sdw, sdc = _past_mean(df, ["sire_id", "dist_bucket"], "is_top3")
        df["sire_dist_top3_rate"] = _smooth_rate(sdw, sdc, 0.24, 60)

    # ---- 脚質(通過順) ----
    if "corner_pos" in df.columns and _usable(df["corner_pos"]):
        cp = _as_text(df["corner_pos"])
        first = pd.to_numeric(cp.str.split("-").str[0], errors="coerce")
        last = pd.to_numeric(cp.str.split("-").str[-1], errors="coerce")
        df["_c1_rel"] = first / df["field_size"]
        df["_c4_rel"] = last / df["field_size"]
        df["_pos_gain"] = df["_c1_rel"] - df["_c4_rel"]      # 正なら差してきた
        for lag in (1, 2):
            df[f"prev{lag}_c1_rel"] = _shifted(df, "horse_id", "_c1_rel", lag)
            df[f"prev{lag}_gain"] = _shifted(df, "horse_id", "_pos_gain", lag)
        m, c = _past_mean(df, ["horse_id"], "_c1_rel")
        df["run_style"] = m                                   # 小さいほど先行型
        mg, _ = _past_mean(df, ["horse_id"], "_pos_gain")
        df["closing_tendency"] = mg
        df.drop(columns=["_c1_rel", "_c4_rel", "_pos_gain"], inplace=True)

    # ---- 本来のペース（前半3F と 後半3F の差） ----
    # 前半が速い＝前傾ラップ＝先行馬に厳しい。後半が速い＝後傾＝差しに厳しい。
    if "pace_first3f" in df.columns and df["pace_first3f"].notna().any():
        df["pace_balance"] = df["pace_first3f"] - df["pace_last3f"]
        for lag in (1, 2):
            df[f"prev{lag}_pace_balance"] = _shifted(df, "horse_id", "pace_balance", lag)
        # その馬が経験したペースの平均と、そこでの成績
        m, c = _past_mean(df, ["horse_id"], "pace_balance")
        df["pace_balance_avg"] = np.where(c >= 2, m, np.nan)
        # 前傾ラップ（前半が速い）での複勝率と、後傾での複勝率
        fast_front = (df["pace_balance"] < 0).astype(float)
        df["_top3_front"] = np.where(fast_front > 0, df["is_top3"], np.nan)
        df["_top3_back"] = np.where(fast_front == 0, df["is_top3"], np.nan)
        for col, out in (("_top3_front", "pace_apt_front"), ("_top3_back", "pace_apt_back")):
            vals = df[col].to_numpy(dtype=float)
            ok = ~np.isnan(vals)
            mm, cc = _past_mean(df, ["horse_id"], col)
            df[out] = np.where(cc >= 2, mm, np.nan)
        df["pace_apt_gap"] = df["pace_apt_front"] - df["pace_apt_back"]
        df.drop(columns=["_top3_front", "_top3_back"], inplace=True)

    # ---- 賞金履歴 ----
    if "prize" in df.columns and df["prize"].notna().any():
        pr = df["prize"].fillna(0.0)
        g = pd.DataFrame({"_p": pr}).groupby(df["horse_id"], sort=False)["_p"]
        df["career_prize"] = g.cumsum() - pr
        df["prize_per_start"] = df["career_prize"] / np.maximum(df["horse_starts"], 1)
        for lag in (1, 2):
            df[f"prev{lag}_prize"] = _shifted(df, "horse_id", "prize", lag)

    # ---- 調教(あれば) ----
    if "training_time_last" in df.columns:
        df["training_time_rel"] = df["training_time_last"] - df.groupby("race_id")[
            "training_time_last"
        ].transform("mean")

    # ---- レース内相対化(強い馬が集まったレースかを表現) ----
    for c in ["horse_win_rate", "horse_top3_rate", "jockey_win_rate", "horse_starts"]:
        g = df.groupby("race_id")[c]
        df[f"{c}_rank"] = g.rank(ascending=False, method="min")
        df[f"{c}_rel"] = df[c] - g.transform("mean")

    # ---- 走破タイム由来（馬場差・タイム指数・ペース適性） ----
    df = add_time_features(df)

    # ---- 前日オッズ特徴量(モデルB専用) ----
    df = _add_odds_features(df)

    return df


def _add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    if "odds_prev_win" not in df.columns:
        return df
    o = df["odds_prev_win"].clip(lower=1.0)
    df["odds_prev_log"] = np.log(o)
    df["_inv_odds"] = 1.0 / o
    df["support_rate"] = df["_inv_odds"] / df.groupby("race_id")["_inv_odds"].transform("sum")
    df.drop(columns=["_inv_odds"], inplace=True)
    df["odds_rank"] = df.groupby("race_id")["odds_prev_win"].rank(method="min")
    top_odds = df.groupby("race_id")["odds_prev_win"].transform("min")
    df["odds_ratio_to_top"] = np.log(o / top_odds.clip(lower=1.0))
    df["support_top3_share"] = df.groupby("race_id")["support_rate"].transform(
        lambda x: x.nlargest(3).sum()
    )
    if "odds_prev_place_low" in df.columns:
        df["odds_prev_place_log"] = np.log(df["odds_prev_place_low"].clip(lower=1.0))
    else:
        df["odds_prev_place_log"] = np.nan
    return df
