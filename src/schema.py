"""
列定義とデータリーク防止ルール。

このファイルの役割は「前日までに知り得ない情報」を機械的に排除すること。
競馬予想モデルで最も多い失敗は、当日情報(確定オッズ・馬体重・上がり3F等)が
特徴量に紛れ込んで、検証では高精度なのに実運用で全く当たらないケース。
"""

ID_COLS = ["race_id", "horse_id", "date"]

# --- 当日にならないと確定しない列(特徴量に入れてはいけない) ---
LEAK_COLS = [
    "finish_pos",          # 着順
    "finish_time",         # 走破タイム
    "last3f",              # 上がり3F
    "corner_pos",          # 通過順
    "odds_win_final",      # 確定単勝オッズ
    "popularity_final",    # 確定人気
    "horse_weight",        # 馬体重(当日発表)
    "horse_weight_diff",   # 馬体重増減
    "going_actual",        # 当日の実馬場状態
    "payout_win",
    "payout_place",
    "payout_wide",
    "prize",                # 当該レースの獲得賞金＝着順そのもの
    "margin",               # 着差
    # --- 走破タイム由来。当該レースの結果そのものなので絶対に入れない ---
    "finish_time",
    "time_sec",
    "speed_fig",            # そのレースでのタイム指数
    "race_dev",             # そのレースの勝ちタイムの基準差
    "race_dev_adj",
    "day_dev",              # その日全体の馬場差（後のレースを含む）
    # --- 戦績ページから補完した列。当該レースの結果なので入れない ---
    "pace_first3f",
    "pace_last3f",
    "pace_balance",
]

TARGET_COLS = ["is_win", "is_top3"]

# --- 前日オッズ由来の特徴量(モデルBのみ使用) ---
ODDS_FEATURES = [
    "odds_prev_log",         # log(前日単勝オッズ)
    "support_rate",          # 支持率 = (1/odds) をレース内で正規化
    "odds_rank",             # 前日オッズ順位
    "odds_ratio_to_top",     # 1番人気オッズとの比
    "support_top3_share",    # 上位3頭の支持率合計(レース単位・混戦度)
    "odds_prev_place_log",   # log(前日複勝オッズ下限)
]

# --- 生データに必須の列 ---
REQUIRED_RAW = [
    "race_id", "date", "horse_id",
    "venue", "surface", "distance", "turn", "class_level", "field_size",
    "horse_no", "frame_no", "age", "sex", "weight_carried",
    "jockey_id", "trainer_id", "sire_id",
    "going_forecast",
    "odds_prev_win",
    "finish_pos",
]

CATEGORICAL_RAW = [
    "venue", "surface", "turn", "sex", "going_forecast",
]


def feature_columns(df, use_odds: bool):
    """学習に使う列を安全に選ぶ。リーク列は常に除外。"""
    banned = set(ID_COLS) | set(LEAK_COLS) | set(TARGET_COLS)
    banned |= {"odds_prev_win", "odds_prev_place_low", "odds_prev_place_high"}
    if not use_odds:
        banned |= set(ODDS_FEATURES)

    cols = []
    for c in df.columns:
        if c in banned:
            continue
        if df[c].dtype.kind in "ifb":
            cols.append(c)
    return cols


def assert_no_leak(feature_cols, use_odds: bool):
    bad = set(feature_cols) & set(LEAK_COLS)
    if bad:
        raise ValueError(f"リーク列が特徴量に含まれています: {sorted(bad)}")
    if not use_odds:
        bad2 = set(feature_cols) & set(ODDS_FEATURES)
        if bad2:
            raise ValueError(f"オッズ非使用モデルにオッズ列: {sorted(bad2)}")
    return True
