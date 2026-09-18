"""
走破タイムから馬場差とタイム指数を作る。

スクレイピング時に取得しているのに使っていなかった「タイム」列を活用する。
通過順と上がり3Fが取得できないため本来のペース（前半3Fの速さ）は計算できないが、
決着タイムの速さは分かるので、そこから次の3つを作る。

1. 基準タイム
   コース・距離・馬場種別ごとの勝ちタイムの平均。過去のレースだけから作る。

2. 馬場差（当日補正）
   同じ日・同じ場・同じ馬場種別で、**それより前に行われたレース**の
   勝ちタイムが基準よりどれだけ速いかの平均。
   1〜6Rの結果から7〜12Rの馬場状態を推し量る、という発想をそのまま実装したもの。
   前のレースが無い（1R など）場合は、その場の直近開催日の馬場差で代用する。
   当日のレースしか使わないので、予想時点で必ず手に入る情報になる。

3. タイム指数
   （基準タイム − 走破タイム − 馬場差）を秒で表したもの。プラスほど速い。
   着順は相手次第で変わるが、タイム指数は相手に依らない絶対的な物差しになる。

さらに、過去のタイム指数から「速い決着に強いか、遅い決着に強いか」を集計する。
これが本来のペース適性の代わりになる。

リーク防止:
    基準タイムは過去のレースのみ、馬場差は同日でも自分より前のレースのみ。
    馬のタイム指数は前走以前のみ（shift 済み）。
"""

import re

import numpy as np
import pandas as pd

_TIME = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$")


def parse_time_sec(s):
    """'1:27.9' を 87.9 秒にする。'27.9' のような形も受ける。"""
    if s is None:
        return np.nan
    t = str(s).strip()
    m = _TIME.match(t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        v = float(t)
        return v if 30 <= v <= 400 else np.nan
    except ValueError:
        return np.nan


def _past_mean_by(keys, values, valid):
    """keys でグループ化し、自分より前の行だけの平均を返す（要 時系列ソート）。"""
    v = np.where(valid, values, 0.0)
    n = valid.astype(float)
    tmp = pd.DataFrame({"_v": v, "_n": n})
    g = tmp.groupby(keys, sort=False)
    csum = g["_v"].cumsum().to_numpy() - v
    ccnt = g["_n"].cumsum().to_numpy() - n
    return np.where(ccnt > 0, csum / np.maximum(ccnt, 1e-9), np.nan), ccnt


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    走破タイム由来の特徴量を付ける。
    df は 1行=1レース1頭。finish_time 列（文字列）が必要。
    """
    d = df.copy()
    if "finish_time" not in d.columns:
        return d

    d["date"] = pd.to_datetime(d["date"])
    if "race_no" not in d.columns:
        d["race_no"] = d["race_id"].astype(str).str[-2:].pipe(
            pd.to_numeric, errors="coerce").fillna(0).astype(int)
    d = d.sort_values(["date", "race_no", "race_id", "horse_no"]).reset_index(drop=True)
    d["time_sec"] = d["finish_time"].map(parse_time_sec)
    if d["time_sec"].notna().sum() < 100:
        return d          # タイムが取れていないなら何もしない

    # ---------- レース単位の表を作る ----------
    win = (d[d["finish_pos"] == 1]
           .groupby("race_id", as_index=False)
           .agg(win_time=("time_sec", "min")))
    races = (d.groupby("race_id", as_index=False)
               .agg(date=("date", "first"), race_no=("race_no", "first"),
                    venue=("venue", "first"), surface=("surface", "first"),
                    distance=("distance", "first"),
                    class_level=("class_level", "first"))
               .merge(win, on="race_id", how="left")
               .sort_values(["date", "race_no", "race_id"])
               .reset_index(drop=True))

    valid = races["win_time"].notna().to_numpy()
    wt = races["win_time"].fillna(0.0).to_numpy()

    # ---------- 1. 基準タイム（過去のみ） ----------
    course_keys = [races["venue"], races["surface"], races["distance"]]
    base, base_n = _past_mean_by(course_keys, wt, valid)
    races["base_time"] = base
    races["base_n"] = base_n

    # クラス差ぶんの補正（上のクラスほど速い）
    races["race_dev"] = races["base_time"] - races["win_time"]   # +なら基準より速い
    cls_keys = [races["venue"], races["surface"], races["distance"],
                races["class_level"]]
    cls_adj, _ = _past_mean_by(cls_keys, races["race_dev"].fillna(0.0).to_numpy(),
                               races["race_dev"].notna().to_numpy())
    races["race_dev_adj"] = races["race_dev"] - np.nan_to_num(cls_adj)

    # ---------- 2. 馬場差（同日・自分より前のレースのみ） ----------
    day_keys = [races["date"], races["venue"], races["surface"]]
    dev = races["race_dev_adj"].fillna(0.0).to_numpy()
    dev_ok = races["race_dev_adj"].notna().to_numpy()
    today_var, today_n = _past_mean_by(day_keys, dev, dev_ok)
    races["variant_today"] = today_var
    races["variant_today_n"] = today_n

    # その日まだ前例が無い場合、同じ場の直近開催日の馬場差で代用する
    day = (races[races["race_dev_adj"].notna()]
           .groupby(["date", "venue", "surface"], as_index=False)
           .agg(day_dev=("race_dev_adj", "mean")))
    day = day.sort_values("date")
    day["prev_day_dev"] = (day.groupby(["venue", "surface"])["day_dev"].shift(1))
    races = races.merge(day[["date", "venue", "surface", "day_dev", "prev_day_dev"]],
                        on=["date", "venue", "surface"], how="left")
    races["track_variant"] = races["variant_today"].fillna(races["prev_day_dev"]).fillna(0.0)
    races["variant_is_today"] = races["variant_today"].notna().astype(int)

    # ---------- 3. タイム指数 ----------
    keep = ["race_id", "base_time", "base_n", "race_dev", "race_dev_adj",
            "track_variant", "variant_today_n", "variant_is_today", "day_dev"]
    d = d.merge(races[keep], on="race_id", how="left")

    # その馬のそのレースでの指数（過去レースの評価に使う。day_dev は当日全体なので
    # 自分より後のレースを含むが、これは「過去のレースを振り返って評価する」用途に限る）
    d["speed_fig"] = (d["base_time"] - d["time_sec"]) - d["day_dev"].fillna(0.0)
    d.loc[d["base_n"].fillna(0) < 20, "speed_fig"] = np.nan

    # ---------- 馬ごとの集計（前走以前のみ） ----------
    d = d.sort_values(["date", "race_no", "race_id", "horse_no"]).reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)["speed_fig"]
    d["speed_last1"] = g.shift(1)
    d["speed_last2"] = g.shift(2)
    fig = d["speed_fig"].to_numpy()
    ok = d["speed_fig"].notna().to_numpy()
    avg, cnt = _past_mean_by([d["horse_id"]], np.nan_to_num(fig), ok)
    d["speed_avg"] = avg
    d["speed_n"] = cnt
    d["speed_best3"] = (g.shift(1).rolling(3, min_periods=1).max()
                        if hasattr(g.shift(1), "rolling") else np.nan)
    # rolling は groupby 経由で計算し直す
    d["speed_best3"] = (d.groupby("horse_id", sort=False)["speed_fig"]
                          .transform(lambda s: s.shift(1).rolling(5, min_periods=1).max()))
    d["speed_trend"] = d["speed_last1"] - d["speed_avg"]

    # ---------- ペース適性の代わり（速い決着/遅い決着への強さ） ----------
    fast = (d["race_dev_adj"] > 0).astype(float)
    d["_top3_fast"] = np.where(fast > 0, d.get("is_top3", np.nan), np.nan)
    d["_top3_slow"] = np.where(fast == 0, d.get("is_top3", np.nan), np.nan)
    for col, out in (("_top3_fast", "apt_fast"), ("_top3_slow", "apt_slow")):
        vals = d[col].to_numpy(dtype=float)
        okv = ~np.isnan(vals)
        m, c = _past_mean_by([d["horse_id"]], np.nan_to_num(vals), okv)
        d[out] = np.where(c >= 2, m, np.nan)
    d["apt_gap"] = d["apt_fast"] - d["apt_slow"]
    d.drop(columns=["_top3_fast", "_top3_slow"], inplace=True)

    # ---------- レース単位の想定水準 ----------
    grp = d.groupby("race_id")["speed_avg"]
    d["race_speed_level"] = grp.transform("mean")
    d["speed_rel"] = d["speed_avg"] - d["race_speed_level"]
    d["speed_rank"] = grp.rank(ascending=False, method="min")
    d["speed_spread"] = grp.transform("std")

    # 当日の馬場が速いか遅いか（予想時点で分かる情報）
    d["variant_fast"] = (d["track_variant"] > 0).astype(int)

    # ---------- 想定ペース × 各馬の適性 ----------
    # 出走馬の指数の平均から、そのレースがどのくらい速い決着になりそうかを見積もる。
    # 同じコース・距離の中で相対化するので、条件差に引きずられない。
    key = [d["venue"], d["surface"], d["distance"]]
    lvl = d["race_speed_level"]
    mean_lvl, n_lvl = _past_mean_by(key, lvl.fillna(0.0).to_numpy(),
                                    lvl.notna().to_numpy())
    d["pace_expect"] = np.where(n_lvl >= 20, lvl - mean_lvl, 0.0)

    # 速い決着に強い馬（apt_gap が大きい）が、速くなりそうなレースに出ると有利。
    # 適性と想定ペースの積で、その噛み合わせを表す。
    d["pace_fit"] = d["apt_gap"].fillna(0.0) * d["pace_expect"]
    # 当日の馬場が速いかどうかとの噛み合わせも同様に。
    d["variant_fit"] = d["apt_gap"].fillna(0.0) * d["track_variant"]
    # 出走馬の指数のばらつきが小さい＝力が拮抗＝ペースが上がりやすい
    d["field_evenness"] = -d["speed_spread"].fillna(0.0)

    return d
