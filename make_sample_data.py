"""
動作確認用のダミーデータ生成。

実データが揃うまでにパイプラインが正しく動くかを確認するためのもの。
各馬に潜在能力を持たせ、能力+ノイズで着順を決め、オッズは
「真の勝率にノイズと控除率をかけたもの」として生成する。
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
_S1 = ["サン","メイ","キタ","トウ","ヒシ","ゴール","シルク","ダイワ","ナリタ","マイ","レッド","カレン","ロード","ウイン","エイ"]
_S2 = ["ブライト","フェスタ","クイーン","ホープ","スカイ","リバー","マーチ","ノヴァ","グロウ","テソーロ","ルミナス","カイザー","フローラ","ゼウス"]


def _name(i):
    return _S1[i % len(_S1)] + _S2[(i // len(_S1)) % len(_S2)] + ("" if i < 200 else str(i % 97))
VENUES = ["東京", "中山", "阪神", "京都", "中京"]
SEXES = ["牡", "牝", "セ"]
TAKEOUT = 0.20


def generate(n_days=200, venues_per_day=2, races_per_day=12, start="2021-01-01"):
    """
    実際の開催に合わせて「1日 × 数場 × 12レース」の形で作る。
    各開催日・各場に固有の馬場差（速い日・遅い日）を持たせるので、
    1〜6Rから7〜12Rの馬場を推し量る仕組みを検証できる。
    """
    n_horses, n_jockeys, n_trainers, n_sires = 4000, 120, 200, 60
    horse_ability = RNG.normal(0, 1.0, n_horses)
    jockey_skill = RNG.normal(0, 0.35, n_jockeys)
    trainer_skill = RNG.normal(0, 0.22, n_trainers)
    sire_turf = RNG.normal(0, 0.3, n_sires)
    horse_sire = RNG.integers(0, n_sires, n_horses)
    horse_age0 = RNG.integers(2, 6, n_horses)
    horse_sex = RNG.integers(0, 3, n_horses)

    day0 = pd.to_datetime(start)
    # 土日開催を想定して週2日ずつ進める
    meeting_days = []
    d, w = day0, 0
    while len(meeting_days) < n_days:
        meeting_days.append(d + pd.Timedelta(days=(w % 2) + 5))
        if w % 2 == 1:
            d = d + pd.Timedelta(days=7)
        w += 1

    rows, truth = [], []
    rid = 0
    for day in meeting_days:
        todays = RNG.choice(VENUES, size=venues_per_day, replace=False)
        for venue in todays:
            # その日その場の馬場差。全レースに共通して効く。
            day_bias = float(RNG.normal(0, 0.55))
            surface_of_day = {}
            for r in range(1, races_per_day + 1):
                n = int(RNG.integers(8, 19))
                horses = RNG.choice(n_horses, size=n, replace=False)
                surface = RNG.choice(["芝", "ダ"], p=[0.6, 0.4])
                distance = int(RNG.choice([1200, 1400, 1600, 1800, 2000, 2400]))
                cls = int(RNG.integers(1, 6))
                jockeys = RNG.choice(n_jockeys, size=n, replace=False)
                trainers = RNG.integers(0, n_trainers, n)
                surface_of_day.setdefault(surface, day_bias)

                strength = (
                    horse_ability[horses]
                    + jockey_skill[jockeys]
                    + trainer_skill[trainers]
                    + (sire_turf[horse_sire[horses]] if surface == "芝" else 0)
                    - 0.02 * (np.arange(n) + 1) * 0.3
                )
                score = strength + RNG.normal(0, 1.15, n)
                finish = np.argsort(np.argsort(-score)) + 1

                market = strength + RNG.normal(0, 0.55, n)
                p_market = np.exp(market) / np.exp(market).sum()
                odds = np.clip((1 - TAKEOUT) / np.maximum(p_market, 1e-4), 1.1, 300).round(1)
                place_odds = np.clip(1 + (odds - 1) * 0.28, 1.0, 60).round(1)

                base_sec = distance / 16.5 + (1.5 if surface == "ダ" else 0.0)
                times = (base_sec - day_bias
                         - (score - score.max()) * 0.22 + RNG.normal(0, 0.12, n))
                ftime = [f"{int(t)//60}:{t - 60*(int(t)//60):04.1f}" for t in times]

                payout_win = np.where(finish == 1, odds * 100, 0.0)
                payout_place = np.where(finish <= 3, place_odds * 100, 0.0)

                race_id = f"R{rid:06d}"
                truth.append({"race_id": race_id, "date": day, "venue": venue,
                              "surface": surface, "race_no": r, "day_bias": day_bias})
                rid += 1

                for k in range(n):
                    h = horses[k]
                    rows.append({
                        "race_id": race_id, "date": day, "race_no": r,
                        "horse_id": int(h), "horse_name": _name(int(h)),
                        "venue": venue, "surface": surface, "distance": distance,
                        "turn": "右" if venue in ("中山", "阪神") else "左",
                        "class_level": cls,
                        "going_forecast": RNG.choice(["良", "稍重", "重"], p=[.75, .18, .07]),
                        "horse_no": k + 1, "frame_no": min(8, k // 2 + 1),
                        "age": int(horse_age0[h] + (day.year - 2021)),
                        "sex": SEXES[horse_sex[h]],
                        "weight_carried": float(RNG.choice([53, 54, 55, 56, 57])),
                        "jockey_id": int(jockeys[k]), "trainer_id": int(trainers[k]),
                        "sire_id": int(horse_sire[h]),
                        "training_time_last": float(RNG.normal(52, 1.2)),
                        "odds_prev_win": float(odds[k]),
                        "odds_prev_place_low": float(place_odds[k]),
                        "finish_pos": int(finish[k]), "finish_time": ftime[k],
                        "payout_win": float(payout_win[k]),
                        "payout_place": float(payout_place[k]),
                    })
    df = pd.DataFrame(rows)
    df.attrs["truth"] = pd.DataFrame(truth)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("sample_races.csv", index=False)
    print(f"生成: {len(df)}行 / {df['race_id'].nunique()}レース")
