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


def generate(n_races=3000, start="2021-01-01"):
    n_horses, n_jockeys, n_trainers, n_sires = 4000, 120, 200, 60
    horse_ability = RNG.normal(0, 1.0, n_horses)
    jockey_skill = RNG.normal(0, 0.35, n_jockeys)
    trainer_skill = RNG.normal(0, 0.22, n_trainers)
    sire_turf = RNG.normal(0, 0.3, n_sires)
    horse_sire = RNG.integers(0, n_sires, n_horses)
    horse_age0 = RNG.integers(2, 6, n_horses)
    horse_sex = RNG.integers(0, 3, n_horses)

    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.sort(RNG.integers(0, 1500, n_races)), unit="D"
    )
    rows = []
    for i in range(n_races):
        n = int(RNG.integers(8, 19))
        horses = RNG.choice(n_horses, size=n, replace=False)
        surface = RNG.choice(["芝", "ダ"], p=[0.6, 0.4])
        distance = int(RNG.choice([1200, 1400, 1600, 1800, 2000, 2400]))
        venue = RNG.choice(VENUES)
        cls = int(RNG.integers(1, 6))
        jockeys = RNG.choice(n_jockeys, size=n, replace=False)
        trainers = RNG.integers(0, n_trainers, n)

        strength = (
            horse_ability[horses]
            + jockey_skill[jockeys]
            + trainer_skill[trainers]
            + (sire_turf[horse_sire[horses]] if surface == "芝" else 0)
            - 0.02 * (np.arange(n) + 1) * 0.3          # 外枠わずかに不利
        )
        noise = RNG.normal(0, 1.15, n)                  # 競馬の不確実性
        score = strength + noise
        finish = np.argsort(np.argsort(-score)) + 1

        # 市場オッズ: 真の実力に別ノイズを乗せた推定 + 控除率
        market = strength + RNG.normal(0, 0.55, n)
        p_market = np.exp(market) / np.exp(market).sum()
        odds = np.clip((1 - TAKEOUT) / np.maximum(p_market, 1e-4), 1.1, 300).round(1)
        place_odds = np.clip(1 + (odds - 1) * 0.28, 1.0, 60).round(1)

        payout_win = np.where(finish == 1, odds * 100, 0.0)
        payout_place = np.where(finish <= 3, place_odds * 100, 0.0)

        for k in range(n):
            h = horses[k]
            rows.append({
                "race_id": f"R{i:06d}",
                "date": dates[i],
                "horse_id": int(h),
                "horse_name": _name(int(h)),
                "venue": venue,
                "surface": surface,
                "distance": distance,
                "turn": "右" if venue in ("中山", "阪神") else "左",
                "class_level": cls,
                "going_forecast": RNG.choice(["良", "稍重", "重"], p=[0.75, 0.18, 0.07]),
                "horse_no": k + 1,
                "frame_no": min(8, k // 2 + 1),
                "age": int(horse_age0[h] + (dates[i].year - 2021)),
                "sex": SEXES[horse_sex[h]],
                "weight_carried": float(RNG.choice([53, 54, 55, 56, 57])),
                "jockey_id": int(jockeys[k]),
                "trainer_id": int(trainers[k]),
                "sire_id": int(horse_sire[h]),
                "training_time_last": float(RNG.normal(52, 1.2)),
                "odds_prev_win": float(odds[k]),
                "odds_prev_place_low": float(place_odds[k]),
                "finish_pos": int(finish[k]),
                "payout_win": float(payout_win[k]),
                "payout_place": float(payout_place[k]),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("sample_races.csv", index=False)
    print(f"生成: {len(df)}行 / {df['race_id'].nunique()}レース")
