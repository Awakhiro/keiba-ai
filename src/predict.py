"""
前日予想の実行。

使い方(概念):
    history = 過去の全レース結果(finish_pos あり)
    entries = 明日の出走表(finish_pos は NaN、前日オッズあり)
    → 2つを縦に結合して特徴量を作り、entries 部分だけ予測する。
      (過去成績の集計に history が必要なため、必ず結合してから build_features する)
"""

import numpy as np
import pandas as pd

from .features import build_features
from .models import make_models
from .value import add_value_columns
from .confidence import race_confidence, confidence_score, pick_confident_races


class KeibaPredictor:
    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.grader = None

    def train(self, history: pd.DataFrame):
        feat = build_features(history)
        feat = feat[feat["finish_pos"].notna()]
        self.model_a, self.model_b = make_models()
        self.model_a.fit(feat)
        self.model_b.fit(feat)
        return self

    def predict_day(self, history: pd.DataFrame, entries: pd.DataFrame):
        """entries(予想対象)に対して2モデルの予測を返す。"""
        entries = entries.copy()
        if "finish_pos" not in entries.columns:
            entries["finish_pos"] = np.nan
        target_races = set(entries["race_id"].astype(str))
        # 同じレースが過去データにもある場合（結果確定後の再実行など）、
        # 両方が残ると1レースに2組の行ができてしまう。出馬表側を優先する。
        hist = history[~history["race_id"].astype(str).isin(target_races)]
        dropped = len(history) - len(hist)
        if dropped:
            print(f"過去データから重複 {dropped}行を除外しました")
        combined = pd.concat([hist, entries], ignore_index=True)
        feat = build_features(combined)
        t = feat[feat["race_id"].astype(str).isin(target_races)].copy()

        t["p_top3"] = self.model_a.predict(t)        # モデルA: 複勝圏確率
        t["p_win_pure"] = self.model_b.predict(t)    # モデルB: 能力のみの勝率
        t = add_value_columns(t)                     # 市場と比較して妙味を算出
        t["p_win"] = t["p_blend"]                    # 買い目の確率は混合後を使う
        if "odds_prev_place_low" in t.columns:
            t["ev_place"] = t["p_top3"] * t["odds_prev_place_low"]
        t["rank_a"] = t.groupby("race_id")["p_top3"].rank(ascending=False, method="min")
        t["rank_b"] = t.groupby("race_id")["p_win"].rank(ascending=False, method="min")
        return t

    def build_report(self, pred: pd.DataFrame, min_grade="B", top_n=None,
                     ev_threshold=1.15):
        """勝負レースを選び、2パターンの買い目を作る。"""
        conf = race_confidence(pred)
        conf["conf_score"] = confidence_score(conf)
        from .confidence import ConfidenceGrader
        grader = self.grader or ConfidenceGrader()
        picked = pick_confident_races(conf, grader, min_grade=min_grade, top_n=top_n)

        reports = []
        for _, row in picked.iterrows():
            g = pred[pred["race_id"] == row["race_id"]]

            a = g.sort_values("p_top3", ascending=False).head(4)
            b = g.sort_values("p_win", ascending=False).head(4)
            ev_horses = g[g["ev_win"] >= ev_threshold].sort_values("ev_win", ascending=False)

            reports.append({
                "race_id": row["race_id"],
                "grade": row["grade"],
                "conf_score": round(row["conf_score"], 1),
                "頭数": int(row["field_size"]),
                "両モデル一致": bool(row["model_agree"]),
                "A_的中率重視": [
                    {"馬番": int(r.horse_no), "複勝圏確率": round(r.p_top3, 3)}
                    for r in a.itertuples()
                ],
                "B_オッズ考慮": [
                    {"馬番": int(r.horse_no), "勝率": round(r.p_win, 3),
                     "前日odds": round(r.odds_prev_win, 1),
                     "期待値": round(r.ev_win, 2)}
                    for r in b.itertuples()
                ],
                "妙味馬(期待値>%.2f)" % ev_threshold: [
                    {"馬番": int(r.horse_no), "期待値": round(r.ev_win, 2),
                     "odds": round(r.odds_prev_win, 1)}
                    for r in ev_horses.itertuples()
                ],
            })
        return reports


def format_report(reports):
    """コンソール向けに整形。"""
    lines = []
    for r in reports:
        lines.append("=" * 62)
        lines.append(f"[{r['grade']}級] {r['race_id']}  自信度 {r['conf_score']}  "
                     f"{r['頭数']}頭  両モデル一致={'○' if r['両モデル一致'] else '×'}")
        lines.append("-" * 62)
        lines.append("  ■ モデルA(的中率重視・オッズ非考慮) 複勝軸候補")
        for h in r["A_的中率重視"]:
            lines.append(f"      {h['馬番']:>2}番   複勝圏確率 {h['複勝圏確率']:.1%}")
        lines.append("  ■ モデルB(オッズ考慮・期待値重視) 単勝候補")
        for h in r["B_オッズ考慮"]:
            lines.append(f"      {h['馬番']:>2}番   勝率 {h['勝率']:.1%}  "
                         f"odds {h['前日odds']:>5.1f}  期待値 {h['期待値']:.2f}")
        key = [k for k in r if k.startswith("妙味馬")][0]
        if r[key]:
            s = "  ".join(f"{h['馬番']}番(EV {h['期待値']:.2f})" for h in r[key])
            lines.append(f"  ★ 妙味: {s}")
        else:
            lines.append("  ★ 妙味: 単勝で期待値がしきい値を超える馬なし → 単勝は見送り推奨")
    return "\n".join(lines)
