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
    """
    2つのモデルで予想する。

    本命 … 全レースで学習した能力推定
    中穴 … 2着以内に6番人気以下が来たレースで学習
    穴   … 馬連が20倍以上だったレースで学習

    3,555レースの検証で、いずれも1点あたり配当10倍以上の買い目に絞った場合:
        本命 75.6% / 中穴 83.8% / 穴 84.0%（的中 17.1% / 16.0% / 15.8%）

    荒れたレースだけで学習したモデルが本命を上回る傾向は、期間を変えても再現した。
    ただしどの閾値（何番人気・何倍）が最良かは期間によって入れ替わり、安定しない。
    """

    # 中穴: 2着以内にこの人気以下が来たレースで学習
    UPSET_POPULARITY = 6
    UPSET_MAX_PLACE = 2
    # 穴: 馬連がこの倍率以上だったレースで学習
    LONGSHOT_ODDS = 20.0

    def __init__(self, high_odds=None):
        self.model_a = None
        self.model_b = None
        self.model_mid_a = None       # 中穴
        self.model_mid_b = None
        self.model_long_a = None      # 穴
        self.model_long_b = None
        self.high_odds = high_odds or self.LONGSHOT_ODDS
        self.grader = None
        self.lam2 = 0.81      # 順位割引。学習時に検証データから推定して上書きする
        self.lam3 = 0.65

    def train(self, history: pd.DataFrame, payouts=None):
        feat = build_features(history)
        feat = feat[feat["finish_pos"].notna()]
        self.model_a, self.model_b = make_models()
        self.model_a.fit(feat)
        self.model_b.fit(feat)

        # 中穴: 2着以内に人気薄が来たレースだけで学習
        self.model_mid_a, self.model_mid_b = self._fit_subset(
            feat, self._upset_races(feat), "中穴")

        # 穴: 馬連が高配当だったレースだけで学習
        if payouts is not None and len(payouts):
            hot = {str(r) for r, v in
                   payouts.set_index("race_id").to_dict("index").items()
                   if v.get("payout_umaren")
                   and v["payout_umaren"] / 100.0 >= self.high_odds}
            self.model_long_a, self.model_long_b = self._fit_subset(
                feat, hot, "穴")
        else:
            print("払戻データが無いため穴モデルは作りません")
        return self

    @staticmethod
    def _fit_subset(feat, race_ids, label, min_races=500):
        """
        指定したレースだけで学習する。対象が少なすぎるときは作らない。
        """
        from .models import RaceProbModel

        if not race_ids:
            return None, None
        sub = feat[feat["race_id"].astype(str).isin(race_ids)]
        n = sub["race_id"].nunique()
        if n < min_races:
            print(f"{label}モデル: 対象が{n}レースと少ないため作りません")
            return None, None
        ma = RaceProbModel(f"A_{label}", "is_top3", use_odds=False,
                           expected_sum=3.0)
        mb = RaceProbModel(f"B_{label}", "is_win", use_odds=False,
                           expected_sum=1.0)
        ma.fit(sub)
        mb.fit(sub)
        print(f"{label}モデルを学習: {n:,}レース")
        return ma, mb

    @classmethod
    def _upset_races(cls, feat):
        """2着以内に人気薄が来たレースのIDを集める。"""
        out = set()
        for race_id, g in feat.groupby("race_id", sort=False):
            pop = None
            if "popularity_final" in g.columns:
                v = g["popularity_final"]
                # 全馬同じ値なら壊れているので使わない
                if v.notna().all() and v.nunique() > 1:
                    pop = v.to_numpy()
            if pop is None:
                col = ("odds_win_final" if "odds_win_final" in g.columns
                       else "odds_prev_win")
                if col not in g.columns:
                    continue
                pop = g[col].rank(method="min").to_numpy()
            fin = g["finish_pos"].to_numpy(dtype=float)
            m = (fin <= cls.UPSET_MAX_PLACE) & ~np.isnan(fin)
            if m.any() and np.nanmax(pop[m]) >= cls.UPSET_POPULARITY:
                out.add(str(race_id))
        return out

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

        t["p_top3"] = self.model_a.predict(t)        # 当てにいく側の複勝圏確率
        t["p_win_pure"] = self.model_b.predict(t)    # 能力のみの勝率
        t = add_value_columns(t)                     # 市場と比較して妙味を算出
        t["p_win"] = t["p_blend"]                    # 買い目の確率は混合後を使う

        # 中穴・穴それぞれの見立ても添える
        if self.model_mid_a is not None:
            t["p_top3_mid"] = self.model_mid_a.predict(t)
            t["p_win_mid"] = self.model_mid_b.predict(t)
        if self.model_long_a is not None:
            t["p_top3_long"] = self.model_long_a.predict(t)
            t["p_win_long"] = self.model_long_b.predict(t)
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
