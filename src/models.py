"""
2種類のモデル。

  ModelA (的中率重視 / オッズ非考慮)
      目的変数 = 3着以内(is_top3)
      オッズを一切見ないので、市場と独立した「純粋な能力評価」になる。
      複勝・ワイドなど的中率の高い券種向け。

  ModelB (オッズ考慮 / 期待値重視)
      目的変数 = 1着(is_win)
      前日オッズを特徴量に含めるので市場情報を取り込み、単勝勝率の推定精度が上がる。
      推定勝率 × オッズ = 期待値 を計算して、期待値がプラスの馬だけ買う判断に使う。

どちらも「レース内で確率の合計が一定になるよう正規化」する。
(単勝は合計1.0、複勝圏は合計3.0)
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .schema import feature_columns, assert_no_leak

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # 本番環境では lightgbm を入れることを推奨
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False


LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.04,
    num_leaves=48,
    min_child_samples=60,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    verbose=-1,
)


class RaceProbModel:
    """レース単位で確率を正規化する二値分類モデル。"""

    def __init__(self, name, target, use_odds, expected_sum, n_estimators=700):
        self.name = name
        self.target = target
        self.use_odds = use_odds
        self.expected_sum = expected_sum
        self.n_estimators = n_estimators
        self.feature_cols = None
        self.model = None
        self.calibrator = None

    # ---------------------------------------------------------------- 学習
    def fit(self, train: pd.DataFrame, calib_frac=0.15):
        self.feature_cols = feature_columns(train, self.use_odds)
        assert_no_leak(self.feature_cols, self.use_odds)

        # 時系列の末尾をキャリブレーション用に取り分ける
        dates = np.sort(train["date"].unique())
        cut = dates[int(len(dates) * (1 - calib_frac))] if len(dates) > 10 else dates[-1]
        tr = train[train["date"] < cut]
        ca = train[train["date"] >= cut]
        if len(tr) < 500 or len(ca) < 100:
            tr, ca = train, train

        X, y = tr[self.feature_cols], tr[self.target]
        if HAS_LGB:
            self.model = lgb.LGBMClassifier(n_estimators=self.n_estimators, **LGB_PARAMS)
            self.model.fit(X, y)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=2.0,
            )
            self.model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))

        # 出力確率を実測頻度に合わせる(等張回帰)
        raw = self._raw_predict(ca)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999)
        self.calibrator.fit(raw, ca[self.target].to_numpy(dtype=float))
        return self

    # ------------------------------------------------------------ 推論
    def _raw_predict(self, df):
        X = df[self.feature_cols]
        if HAS_LGB:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X.to_numpy(dtype=float))[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """レース内正規化済みの確率を返す。"""
        p = self._raw_predict(df)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        p = np.clip(p, 1e-5, 0.999)
        s = pd.Series(p, index=df.index)
        total = s.groupby(df["race_id"]).transform("sum")
        norm = s / total * self.expected_sum
        # 複勝確率が1を超えないように
        return norm.clip(upper=0.98)

    def importance(self, top=25):
        if HAS_LGB:
            imp = self.model.feature_importances_
        else:
            return pd.DataFrame({"feature": self.feature_cols})
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )


def make_models():
    """2パターンのモデルを生成。"""
    model_a = RaceProbModel(
        name="A_的中率重視(オッズ非考慮)",
        target="is_top3",
        use_odds=False,
        expected_sum=3.0,
    )
    model_b = RaceProbModel(
        name="B_期待値重視(オッズ考慮)",
        target="is_win",
        use_odds=True,
        expected_sum=1.0,
    )
    return model_a, model_b
