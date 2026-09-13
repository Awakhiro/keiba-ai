"""デモ実行: 合成データで学習→ウォークフォワード検証→前日予想サンプル出力"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from make_sample_data import generate
from src.features import build_features
from src.backtest import walk_forward, evaluate, ev_filter_report
from src.predict import KeibaPredictor, format_report

pd.set_option("display.width", 160)

print("1) データ生成"); raw = generate(n_races=2500)
print(f"   {len(raw)}行 / {raw['race_id'].nunique()}レース\n")

print("2) 特徴量生成"); feat = build_features(raw)
print(f"   特徴量数: {feat.shape[1]}列\n")

print("3) ウォークフォワード検証")
pred, conf = walk_forward(feat, n_folds=4)
summary, overall, grader, detail = evaluate(pred, conf)
print("\n--- 全体 ---"); print(overall.to_string())
print("\n--- 自信度グレード別 ---"); print(summary.to_string())
print("\n--- 期待値フィルタ別(モデルB単勝) ---"); print(ev_filter_report(pred).to_string(index=False))

print("\n4) 前日予想デモ")
last_day = raw["date"].max()
history = raw[raw["date"] < last_day]
entries = raw[raw["date"] == last_day].drop(columns=["finish_pos","payout_win","payout_place"])
p = KeibaPredictor(); p.train(history); p.grader = grader
out = p.predict_day(history, entries)
reports = p.build_report(out, min_grade="A", top_n=3)
print(format_report(reports))
