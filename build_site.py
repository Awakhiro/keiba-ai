"""
学習 → 予想 → スマホ用HTML書き出し まで一気に実行する。

実データを使う場合:
    python build_site.py --history history.csv --entries tomorrow.csv
サンプルで動作確認:
    python build_site.py --demo
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import pandas as pd

from src.features import build_features
from src.backtest import walk_forward, evaluate
from src.exotics import fit_lambdas
from src.predict import KeibaPredictor
from src.webexport import build_payload, write_site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history"); ap.add_argument("--entries")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="予想.html")
    ap.add_argument("--min-grade", default="C")
    ap.add_argument("--ev", type=float, default=1.05)
    a = ap.parse_args()

    if a.demo:
        from make_sample_data import generate
        raw = generate(n_races=2400)
        # 直近36レースを「明日の開催」に見立てる（3場 × 12R）
        day_ids = sorted(raw["race_id"].unique())[-36:]
        history = raw[~raw["race_id"].isin(day_ids)].copy()
        entries = raw[raw["race_id"].isin(day_ids)].drop(
            columns=["finish_pos", "payout_win", "payout_place"]).copy()
        entries["date"] = raw["date"].max() + pd.Timedelta(days=7)
        venues = ["中山", "中京", "阪神"]
        pos = {r: i for i, r in enumerate(day_ids)}
        entries["race_no"] = entries["race_id"].map(lambda r: pos[r] % 12 + 1)
        entries["venue"] = entries["race_id"].map(lambda r: venues[pos[r] // 12])
        entries["post_time"] = entries["race_no"].map(
            lambda n: f"{9 + (n * 35 + 20) // 60}:{(n * 35 + 20) % 60:02d}")
    else:
        history = pd.read_csv(a.history)
        entries = pd.read_csv(a.entries)

    print("学習中…")
    feat = build_features(history)
    pred_bt, conf = walk_forward(feat, n_folds=3, verbose=False)
    _, _, grader, _ = evaluate(pred_bt, conf)
    lam2, lam3 = fit_lambdas(pred_bt)
    print(f"  自信度しきい値 {[round(t,1) for t in grader.thresholds]} / λ2={lam2:.2f} λ3={lam3:.2f}")
    print(grader.stats.to_string())

    p = KeibaPredictor().train(history)
    p.grader = grader
    pred = p.predict_day(history, entries)

    payload = build_payload(
        pred, grader=grader, lam2=lam2, lam3=lam3,
        min_grade=a.min_grade, ev_threshold=a.ev,
        meta={"date": str(pd.to_datetime(entries["date"]).max().date()),
              "venues": "・".join(sorted(entries["venue"].unique()))},
    )
    out = write_site(payload, "webapp/template.html", a.out)
    print(f"書き出し: {out}  （{len(payload['races'])}レース）")


if __name__ == "__main__":
    main()
