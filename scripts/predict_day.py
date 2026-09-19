"""
これから走るレースを予想する。

    python scripts/predict_day.py --date 2026-09-20
    python scripts/predict_day.py --date 2026-09-20 --seed-race-id 202609040811

過去データ（data/races.*）で学習し、指定日の出馬表に対して予想を出す。
出力は docs/next.html。
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table                       # noqa: E402
from src.netkeiba import to_model_schema               # noqa: E402
from src.netkeiba_live import fetch_race_card, apply_manual_odds  # noqa: E402
from src.predict import KeibaPredictor                 # noqa: E402
from src.features import build_features                # noqa: E402
from src.backtest import walk_forward, evaluate        # noqa: E402
from src.exotics import fit_lambdas                    # noqa: E402
from src.webexport import build_payload, write_site    # noqa: E402
from src.static_report import write as write_static     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="対象日 (既定: 次の土曜)")
    ap.add_argument("--seed-race-id", default=None,
                    help="一覧が取れない場合、その日のレースIDを1つ渡す")
    ap.add_argument("--odds-file", default=None, help="手入力オッズのテキスト")
    ap.add_argument("--min-grade", default="B")
    ap.add_argument("--out", default="docs/next.html")
    a = ap.parse_args()

    if a.date:
        target = date.fromisoformat(a.date)
    else:
        today = date.today()
        target = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    print(f"対象日: {target}")

    history = to_model_schema(load_table("data/races"))
    print(f"学習データ {len(history):,}行 / {history['race_id'].nunique():,}レース")

    print("\n出馬表を取得します")
    entries, odds_tables = fetch_race_card(target, seed_race_id=a.seed_race_id)
    if a.odds_file and os.path.exists(a.odds_file):
        entries = apply_manual_odds(entries, open(a.odds_file, encoding="utf-8").read())

    has_odds = entries["odds_prev_win"].notna()

    # オッズが無い馬は、モデルBの入力として中立な値で埋める。
    # モデルAはオッズを使わないので影響を受けない。
    if not has_odds.all():
        entries["odds_prev_win"] = entries["odds_prev_win"].fillna(
            entries.groupby("race_id")["horse_no"].transform("size").astype(float))

    print("\n学習中")
    feat = build_features(history)
    _pred, _conf = walk_forward(feat, n_folds=3, verbose=False)
    _s, _o, grader, _d = evaluate(_pred, _conf)
    lam2, lam3 = fit_lambdas(_pred)

    p = KeibaPredictor().train(history)
    p.grader = grader
    out = p.predict_day(history, entries)

    payload = build_payload(out, grader=grader, lam2=lam2, lam3=lam3,
                            min_grade=a.min_grade, odds_tables=odds_tables,
                            meta={"date": target.isoformat(),
                                  "venues": "・".join(sorted(entries["venue"].unique()))})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_site(payload, "webapp/template.html", a.out)
    static_path = a.out.replace(".html", "_静的.html")
    write_static(payload, static_path, title=f"予想 {target}")
    print(f"静的版（JavaScript不要）: {static_path}")
    print(f"\n書き出し: {a.out}（{len(payload['races'])}レース）")
    if not has_odds.all():
        print("オッズが欠けたレースがあります。「当てにいく」（モデルA）で見てください。")


if __name__ == "__main__":
    main()
