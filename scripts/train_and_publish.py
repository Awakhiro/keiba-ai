"""
収集済みデータで学習し、検証して、公開用の予想ページを書き出す。

    python scripts/train_and_publish.py --test-start 2026-08-01

出力:
    docs/index.html   スマホ用の予想ページ（GitHub Pages で公開される）
    docs/report.md    検証レポート
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.netkeiba import to_model_schema                       # noqa: E402
from src.store import load_table, exists                       # noqa: E402
from src.features import build_features                        # noqa: E402
from src.backtest import holdout, evaluate, evaluate_exotics_real  # noqa: E402
from src.exotics import fit_lambdas                            # noqa: E402
from src.webexport import build_payload, write_site            # noqa: E402

BETS = ["馬連", "馬単", "3連複", "3連単"]


def summarize_by_scope(ex):
    """S級のみ / A級以上 / 全レース で的中率と回収率を比べる。"""
    import numpy as np
    rows = []
    scopes = {"S級のみ": ["S"], "A級以上": ["S", "A"], "全レース": ["S", "A", "B", "C"]}
    for bt in BETS:
        for m in ("A", "B"):
            sub = ex[(ex["券種"] == bt) & (ex["model"] == m)]
            if sub.empty:
                continue
            for label, gs in scopes.items():
                s = sub[sub["grade"].isin(gs)]
                if s.empty:
                    continue
                w = s["レース数"]
                rows.append({
                    "券種": bt,
                    "モデル": "A 的中率重視" if m == "A" else "B 期待値重視",
                    "対象": label, "レース数": int(w.sum()),
                    "的中率%": round(float(np.average(s["的中率"], weights=w)), 1),
                    "回収率%": round(float(np.average(s["回収率"], weights=w)), 1),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-08-01")
    ap.add_argument("--min-grade", default="B")
    ap.add_argument("--ev", type=float, default=1.05)
    a = ap.parse_args()

    raw = load_table("data/races")
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()

    data = to_model_schema(raw)
    print(f"{len(data):,}行 / {data['race_id'].nunique():,}レース "
          f"({data['date'].min().date()}〜{data['date'].max().date()})", flush=True)

    feat = build_features(data)
    try:
        pred, conf, _ = holdout(feat, a.test_start)
    except ValueError as e:
        print(f"まだ学習できません: {e}")
        print("収集が進めば自動で再実行されます。")
        return
    summary, overall, grader, detail = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)
    ex = evaluate_exotics_real(pred, detail, pays, lam2=lam2, lam3=lam3)
    scope = summarize_by_scope(ex) if not ex.empty else pd.DataFrame()

    os.makedirs("docs", exist_ok=True)
    payload = build_payload(pred, grader=grader, lam2=lam2, lam3=lam3,
                            min_grade=a.min_grade, ev_threshold=a.ev,
                            meta={"date": f"{a.test_start} 以降の検証",
                                  "venues": "中央競馬"})
    write_site(payload, "webapp/template.html", "docs/index.html")

    lines = [
        f"# 検証レポート", "",
        f"更新 {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
        f"- 学習データ {data['date'].min().date()} 〜 {pd.Timestamp(a.test_start).date()} 手前",
        f"- 検証データ {a.test_start} 以降 {pred['race_id'].nunique():,}レース",
        f"- 順位割引 λ2={lam2:.2f} λ3={lam3:.2f}",
        "", "## 単勝・複勝", "", overall.to_frame("値").to_markdown(),
        "", "## 自信度グレード別", "", summary.to_markdown(),
    ]
    if not scope.empty:
        lines += ["", "## 連勝式（実際の払戻で計算）", "",
                  "控除率は馬連・馬単22.5%、3連複25%、3連単27.5%。",
                  "回収率がこれを超えているかが判断の目安。", "",
                  scope.to_markdown(index=False)]
    with open("docs/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + overall.to_string())
    print("\n" + summary.to_string())
    if not scope.empty:
        print("\n" + scope.to_string(index=False))
    print(f"\n書き出し: docs/index.html（{len(payload['races'])}レース） / docs/report.md")


if __name__ == "__main__":
    main()
