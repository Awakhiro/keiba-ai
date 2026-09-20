"""
「的中率だけ」「妙味だけ」「レースごとに使い分け」の3つを比べる。

    python scripts/eval_strategy.py --test-start 2026-01-01

狙い:
    実力が拮抗して人気馬が飛びそうなレースは、的中率の格付けでは低く出る。
    そこを妙味側が拾えていれば、使い分けた方が全体の回収率は上がるはず。
    それが本当かを、実際の払戻で確かめる。

見るべきは回収率と、その信頼区間の下限。
使い分けが単独戦略を上回っていても、区間が重なっていれば差は運の範囲。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import load_table, exists                   # noqa: E402
from src.netkeiba import to_model_schema                   # noqa: E402
from src.features import build_features                    # noqa: E402
from src.backtest import holdout, evaluate                 # noqa: E402
from src.exotics import fit_lambdas                        # noqa: E402
from src.recommend import recommend_for_race               # noqa: E402
from src.confidence import (                               # noqa: E402
    value_metrics, value_score, choose_mode,
)

KEY = {"馬連": "umaren", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}


def truth_of(g):
    fin = g["finish_pos"].to_numpy()
    try:
        i = [int(np.where(fin == k)[0][0]) for k in (1, 2, 3)]
    except IndexError:
        return None
    no = g["horse_no"].to_numpy()
    a, b, c = (int(no[x]) for x in i)
    return {"馬連": sorted([a, b]), "馬単": [a, b],
            "3連複": sorted([a, b, c]), "3連単": [a, b, c]}


def simulate(pred, pays, modes, lam2, lam3, grade_filter=None, detail=None):
    """
    modes: {race_id: "的中率" or "妙味"}
    戻り値: レースごとの賭け金と払戻
    """
    pay_map = pays.set_index("race_id").to_dict("index")
    gmap = detail["grade"].to_dict() if detail is not None else {}
    rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        if grade_filter and gmap.get(race_id) not in grade_filter:
            continue
        g = g.reset_index(drop=True)
        t = truth_of(g)
        if t is None:
            continue
        mode = modes.get(race_id, "的中率")
        rec = recommend_for_race(g, mode=("hit" if mode == "的中率" else "ev"),
                                 lam2=lam2, lam3=lam3)
        if rec.get("見送り"):
            continue
        bt = rec["券種"]
        real = pay_map.get(race_id, {}).get(f"payout_{KEY[bt]}")
        if real is None or not np.isfinite(real):
            continue
        want = t[bt]
        hit = any((sorted(c) if bt in ("馬連", "3連複") else c) == want
                  for c in rec["買い目"])
        rows.append({"race_id": race_id, "mode": mode, "券種": bt,
                     "点数": rec["点数"], "stake": 100.0 * rec["点数"],
                     "payout": float(real) if hit else 0.0, "hit": int(hit)})
    return pd.DataFrame(rows)


def report(df, label, n_boot=4000, seed=0):
    if df.empty:
        print(f"\n{label}: 対象なし")
        return None
    stake, payout = df["stake"].to_numpy(), df["payout"].to_numpy()
    roi = payout.sum() / stake.sum() * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    boots = payout[idx].sum(1) / stake[idx].sum(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mix = df["券種"].value_counts().to_dict()
    modes = df["mode"].value_counts().to_dict()
    print(f"\n── {label} ──")
    print(f"  {len(df)}レース / 的中率 {df['hit'].mean()*100:.1f}% / "
          f"回収率 {roi:.1f}%  95%区間 {lo:.0f}〜{hi:.0f}%")
    print(f"  券種: {mix}")
    if len(modes) > 1:
        print(f"  内訳: {modes}")
    return {"label": label, "roi": roi, "lo": lo, "hi": hi,
            "n": len(df), "hit": df["hit"].mean() * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--grades", default="", help="例 S,A（空なら全レース）")
    ap.add_argument("--margin", type=float, default=12.0,
                    help="妙味スコアが的中率スコアをこれ以上上回ったら妙味を選ぶ")
    a = ap.parse_args()

    data = to_model_schema(load_table("data/races"))
    pays = load_table("data/pays") if exists("data/pays") else pd.DataFrame()
    if pays.empty:
        print("払戻データがありません。"); return

    pred, conf, _ = holdout(build_features(data), a.test_start)
    _s, _o, _g, detail = evaluate(pred, conf)
    lam2, lam3 = fit_lambdas(pred)

    # 妙味の格付けを作る
    vm = value_metrics(pred)
    vm["value_score"] = value_score(vm)
    conf_s = conf.set_index("race_id")["conf_score"]
    val_s = vm.set_index("race_id")["value_score"]
    both = pd.DataFrame({"conf": conf_s, "value": val_s}).dropna()
    both["mode"] = choose_mode(both["conf"], both["value"], margin=a.margin)

    print(f"\n使い分けの内訳: {both['mode'].value_counts().to_dict()}")
    print(f"的中率スコアと妙味スコアの相関: "
          f"{both['conf'].corr(both['value']):.2f}"
          f"（低いほど別のものを測れている）")

    grades = set(a.grades.split(",")) if a.grades else None
    if grades:
        print(f"対象グレード: {sorted(grades)}")

    all_hit = {r: "的中率" for r in both.index}
    all_val = {r: "妙味" for r in both.index}
    mixed = both["mode"].to_dict()

    res = []
    res.append(report(simulate(pred, pays, all_hit, lam2, lam3, grades, detail),
                      "的中率モデルだけ"))
    res.append(report(simulate(pred, pays, all_val, lam2, lam3, grades, detail),
                      "妙味モデルだけ"))
    res.append(report(simulate(pred, pays, mixed, lam2, lam3, grades, detail),
                      "レースごとに使い分け"))

    res = [r for r in res if r]
    if len(res) >= 3:
        base = max(res[0]["roi"], res[1]["roi"])
        diff = res[2]["roi"] - base
        print("\n" + "=" * 52)
        print(f"使い分けは、単独で良い方より {diff:+.1f} ポイント")
        overlap = res[2]["lo"] <= max(res[0]["hi"], res[1]["hi"])
        if diff > 0 and not overlap:
            print("→ 信頼区間が重ならないので、差は本物の可能性があります。")
        elif diff > 0:
            print("→ ただし信頼区間が重なるので、まだ運の範囲です。")
        else:
            print("→ 使い分けても改善していません。単独の方が単純で良いです。")

    # margin を変えるとどうなるか
    print("\n【しきい値を変えた場合】妙味を選ぶ基準の緩さ")
    for m in (0, 5, 10, 15, 20, 30):
        mm = dict(zip(both.index, choose_mode(both["conf"], both["value"], margin=m)))
        df = simulate(pred, pays, mm, lam2, lam3, grades, detail)
        if df.empty:
            continue
        roi = df["payout"].sum() / df["stake"].sum() * 100
        n_val = (df["mode"] == "妙味").sum()
        print(f"  margin={m:>2}: 妙味を選んだのは {n_val:>4}/{len(df):>4}レース  "
              f"的中率 {df['hit'].mean()*100:>4.1f}%  回収率 {roi:>6.1f}%")

    # 絶対値での二重条件。
    # 「的中率の自信が conf_max 以下」かつ「妙味が val_min 以上」のときだけ妙味を使う。
    # 的中率モデルが苦手なレースに限って妙味を差し込む、という考え方。
    print("\n【二重条件】的中率スコアが低く、かつ妙味スコアが高いレースだけ妙味を使う")
    print(f"{'的中≤':>6}{'妙味≥':>6}{'妙味を使う':>10}{'的中率':>8}{'回収率':>8}"
          f"{'区間下限':>9}")
    best = None
    for conf_max in (20, 30, 40, 50, 60):
        for val_min in (50, 60, 70, 80, 90):
            sel = (both["conf"] <= conf_max) & (both["value"] >= val_min)
            if sel.sum() < 30:
                continue
            mm = {r: ("妙味" if v else "的中率")
                  for r, v in zip(both.index, sel)}
            df = simulate(pred, pays, mm, lam2, lam3, grades, detail)
            if df.empty:
                continue
            stake, payout = df["stake"].to_numpy(), df["payout"].to_numpy()
            roi = payout.sum() / stake.sum() * 100
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(df), size=(2000, len(df)))
            lo = float(np.percentile(payout[idx].sum(1) / stake[idx].sum(1) * 100, 2.5))
            n_val = int((df["mode"] == "妙味").sum())
            print(f"{conf_max:>6}{val_min:>6}{n_val:>10}"
                  f"{df['hit'].mean()*100:>7.1f}%{roi:>7.1f}%{lo:>8.0f}%")
            if best is None or roi > best[0]:
                best = (roi, conf_max, val_min, n_val, lo)

    if best:
        roi, cm, vm_, nv, lo = best
        base = res[0]["roi"] if res else 0
        print(f"\n  最良: 的中≤{cm} かつ 妙味≥{vm_} で妙味を使う "
              f"（{nv}レース）→ 回収率 {roi:.1f}%")
        print(f"  的中率モデル単独は {base:.1f}%  差 {roi-base:+.1f} ポイント")
        if roi > base and lo > base:
            print("  → 条件を絞れば改善する可能性があります。別期間で再現するか要確認。")
        elif roi > base:
            print("  → 上回っていますが、信頼区間の下限が単独を下回るのでまだ運の範囲です。")
        else:
            print("  → どの条件でも的中率モデル単独を上回りません。")


if __name__ == "__main__":
    main()
