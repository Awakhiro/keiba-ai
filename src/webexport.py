"""
予想結果をスマホビューア用の JSON にまとめる。

出力は 1 レース 1 オブジェクト。各レースに
  - 出走馬（枠・馬番・両モデルの評価・印）
  - 券種ごとの買い目（モデルA=的中率重視 / モデルB=期待値重視）
  - 自信度グレード
を持たせる。
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .exotics import (
    build_bets, bet_summary, market_win_probs,
    LAMBDA2_DEFAULT, LAMBDA3_DEFAULT,
)
from .confidence import race_confidence, confidence_score, ConfidenceGrader

MARKS = ["◎", "○", "▲", "△", "△", "×"]

# 券種ごとの標準点数（買いすぎを防ぐ上限）
DEFAULT_POINTS = {"馬連": 6, "馬単": 8, "3連複": 10, "3連単": 16}


def _marks(order_idx, n):
    m = [""] * n
    for rank, i in enumerate(order_idx):
        if rank < len(MARKS):
            m[i] = MARKS[rank]
    return m


def _truth(g):
    """着順が揃っていれば的中判定用の正解を返す。"""
    if "finish_pos" not in g.columns or g["finish_pos"].isna().all():
        return None
    fin = g["finish_pos"].to_numpy()
    try:
        i = [int(np.where(fin == k)[0][0]) for k in (1, 2, 3)]
    except IndexError:
        return None
    no = g["horse_no"].to_numpy()
    a, b, c = (int(no[x]) for x in i)
    return {"馬連": sorted([a, b]), "馬単": [a, b],
            "3連複": sorted([a, b, c]), "3連単": [a, b, c]}


def _bets_payload(g, model_probs, market, strategy, bet_types, points,
                  lam2, lam3, min_ev=None):
    out = {}
    truth = _truth(g)
    numbers = g["horse_no"].to_numpy()
    for bt in bet_types:
        bets = build_bets(model_probs, market, bt,
                          max_points=points.get(bt, 8),
                          strategy=strategy, min_ev=min_ev,
                          lam2=lam2, lam3=lam3)
        if bets.empty:
            out[bt] = {"points": [], "summary": bet_summary(bets)}
            continue
        pts = []
        hit_any = False
        for r in bets.itertuples():
            combo = [int(numbers[i]) for i in r.combo]
            key = sorted(combo) if bt in ("馬連", "3連複") else combo
            is_hit = bool(truth and truth[bt] == key)
            hit_any = hit_any or is_hit
            pts.append({"combo": combo, "p": round(float(r.p), 4),
                        "odds": round(float(r.odds), 1),
                        "ev": round(float(r.ev), 2), "hit": is_hit})
        out[bt] = {
            "points": pts,
            "result": (None if truth is None else ("hit" if hit_any else "miss")),
            "summary": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in bet_summary(bets).items()},
        }
    return out


def build_payload(pred: pd.DataFrame, grader: ConfidenceGrader = None,
                  bet_types=("馬連", "馬単", "3連複", "3連単"),
                  points=None, lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT,
                  min_grade="C", ev_threshold=1.05, meta=None):
    points = points or DEFAULT_POINTS
    grader = grader or ConfidenceGrader()

    conf = race_confidence(pred)
    conf["conf_score"] = confidence_score(conf)
    conf["grade"] = grader.transform(conf["conf_score"])
    cmap = conf.set_index("race_id").to_dict("index")

    order = {"S": 3, "A": 2, "B": 1, "C": 0}
    races = []

    for race_id, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        info = cmap.get(race_id, {})
        grade = info.get("grade", "C")
        if order[grade] < order[min_grade]:
            continue

        market = market_win_probs(g["odds_prev_win"].to_numpy())
        sA = np.clip(g["p_top3"].to_numpy(dtype=float), 1e-9, None)
        sA = sA / sA.sum()
        sB = np.clip(g["p_win"].to_numpy(dtype=float), 1e-9, None)
        sB = sB / sB.sum()

        ordA = list(np.argsort(sA)[::-1])
        ordB = list(np.argsort(sB)[::-1])
        markA, markB = _marks(ordA, len(g)), _marks(ordB, len(g))

        horses = []
        for i, r in g.iterrows():
            horses.append({
                "no": int(r["horse_no"]),
                "frame": int(r["frame_no"]),
                "name": str(r.get("horse_name", "") or f"{int(r['horse_id'])}"),
                "jockey": str(r.get("jockey_name", "") or ""),
                "odds": round(float(r["odds_prev_win"]), 1),
                "pA": round(float(r["p_top3"]), 3),
                "pB": round(float(r["p_win"]), 3),
                "ev": round(float(r["p_win"] * r["odds_prev_win"]), 2),
                "markA": markA[i],
                "markB": markB[i],
                "fin": (int(r["finish_pos"]) if pd.notna(r.get("finish_pos")) else None),
            })

        races.append({
            "race_id": str(race_id),
            "venue": str(g["venue"].iloc[0]),
            "race_no": int(g["race_no"].iloc[0]) if "race_no" in g.columns else None,
            "surface": str(g["surface"].iloc[0]),
            "distance": int(g["distance"].iloc[0]),
            "field_size": int(len(g)),
            "post_time": str(g["post_time"].iloc[0]) if "post_time" in g.columns else "",
            "grade": grade,
            "conf": round(float(info.get("conf_score", 0)), 1),
            "agree": bool(info.get("model_agree", 0)),
            "horses": sorted(horses, key=lambda h: h["no"]),
            "bets": {
                "A": _bets_payload(g, sA, market, "hit", bet_types, points, lam2, lam3),
                "B": _bets_payload(g, sB, market, "ev", bet_types, points, lam2, lam3,
                                   min_ev=ev_threshold),
            },
        })

    races.sort(key=lambda r: (-order[r["grade"]], -r["conf"]))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "meta": meta or {},
        "bet_types": list(bet_types),
        "races": races,
    }


def write_site(payload, template_path, out_path):
    """テンプレートHTMLにデータを埋め込み、1ファイルで完結するページを書き出す。"""
    with open(template_path, encoding="utf-8") as f:
        html = f.read()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("/*__DATA__*/null", data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
