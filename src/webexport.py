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
from .confidence import (race_confidence, confidence_score, ConfidenceGrader,
                         model_confidence, model_score, grade_from_score,
                         value_metrics, value_score)
from .recommend import recommend_for_race

MARKS = ["◎", "○", "▲", "△", "△", "×"]

# 券種ごとの点数。3〜5点に絞る方針。
DEFAULT_POINTS = {"馬連": 5, "馬単": 5, "3連複": 5, "3連単": 5}


def _marks(order_idx, n):
    m = [""] * n
    for rank, i in enumerate(order_idx):
        if rank < len(MARKS):
            m[i] = MARKS[rank]
    return m


# モデルごとの買い目の条件。
#   min_odds … 1点あたりの配当の下限（倍）
#   min_ev   … 期待回収率のしきい値。届かなければ「基準未満」として表示
# 穴寄りのモデルほど期待値のしきい値を高くする。
MODEL_RULES = {
    # min_odds … 1点あたりの配当の下限（倍）
    # min_comb … 合成オッズの下限。1 ÷ Σ(1/各組のオッズ) で計算する一般的な定義
    # min_ev   … 期待回収率のしきい値。届かなければ「基準未満」として表示
    "A": {"min_odds": None, "min_comb": 2.0, "min_ev": None},   # 本命
    "M": {"min_odds": 10.0, "min_comb": 3.0, "min_ev": 1.00},   # 中穴
    "L": {"min_odds": 10.0, "min_comb": 5.0, "min_ev": 1.20},   # 穴
}


def _rec_payload(g, mode, odds_by_no, lam2, lam3, rule=None):
    """レースごとの推奨買い目。的中判定も付ける。"""
    try:
        rule = rule or {}
        rec = recommend_for_race(g, mode=mode, odds_by_no=odds_by_no,
                                 lam2=lam2, lam3=lam3,
                                 min_odds=rule.get("min_odds"),
                                 min_ev=rule.get("min_ev"),
                                 min_comb=rule.get("min_comb"))
    except Exception:
        return None
    if rec.get("見送り"):
        return {"skip": True, "reason": rec.get("理由", "")}
    truth = _truth(g)
    key = rec["券種"]
    hit = None
    if truth is not None:
        want = truth[key]
        hit = any((sorted(c) if key in ("馬連", "3連複") else c) == want
                  for c in rec["買い目"])
    # どの組が当たったかを買い目ごとに持たせる
    detail = []
    for d in rec["明細"]:
        c = d["combo"]
        won = bool(truth and (sorted(c) if key in ("馬連", "3連複") else c) == truth[key])
        detail.append({**d, "hit": won})

    return {
        "skip": False,
        "tentative": bool(rec.get("参考")),
        "note": rec.get("理由", "") if rec.get("参考") else "",
        "type": key,
        "points": rec["点数"],
        "shape": rec["形"],
        "p": round(float(rec["的中確率"]), 4),
        "odds": round(float(rec["合成オッズ"]), 1),
        "ret": round(float(rec["期待回収率"]), 2),
        "real_odds": bool(rec["実オッズ"]),
        "combos": rec["買い目"],
        "detail": detail,
        "hit": hit,
        "alts": [{"type": o["券種"], "points": o["点数"],
                  "p": round(float(o["的中確率"]), 4),
                  "ret": round(float(o["期待回収率"]), 2)} for o in rec.get("次点", [])],
    }


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


def _index_odds(odds_by_no, numbers):
    """馬番キーのオッズ表を、行番号キーに変換する。"""
    if not odds_by_no:
        return None
    pos = {int(n): i for i, n in enumerate(numbers)}
    out = {}
    for combo, o in odds_by_no.items():
        combo = combo if isinstance(combo, tuple) else (combo,)
        try:
            out[tuple(pos[int(x)] for x in combo)] = float(o)
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


def _bets_payload(g, model_probs, market, strategy, bet_types, points,
                  lam2, lam3, min_ev=None, real_odds=None):
    out = {}
    truth = _truth(g)
    numbers = g["horse_no"].to_numpy()
    real_odds = real_odds or {}
    for bt in bet_types:
        actual = _index_odds(real_odds.get(bt), numbers)
        bets = build_bets(model_probs, market, bt,
                          max_points=points.get(bt, 8),
                          strategy=strategy, min_ev=min_ev,
                          actual_odds=actual,
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
            "real_odds": actual is not None,
            "result": (None if truth is None else ("hit" if hit_any else "miss")),
            "summary": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in bet_summary(bets).items()},
        }
    return out


def build_payload(pred: pd.DataFrame, grader: ConfidenceGrader = None,
                  bet_types=("馬連", "馬単", "3連複", "3連単"),
                  points=None, lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT,
                  min_grade="C", ev_threshold=1.05, meta=None,
                  odds_tables=None):
    """
    odds_tables: {race_id: {"馬連": {(馬番,馬番): 倍率}, ...}}
        実オッズが渡されればそれで期待値を計算する。無ければ単勝オッズからの推定。
    """
    points = points or DEFAULT_POINTS
    grader = grader or ConfidenceGrader()

    conf = race_confidence(pred)
    conf["conf_score"] = confidence_score(conf)
    conf["grade"] = grader.transform(conf["conf_score"])
    cmap = conf.set_index("race_id").to_dict("index")

    # モデルごとの格付け。それぞれの物差しで「読みやすさ」を測る。
    grades = {}
    for key, col in (("A", "p_top3"), ("H", "p_top3_hi")):
        if col not in pred.columns:
            continue
        cm = model_confidence(pred, col)
        cm["score"] = model_score(cm)
        cm["grade"] = grade_from_score(cm["score"])
        grades[key] = cm.set_index("race_id")[["score", "grade"]].to_dict("index")
    # 妙味は別の物差し（市場との乖離）で測る
    vm = value_metrics(pred)
    vm["score"] = value_score(vm)
    vm["grade"] = grade_from_score(vm["score"])
    grades["B"] = vm.set_index("race_id")[["score", "grade"]].to_dict("index")

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
                "ev": round(float(r.get("ev_win", r["p_win"] * r["odds_prev_win"])), 2),
                "edge": (round(float(r["edge_blend"]), 2) if "edge_blend" in g.columns else None),
                "gap": (int(r["pop_gap"]) if "pop_gap" in g.columns else None),
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
                "A": _bets_payload(g, sA, market, "hit", bet_types, points, lam2, lam3,
                                   real_odds=(odds_tables or {}).get(race_id)),
                "B": _bets_payload(g, sB, market, "ev", bet_types, points, lam2, lam3,
                                   min_ev=ev_threshold,
                                   real_odds=(odds_tables or {}).get(race_id)),
            },
            "grades": {k: {"grade": v.get(race_id, {}).get("grade", "C"),
                           "score": round(float(v.get(race_id, {}).get("score", 0)), 1)}
                       for k, v in grades.items()},
            "recommend": {
                # 本命 … 全レースで学習
                "A": _rec_payload(g, "hit", (odds_tables or {}).get(race_id),
                                  lam2, lam3, MODEL_RULES["A"]),
                # 中穴 … 2着以内に6番人気以下が来たレースで学習
                "M": (_rec_payload(g.assign(p_top3=g["p_top3_mid"],
                                            p_blend=g.get("p_win_mid", g["p_blend"])),
                                   "hit", (odds_tables or {}).get(race_id),
                                   lam2, lam3, MODEL_RULES["M"])
                      if "p_top3_mid" in g.columns else None),
                # 穴 … 馬連が20倍以上だったレースで学習
                "L": (_rec_payload(g.assign(p_top3=g["p_top3_long"],
                                            p_blend=g.get("p_win_long", g["p_blend"])),
                                   "hit", (odds_tables or {}).get(race_id),
                                   lam2, lam3, MODEL_RULES["L"])
                      if "p_top3_long" in g.columns else None),
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


# ------------------------------------------------------------------ 終了レースの固定
def _race_started(race, day, now):
    """発走時刻を過ぎたか。時刻が読めなければ False。"""
    from datetime import datetime, timedelta, timezone
    jst = timezone(timedelta(hours=9))
    try:
        h, m = str(race.get("post_time") or "").split(":")
        d = pd.Timestamp(day).date()
        post = datetime(d.year, d.month, d.day, int(h), int(m), tzinfo=jst)
    except (ValueError, TypeError):
        return False
    return now >= post


def _truth_from_horses(horses):
    by_fin = {h.get("fin"): h["no"] for h in horses if h.get("fin")}
    if not all(k in by_fin for k in (1, 2, 3)):
        return None
    a, b, c = by_fin[1], by_fin[2], by_fin[3]
    return {"馬連": sorted([a, b]), "馬単": [a, b],
            "3連複": sorted([a, b, c]), "3連単": [a, b, c]}


def _rescore(race):
    """固定した買い目に、いまの着順で的中を付け直す。"""
    truth = _truth_from_horses(race.get("horses", []))

    def judge(bt, combo):
        if truth is None or bt not in truth:
            return None
        key = sorted(combo) if bt in ("馬連", "3連複") else list(combo)
        return key == truth[bt]

    for rec in (race.get("recommend") or {}).values():
        if not rec or rec.get("skip"):
            continue
        bt = rec.get("type")
        hits = []
        for d in rec.get("detail", []):
            d["hit"] = judge(bt, d["combo"])
            hits.append(d["hit"])
        rec["hit"] = None if truth is None else any(bool(x) for x in hits)

    for mode in (race.get("bets") or {}).values():
        for bt, st in (mode or {}).items():
            pts = st.get("points") or []
            for p in pts:
                p["hit"] = bool(judge(bt, p["combo"]))
            st["result"] = (None if truth is None
                            else ("hit" if any(p["hit"] for p in pts) else "miss"))


def freeze_finished(payload, data_dir, now=None):
    """
    発走したレースの予想を固定する。

    当日は何度もページを作り直すため、発走後にオッズが最終値へ動いたり
    コードを更新したりすると、終わったレースの買い目まで変わってしまう。
    そこで発走した時点の予想を保存し、以後はそれを表示し続ける。

    固定するのは 買い目・印・確率・格付け。更新するのは 単勝オッズ と 着順 だけで、
    的中はいまの着順で付け直す。

    保存先:
        frozen.json        固定したレース（日付が変わったら破棄）
        last_payload.json  直前に書き出した内容（発走前の最後の予想を拾うため）
    """
    import copy
    import json
    import os
    from datetime import datetime, timedelta, timezone

    jst = timezone(timedelta(hours=9))
    now = now or datetime.now(jst)
    day = str((payload.get("meta") or {}).get("date") or now.date())[:10]
    os.makedirs(data_dir, exist_ok=True)
    fz_p = os.path.join(data_dir, "frozen.json")
    lp_p = os.path.join(data_dir, "last_payload.json")

    def load(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return d if d.get("date") == day else None
        except (OSError, json.JSONDecodeError):
            return None

    frozen = (load(fz_p) or {}).get("races", {})
    last = {r["race_id"]: r for r in ((load(lp_p) or {}).get("races") or [])}

    races, newly, newly_filled = [], [], []
    for r in payload.get("races", []):
        rid = str(r["race_id"])
        started = (any(h.get("fin") for h in r.get("horses", []))
                   or _race_started(r, day, now))
        if started and rid not in frozen:
            # 発走前に最後に出していた予想を固定する。無ければ今の予想で。
            frozen[rid] = copy.deepcopy(last.get(rid) or r)
            newly.append(rid)
        if rid in frozen:
            # 固定時に無かったモデルだけは、いまの予想で補う。
            # 途中の不具合で中穴・穴が作られていなかった場合に、
            # 本命しか表示されない状態が固まってしまうのを防ぐ。
            # 既にある買い目は変えない。
            f_rec = frozen[rid].setdefault("recommend", {})
            for key, cur_rec in (r.get("recommend") or {}).items():
                if cur_rec and not f_rec.get(key):
                    added = copy.deepcopy(cur_rec)
                    added["late"] = True
                    f_rec[key] = added
                    newly_filled.append(f"{rid}:{key}")
            fr = copy.deepcopy(frozen[rid])
            cur = {h["no"]: h for h in r.get("horses", [])}
            for h in fr.get("horses", []):
                c = cur.get(h["no"])
                if c:
                    h["odds"] = c.get("odds", h.get("odds"))
                    h["fin"] = c.get("fin")
            _rescore(fr)
            fr["frozen"] = True
            races.append(fr)
        else:
            races.append(r)
    payload["races"] = races

    with open(fz_p, "w", encoding="utf-8") as f:
        json.dump({"date": day, "races": frozen}, f, ensure_ascii=False)
    with open(lp_p, "w", encoding="utf-8") as f:
        json.dump({"date": day, "races": races}, f, ensure_ascii=False)
    if newly:
        print(f"  発走したレースの予想を固定: {len(newly)}レース", flush=True)
    if newly_filled:
        print(f"  固定済みで欠けていた予想を補完: {len(newly_filled)}件", flush=True)
    return payload
