"""
レースごとに「どの券種を何点買うか」を決める。

対象は 馬連・馬単・3連複・3連単 の4つ。点数は3〜5点。

    的中率優先  能力推定の確率だけを見て、当たる確率が最も高くなる組み合わせを選ぶ。
                オッズは選択に使わない（配当の目安としてだけ表示する）。
                ただし合成オッズが低すぎる買い目は意味がないので下限を設ける。

    期待値優先  その券種の実オッズを使い、期待回収率が最も高くなる組み合わせを選ぶ。
                当たらなすぎる買い目は続けられないので、的中率に下限を設ける。
                期待回収率が1.0を割るなら「見送り」を返す。

券種ごとに候補を作って比べるので、堅いレースでは馬連、荒れそうなレースでは
3連複、といった具合にレースの形に応じて推奨が変わる。
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .exotics import build_bets, bet_summary, market_win_probs

BET_TYPES = ("馬連", "馬単", "3連複", "3連単")
# 点数は5点に固定する。買い目は自信のある順（的中確率の高い順）に並べる。
POINT_RANGE = (5,)

# 的中率優先: 合成オッズの下限。当てるだけなら馬連が常に有利だが、
# それでは配当が見合わないので、一定以上の配当がある券種の中から当たる確率で選ぶ。
MIN_COMBINED_ODDS = 2.6
# 的中率優先: さすがに期待回収率がこれを割るものは避ける
MIN_RETURN_HIT_MODE = 0.55
# 期待値優先: 期待回収率がこれを割るなら「基準未満」として扱う
MIN_RETURN_EV_MODE = 1.0
# 券種をまたいで期待値を比べると、推定誤差が最も大きい券種が常に勝つ。
# 最良から一定の幅に収まる案の中では、当たる確率が高い方を選ぶ。
EV_TIE_BAND = 0.90

# 期待値優先の足切り。
#
# 以前は「配当の上限」で高配当を機械的に弾いていたが、それでは妙味を狙う趣旨に
# 反する。上限は撤廃し、代わりに「市場より高く評価している根拠があるか」で絞る。
# 単に配当が大きいだけの組ではなく、モデルが市場と食い違っている組を選ぶ。
MIN_HIT_EV_MODE = 0.05
# 買い目に入れる馬は能力上位のみ。ここは推定誤差が暴れる領域を避けるため残す。
EV_MEMBER_TOP_N = 8
EV_MEMBER_MIN_P = 0.015
# 組全体の乖離。モデルの確率が市場の確率の何倍か。
# 1.0 なら市場と同じ見立て、大きいほど市場が見落としているという判断。
MIN_COMBO_EDGE = 1.15
# 1組あたりの確率の下限。
# 超人気薄では市場確率もモデル確率もほぼゼロになり、その比（乖離）が
# 不安定に跳ね上がる。配当で切る代わりに、確率の絶対値で歯止めをかける。
# 券種ごとの標準的な当たりやすさに合わせて変える。
MIN_COMBO_PROB = {"馬連": 0.010, "馬単": 0.005, "3連複": 0.004, "3連単": 0.0015}


def _combo_edge(probs, market, combo, lam2, lam3, bet_type):
    """その組について、モデルの確率が市場の確率の何倍かを返す。"""
    from .exotics import combo_tables

    key = tuple(combo)
    p = combo_tables(probs, lam2, lam3)[bet_type].get(key)
    q = combo_tables(market, lam2, lam3)[bet_type].get(key)
    if not p or not q:
        return None
    return p / q


def _index_odds(odds_by_no, numbers):
    """馬番キーのオッズ表を行番号キーに変換する。"""
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


def describe_shape(combos, bet_type):
    """買い目の形を言葉にする。軸流しかBOXかフォーメーションか。"""
    if not combos:
        return ""
    sets = [set(c) for c in combos]
    horses = sorted(set().union(*sets))
    size = len(combos[0])

    # 全ての組み合わせに共通して入っている馬があれば軸
    common = set.intersection(*sets)
    if common:
        axis = sorted(common)
        partners = sorted(set(horses) - set(axis))
        if len(axis) == 1:
            return f"{axis[0]}番から{len(partners)}頭流し"
        return f"{'・'.join(str(a) for a in axis)}番の2頭軸"

    # 選んだ馬の総当たりになっていればBOX
    n = len(horses)
    if bet_type in ("馬連", "3連複"):
        full = len(list(combinations(horses, size)))
        if len(combos) == full:
            return f"{n}頭BOX"
    return f"{n}頭のフォーメーション"


def _option(probs, market, bet_type, k, odds_by_no, numbers, strategy,
            lam2, lam3):
    """ある券種・点数の買い目を1案として評価する。"""
    actual = _index_odds(odds_by_no.get(bet_type) if odds_by_no else None, numbers)

    allowed = None
    if strategy == "ev":
        # 能力上位の馬だけを組み合わせの対象にする
        order = np.argsort(probs)[::-1]
        allowed = {int(i) for i in order[:EV_MEMBER_TOP_N]}
        allowed |= {int(i) for i in np.where(probs >= EV_MEMBER_MIN_P)[0]}

    # 絞り込みで足りなくなるのを避けるため、多めに候補を作ってから選ぶ
    n_cand = k * 8 if strategy == "ev" else k
    bets = build_bets(probs, market, bet_type, max_points=n_cand,
                      strategy=strategy, actual_odds=actual,
                      lam2=lam2, lam3=lam3,
                      allowed_members=allowed)

    if strategy == "ev" and not bets.empty:
        # 市場より高く評価している組だけを残す（配当の大小では切らない）
        from .exotics import combo_tables
        pt = combo_tables(probs, lam2, lam3)[bet_type]
        qt = combo_tables(market, lam2, lam3)[bet_type]
        floor = MIN_COMBO_PROB.get(bet_type, 0.002)
        keep = []
        for r in bets.itertuples():
            q = qt.get(tuple(r.combo))
            p = pt.get(tuple(r.combo))
            keep.append(bool(p and q and p >= floor and p / q >= MIN_COMBO_EDGE))
        if sum(keep) >= k:
            bets = bets[keep].reset_index(drop=True)
        else:
            # 条件を満たす組が足りないときは、確率の下限だけ守って埋める
            keep2 = [bool(pt.get(tuple(r.combo), 0) >= floor)
                     for r in bets.itertuples()]
            if any(keep2):
                bets = bets[keep2].reset_index(drop=True)
    if bets.empty or len(bets) < k:
        return None
    if strategy == "ev":
        bets = bets.sort_values("ev", ascending=False).head(k)
    else:
        bets = bets.head(k)
    if len(bets) < k:
        return None
    # 表示は常に「自信のある順」＝的中確率の高い順に並べる。
    bets = bets.sort_values("p", ascending=False).reset_index(drop=True)
    s = bet_summary(bets)
    combos = [[int(numbers[i]) for i in c] for c in bets["combo"]]
    return {
        "券種": bet_type,
        "点数": k,
        "的中確率": s["的中確率"],
        "期待回収率": s["期待回収率"],
        "合成オッズ": s["合成オッズ"],
        "実オッズ": actual is not None,
        "買い目": combos,
        "形": describe_shape(combos, bet_type),
        "明細": [
            {"combo": [int(numbers[i]) for i in r.combo],
             "p": round(float(r.p), 4), "odds": round(float(r.odds), 1),
             "ev": round(float(r.ev), 2)}
            for r in bets.itertuples()
        ],
    }


def recommend(probs, odds_win, bet_types=BET_TYPES, points=POINT_RANGE,
              odds_by_no=None, mode="hit", lam2=0.81, lam3=0.65,
              numbers=None):
    """
    1レース分の推奨を返す。

    probs      レース内の勝率ベクトル（合計1）
    odds_win   単勝オッズ（市場確率の復元に使う）
    odds_by_no {券種: {(馬番,...): 倍率}} 実オッズ。無ければ単勝から推定。
    mode       "hit" 的中率優先 / "ev" 期待値優先
    """
    numbers = np.arange(1, len(probs) + 1) if numbers is None else np.asarray(numbers)
    market = market_win_probs(odds_win)
    strategy = "hit" if mode == "hit" else "ev"

    options = []
    for bt in bet_types:
        for k in points:
            o = _option(probs, market, bt, k, odds_by_no or {}, numbers,
                        strategy, lam2, lam3)
            if o:
                options.append(o)
    if not options:
        return {"見送り": True, "理由": "買い目を作れませんでした", "候補": []}

    if mode == "hit":
        ok = [o for o in options
              if o["合成オッズ"] >= MIN_COMBINED_ODDS
              and o["期待回収率"] >= MIN_RETURN_HIT_MODE]
        reason = "当たる確率を最優先に選びました"
        if not ok:
            ok = [o for o in options if o["合成オッズ"] >= MIN_COMBINED_ODDS]
            reason = "配当が見合う範囲で、当たる確率を優先しました"
        if not ok:
            best = dict(max(options, key=lambda x: x["的中確率"]))
            best["見送り"] = False
            best["参考"] = True
            best["理由"] = "堅すぎて配当が見合いませんが、当てるならこの形です。"
            best["次点"] = []
            return best
        best = max(ok, key=lambda x: x["的中確率"])
    else:
        ok = [o for o in options
              if o["的中確率"] >= MIN_HIT_EV_MODE
              and o["期待回収率"] >= MIN_RETURN_EV_MODE]
        reason = "実オッズと比べて割安な組み合わせを選びました"
        if not ok:
            # 基準に届かなくても、一番ましな案は見せる。
            # 控除率のぶん期待回収率が1.0を割るのは普通のことなので、
            # 「何も無い」ではなく「買うなら これ」を示す。
            fallback = [o for o in options if o["的中確率"] >= MIN_HIT_EV_MODE]
            best = max(fallback or options, key=lambda x: x["期待回収率"])
            best = dict(best)
            best["見送り"] = False
            best["参考"] = True
            best["理由"] = ("期待回収率が100%に届きません。"
                            "控除率を考えると通常のことで、買うなら この形が最良です。")
            others = [o for o in options if o["券種"] != best["券種"]]
            seen, alts = set(), []
            for o in sorted(others, key=lambda x: -x["期待回収率"]):
                if o["券種"] in seen:
                    continue
                seen.add(o["券種"])
                alts.append(o)
                if len(alts) == 2:
                    break
            best["次点"] = alts
            return best
        best_ret = max(o["期待回収率"] for o in ok)
        band = [o for o in ok if o["期待回収率"] >= best_ret * EV_TIE_BAND]
        best = max(band, key=lambda x: x["的中確率"])

    best = dict(best)
    best["見送り"] = False
    best["参考"] = False
    best["理由"] = reason
    # 同じ券種の別点数は省き、他券種の次点を2つ添える
    others = [o for o in options if o["券種"] != best["券種"]]
    key = (lambda x: -x["的中確率"]) if mode == "hit" else (lambda x: -x["期待回収率"])
    seen, alts = set(), []
    for o in sorted(others, key=key):
        if o["券種"] in seen:
            continue
        seen.add(o["券種"])
        alts.append(o)
        if len(alts) == 2:
            break
    best["次点"] = alts
    return best


def recommend_for_race(g: pd.DataFrame, mode="hit", prob_col=None,
                       odds_by_no=None, lam2=0.81, lam3=0.65,
                       bet_types=BET_TYPES, points=POINT_RANGE):
    """
    出走表1レース分のDataFrameから推奨を作る。

    的中率優先は能力のみの確率（p_top3 を正規化したもの）を使い、
    期待値優先は市場と混ぜた確率（p_blend）を使う。
    """
    if prob_col is None:
        prob_col = "p_top3" if mode == "hit" else (
            "p_blend" if "p_blend" in g.columns else "p_win")
    s = np.clip(g[prob_col].to_numpy(dtype=float), 1e-9, None)
    s = s / s.sum()
    return recommend(s, g["odds_prev_win"].to_numpy(dtype=float),
                     bet_types=bet_types, points=points,
                     odds_by_no=odds_by_no, mode=mode,
                     lam2=lam2, lam3=lam3,
                     numbers=g["horse_no"].to_numpy())


def format_recommendation(rec, race_label=""):
    """コンソール向けの1行表示。"""
    if rec.get("見送り"):
        return f"{race_label} 見送り（{rec['理由']}）"
    combos = rec["買い目"]
    sep = " → " if "単" in rec["券種"] else "-"
    body = "  ".join(sep.join(str(x) for x in c) for c in combos)
    return (f"{race_label} {rec['券種']} {rec['点数']}点  {rec['形']}\n"
            f"    {body}\n"
            f"    的中率 {rec['的中確率']*100:.0f}%  "
            f"合成オッズ {rec['合成オッズ']:.1f}倍  "
            f"期待回収率 {rec['期待回収率']*100:.0f}%"
            f"{'' if rec['実オッズ'] else '（配当は推定）'}")
