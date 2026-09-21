"""
連勝式の確率計算。

単勝の勝率 p_i から「1着a・2着b・3着c」の確率を出すには、
1着が決まった後の残りをどう配分するかのモデルが要る。

素朴な Harville モデル（2着以降も同じ強さ比で配分）は、
実データでは人気馬の連対率を過大評価することが知られている。
ここでは Lo & Bacon-Shone の指数割引モデルを使う：

    P(1着a)     = s_a / Σs
    P(2着b|a)   = s_b^λ2 / Σ_{j≠a} s_j^λ2      λ2 ≈ 0.81
    P(3着c|a,b) = s_c^λ3 / Σ_{j≠a,b} s_j^λ3    λ3 ≈ 0.65

λ<1 は「2着・3着の争いは1着争いより混戦になる」ことを表す。
λ は fit_lambdas() で自前のデータから推定し直せる。

配当の推定:
    馬連などのオッズが手元に無い場合、前日単勝オッズから市場の勝率を復元し、
    同じモデルで市場の組み合わせ確率 q を出して、
        推定配当 = (1 - 控除率) / q
    とする。モデル確率 p との比 p/q × (1-控除率) が期待値になる。
    実際の連勝式オッズがあればそちらを優先する。
"""

from itertools import combinations, permutations

import numpy as np
import pandas as pd

# JRA の控除率（2024年時点）
TAKEOUT = {
    "馬連": 0.225,
    "馬単": 0.225,
    "ワイド": 0.225,
    "3連複": 0.25,
    "3連単": 0.275,
}

LAMBDA2_DEFAULT = 0.81
LAMBDA3_DEFAULT = 0.65


# ------------------------------------------------------------------ 確率行列
def order_probabilities(p, lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT):
    """
    p: 長さ n の勝率ベクトル（合計1に正規化される）
    戻り値: dict
        exacta[a, b]      = P(1着a, 2着b)          … 馬単
        trifecta[a, b, c] = P(1着a, 2着b, 3着c)    … 3連単
    """
    s = np.asarray(p, dtype=float)
    s = np.clip(s, 1e-9, None)
    s = s / s.sum()
    n = len(s)

    w2 = s ** lam2
    w3 = s ** lam3
    W2, W3 = w2.sum(), w3.sum()

    # 馬単: P(a) * w2[b] / (W2 - w2[a])
    denom2 = W2 - w2[:, None]                     # (n,1) a を除いた分母
    exacta = (s[:, None] * w2[None, :]) / np.maximum(denom2, 1e-12)
    np.fill_diagonal(exacta, 0.0)

    # 3連単: exacta[a,b] * w3[c] / (W3 - w3[a] - w3[b])
    denom3 = W3 - w3[:, None, None] - w3[None, :, None]
    trifecta = exacta[:, :, None] * w3[None, None, :] / np.maximum(denom3, 1e-12)
    idx = np.arange(n)
    trifecta[idx, idx, :] = 0.0
    trifecta[idx, :, idx] = 0.0
    trifecta[:, idx, idx] = 0.0

    return {"exacta": exacta, "trifecta": trifecta}


def combo_tables(p, lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT):
    """各券種ごとに {組み合わせ: 確率} の辞書を返す。索引は 0始まりの行番号。"""
    m = order_probabilities(p, lam2, lam3)
    exacta, trifecta = m["exacta"], m["trifecta"]
    n = len(p)

    umatan = {(a, b): float(exacta[a, b]) for a in range(n) for b in range(n) if a != b}
    umaren = {
        (a, b): float(exacta[a, b] + exacta[b, a]) for a, b in combinations(range(n), 2)
    }
    sanrentan = {
        (a, b, c): float(trifecta[a, b, c])
        for a in range(n) for b in range(n) for c in range(n)
        if len({a, b, c}) == 3
    }
    sanrenpuku = {}
    for combo in combinations(range(n), 3):
        sanrenpuku[combo] = float(sum(trifecta[x] for x in permutations(combo)))

    # ワイド: 2頭がともに3着以内
    wide = {}
    for a, b in combinations(range(n), 2):
        tot = 0.0
        for c in range(n):
            if c in (a, b):
                continue
            tot += sanrenpuku.get(tuple(sorted((a, b, c))), 0.0)
        wide[(a, b)] = tot

    return {
        "馬連": umaren,
        "馬単": umatan,
        "ワイド": wide,
        "3連複": sanrenpuku,
        "3連単": sanrentan,
    }


# ------------------------------------------------------------------ 市場側
def market_win_probs(odds_win):
    """前日単勝オッズから市場の勝率を復元（控除率ぶんを均等に剥がす簡易版）。"""
    o = np.clip(np.asarray(odds_win, dtype=float), 1.01, None)
    inv = 1.0 / o
    return inv / inv.sum()


def estimated_payouts(market_probs, bet_type, lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT):
    """市場確率から各組み合わせの推定配当倍率を作る。"""
    tables = combo_tables(market_probs, lam2, lam3)
    q = tables[bet_type]
    t = TAKEOUT[bet_type]
    return {k: (1.0 - t) / max(v, 1e-9) for k, v in q.items()}


# ------------------------------------------------------------------ 買い目構築
def build_bets(model_probs, market_probs, bet_type, actual_odds=None,
               max_points=12, min_ev=None, strategy="hit",
               lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT,
               allowed_members=None, max_odds=None):
    """
    strategy="hit" : 的中確率の高い順に買う（能力重視）
    strategy="ev"  : 期待値の高い順に買う（妙味重視）

    actual_odds:     {組み合わせ: オッズ} が渡されればそれを使う。無ければ市場から推定。
    allowed_members: 買い目に入れてよい馬の行番号の集合。
                     期待値順に並べると、確率がほぼゼロの馬ほどオッズが高いせいで
                     上位に来てしまう。能力の下限で足切りするために使う。
    max_odds:        これを超える配当の組み合わせは買わない。
                     推定誤差が配当倍率で増幅されるのを防ぐ。
    """
    p_tbl = combo_tables(model_probs, lam2, lam3)[bet_type]
    odds_tbl = actual_odds or estimated_payouts(market_probs, bet_type, lam2, lam3)
    allowed = set(allowed_members) if allowed_members is not None else None

    rows = []
    for combo, p in p_tbl.items():
        o = odds_tbl.get(combo)
        if o is None:
            continue
        if allowed is not None and not set(combo) <= allowed:
            continue
        if max_odds is not None and o > max_odds:
            continue
        rows.append({"combo": combo, "p": p, "odds": o, "ev": p * o})
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if strategy == "ev":
        df = df.sort_values("ev", ascending=False)
        if min_ev is not None:
            df = df[df["ev"] >= min_ev]
    else:
        df = df.sort_values("p", ascending=False)

    return df.head(max_points).reset_index(drop=True)


def bet_summary(bets: pd.DataFrame):
    """
    買い目セット全体の点数・的中率・期待回収率・合成オッズ。

    合成オッズ   … 一般的な定義。1 ÷ Σ(1/オッズ)。オッズだけで決まる。
                   「どれが当たっても同じ払戻になるよう賭け金を配分したときの倍率」。
    モデル倍率   … 1 ÷ モデルが見積もった的中確率の合計。
                   「モデルの見立てで何倍なら元が取れるか」を表す内部指標。
                   以前はこちらを合成オッズと呼んでいたが、一般的な定義と違うので分けた。
                   券種を選ぶときの条件（MIN_COMBINED_ODDS）はこちらを使う。
    """
    if bets is None or bets.empty:
        return {"点数": 0, "的中確率": 0.0, "期待回収率": 0.0,
                "合成オッズ": 0.0, "モデル倍率": 0.0}
    n = len(bets)
    hit = float(bets["p"].sum())
    ret = float((bets["p"] * bets["odds"]).sum()) / n     # 1点100円あたり
    inv = float((1.0 / bets["odds"].clip(lower=1.0)).sum())
    return {
        "点数": n,
        "的中確率": hit,
        "期待回収率": ret,
        "合成オッズ": float(1.0 / inv) if inv > 0 else 0.0,
        "モデル倍率": float(1.0 / hit) if hit > 0 else 0.0,
    }


def axis_nagashi(model_probs, market_probs, bet_type, axis_idx, n_partners=5,
                 actual_odds=None, lam2=LAMBDA2_DEFAULT, lam3=LAMBDA3_DEFAULT):
    """軸1頭流し。軸を含む組み合わせだけに絞る。"""
    p_tbl = combo_tables(model_probs, lam2, lam3)[bet_type]
    odds_tbl = actual_odds or estimated_payouts(market_probs, bet_type, lam2, lam3)
    order = np.argsort(model_probs)[::-1]
    partners = [i for i in order if i != axis_idx][:n_partners]

    rows = []
    for combo, p in p_tbl.items():
        if axis_idx not in combo:
            continue
        others = [x for x in combo if x != axis_idx]
        if not all(x in partners for x in others):
            continue
        o = odds_tbl.get(combo)
        if o is None:
            continue
        rows.append({"combo": combo, "p": p, "odds": o, "ev": p * o})
    return pd.DataFrame(rows).sort_values("p", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ λ推定
def fit_lambdas(pred: pd.DataFrame, prob_col="p_win", grid=None):
    """
    実データで λ2, λ3 を推定する（2着・3着の対数尤度を最大化）。
    pred は race_id / finish_pos / prob_col を含む検証データ。
    """
    if grid is None:
        grid = np.arange(0.5, 1.05, 0.05)

    races = []
    for _, g in pred.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        s = np.clip(g[prob_col].to_numpy(dtype=float), 1e-9, None)
        s = s / s.sum()
        fin = g["finish_pos"].to_numpy()
        try:
            i1, i2, i3 = (int(np.where(fin == k)[0][0]) for k in (1, 2, 3))
        except IndexError:
            continue
        races.append((s, i1, i2, i3))

    def ll2(lam):
        tot = 0.0
        for s, i1, i2, _ in races:
            w = s ** lam
            tot += np.log(max(w[i2] / max(w.sum() - w[i1], 1e-12), 1e-12))
        return tot

    def ll3(lam):
        tot = 0.0
        for s, i1, i2, i3 in races:
            w = s ** lam
            tot += np.log(max(w[i3] / max(w.sum() - w[i1] - w[i2], 1e-12), 1e-12))
        return tot

    lam2 = float(max(grid, key=ll2))
    lam3 = float(max(grid, key=ll3))
    return lam2, lam3
