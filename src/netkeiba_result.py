"""
当日のレース結果を取りにいく。

発走済みのレースは結果ページに着順が載る。これを出馬表に突き合わせれば、
予想と結果を並べて見られるようになる。

    https://race.netkeiba.com/race/result.html?race_id=...

過去のレース結果（db.netkeiba.com）とは別のページで、こちらは当日すぐに反映される。
確定前は「速報」として載ることもあるので、着順が揃っているかを確認してから使う。
"""

import io
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .netkeiba import fetch, _norm

RESULT_URL = "https://race.netkeiba.com/race/result.html?race_id={}"

_PAY_KIND = {"馬連": "umaren", "馬単": "umatan", "三連複": "sanrenpuku",
             "3連複": "sanrenpuku", "三連単": "sanrentan", "3連単": "sanrentan",
             "単勝": "win", "複勝": "place", "ワイド": "wide", "枠連": "wakuren"}


def _pick(df, *names):
    cols = {_norm(str(c)): c for c in df.columns}
    for n in names:
        if n in cols:
            return df[cols[n]]
    for n in names:
        for k, c in cols.items():
            if n in k:
                return df[c]
    return pd.Series([None] * len(df))


def parse_result(race_id, html):
    """
    結果ページから (馬番, 着順) の表と払戻を取り出す。
    まだ結果が出ていなければ (None, None)。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile("RaceTable01|ResultTable"))
    if table is None:
        return None, None
    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
    except ValueError:
        return None, None

    out = pd.DataFrame({
        "race_id": str(race_id),
        "horse_no": pd.to_numeric(_pick(df, "馬番"), errors="coerce"),
        "finish_pos": pd.to_numeric(_pick(df, "着順"), errors="coerce"),
    })
    out = out[out["horse_no"].notna()]
    # 1着から3着が揃っていなければ、まだ確定していないとみなす
    if out["finish_pos"].notna().sum() < 3:
        return None, None
    if not {1.0, 2.0, 3.0} <= set(out["finish_pos"].dropna()):
        return None, None

    pays = {}
    for tbl in soup.find_all("table", class_=re.compile("Payout_Detail|pay_table")):
        for tr in tbl.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if not th or len(tds) < 2:
                continue
            key = _PAY_KIND.get(_norm(th.get_text()))
            if not key or key in ("win", "place", "wide", "wakuren"):
                continue
            yen = re.sub(r"[^\d]", "", tds[1].get_text(" ", strip=True).split("円")[0])
            if yen:
                pays[f"payout_{key}"] = float(yen)
    return out, pays


def fetch_result(race_id, sleep=1.0):
    try:
        return parse_result(race_id, fetch(RESULT_URL.format(race_id), sleep=sleep))
    except Exception:
        return None, None


def attach_results(card: pd.DataFrame, race_ids=None, sleep=1.0, verbose=True):
    """
    出馬表に当日の着順を書き込む。
    race_ids を省略すると、発走時刻を過ぎたレースを自動で選ぶ。

    戻り値: (着順を入れた出馬表, 結果が取れたレースID, 払戻)
    """
    d = card.copy()
    d["race_id"] = d["race_id"].astype(str)
    if "finish_pos" not in d.columns:
        d["finish_pos"] = np.nan

    if race_ids is None:
        race_ids = sorted(d["race_id"].unique())

    got, payouts = [], {}
    fin_map = {}
    for rid in race_ids:
        # 既に着順が入っているレースは取りに行かない
        rows = d[d["race_id"] == rid]
        if rows.empty or rows["finish_pos"].notna().any():
            continue
        res, pay = fetch_result(rid, sleep=sleep)
        if res is None:
            continue
        for no, pos in zip(res["horse_no"], res["finish_pos"]):
            fin_map[(rid, int(no))] = pos
        got.append(rid)
        if pay:
            payouts[rid] = pay
        if verbose:
            g = rows.iloc[0]
            print(f"  結果: {g.get('venue','')}{g.get('race_no','')}R", flush=True)

    if fin_map:
        keys = list(zip(d["race_id"], d["horse_no"].astype(int)))
        d["finish_pos"] = [fin_map.get(k, v) for k, v in zip(keys, d["finish_pos"])]
    return d, got, payouts
