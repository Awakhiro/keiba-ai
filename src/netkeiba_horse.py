"""
馬の戦績ページから、レース結果ページでは有料化された情報を補完する。

    https://db.netkeiba.com/horse/result/{horse_id}/

レース結果ページ（db.netkeiba.com/race/...）からは通過順・上がり3Fが
取得できなくなったが、馬の戦績ページには残っている。しかも

    通過   4-4-3-3      各コーナーの位置取り
    ペース 34.9-35.1    前半3F - 後半3F
    上り   34.1         上がり3F

の3つが揃う。ペースが取れるので、前傾ラップ（前半が速い）か後傾ラップかを
判定でき、本来の意味での脚質・ペース分析ができる。

プレミアム限定なのはタイム指数・馬場指数・スタート指数・厩舎コメントなどで、
これらの列は値が `**` になっている。上の3つにはその印が無い。

効率:
    馬1頭につき1リクエストで、その馬の全出走歴ぶんが一度に手に入る。
    レースを個別に回るより通信量がはるかに少ない。

お願い:
    アクセス間隔は最低1秒。netkeiba の利用規約を確認のうえ、個人利用の範囲で。
"""

import io
import re

import pandas as pd
from bs4 import BeautifulSoup

from .netkeiba import fetch, _norm

HORSE_RESULT = "https://db.netkeiba.com/horse/result/{}/"

# プレミアム限定の列に入る伏せ字
_MASKED = re.compile(r"^\**$")
_PACE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_CORNER = re.compile(r"^\s*\d+(?:-\d+)*\s*$")


def _pick(df, *names):
    """列名を完全一致 → 部分一致の順で探す。"""
    cols = {_norm(c): c for c in df.columns}
    for n in names:
        if n in cols:
            return df[cols[n]]
    for n in names:
        for k, c in cols.items():
            if n in k:
                return df[c]
    return pd.Series([None] * len(df))


def parse_horse_result(horse_id, html) -> pd.DataFrame:
    """
    戦績ページを (race_id, horse_id, 通過, ペース, 上り) の表にする。
    race_id はレース名のリンクから取る。中央のレースだけ残す。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile("db_h_race_results|race_table|nk_tb"))
    if table is None:
        return pd.DataFrame()
    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
    except ValueError:
        return pd.DataFrame()

    rows = [tr for tr in table.find_all("tr") if tr.find("td")]
    if len(rows) != len(df):
        return pd.DataFrame()

    race_ids = []
    for tr in rows:
        rid = None
        for a in tr.find_all("a", href=True):
            m = re.search(r"/race/(\d{12})/?", a["href"])
            if m:
                rid = m.group(1)
                break
        race_ids.append(rid)

    out = pd.DataFrame({
        "race_id": race_ids,
        "horse_id": str(horse_id),
        "corner_pos": _pick(df, "通過").astype(str).map(_norm),
        "pace_raw": _pick(df, "ペース").astype(str).map(_norm),
        "last3f": pd.to_numeric(_pick(df, "上り", "上がり"), errors="coerce"),
    })
    out = out[out["race_id"].notna()].copy()

    # 伏せ字や空欄を落とす
    out.loc[out["corner_pos"].str.match(_MASKED) |
            ~out["corner_pos"].str.match(_CORNER), "corner_pos"] = None
    pace = out["pace_raw"].str.extract(_PACE)
    out["pace_first3f"] = pd.to_numeric(pace[0], errors="coerce")
    out["pace_last3f"] = pd.to_numeric(pace[1], errors="coerce")
    out.drop(columns=["pace_raw"], inplace=True)

    # 中央のレースのみ（場コード 01〜10）
    out = out[out["race_id"].str[4:6].isin(
        {f"{i:02d}" for i in range(1, 11)})].reset_index(drop=True)
    return out


def fetch_horse_result(horse_id, sleep=1.0) -> pd.DataFrame:
    html = fetch(HORSE_RESULT.format(horse_id), sleep=sleep)
    return parse_horse_result(horse_id, html)


def merge_into(races: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """
    収集済みのレースデータに、戦績ページから取れた列を突き合わせる。
    既に値がある行は上書きしない。
    """
    if extra.empty:
        return races
    d = races.copy()
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    e = (extra.drop_duplicates(subset=["race_id", "horse_id"])
              .set_index(["race_id", "horse_id"]))

    idx = pd.MultiIndex.from_arrays([d["race_id"], d["horse_id"]])
    for col in ("corner_pos", "last3f", "pace_first3f", "pace_last3f"):
        if col not in e.columns:
            continue
        vals = e[col].reindex(idx).to_numpy()
        if col in d.columns:
            cur = d[col]
            fill = pd.Series(vals, index=d.index)
            # 既存が空のところだけ埋める
            blank = cur.isna() | (cur.astype(str).str.strip().isin(["", "nan", "None"]))
            d.loc[blank, col] = fill[blank]
        else:
            d[col] = vals
    return d
