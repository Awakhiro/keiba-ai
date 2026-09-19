"""
netkeiba（db.netkeiba.com）から中央競馬のレース結果を収集する。

ページ構造（2026年9月時点で確認）:
    一覧  https://db.netkeiba.com/race/list/YYYYMMDD/
    結果  https://db.netkeiba.com/race/{race_id}/        文字コード EUC-JP
    race_id = 年(4) + 場コード(2) + 回(2) + 日(2) + R(2)
    場コード 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉

取得できるもの:
    着順・枠番・馬番・馬名・性齢・斤量・騎手・タイム・着差・通過順・上り3F・
    単勝オッズ・人気・馬体重・調教師・賞金、および全券種の払戻

取得できないもの:
    前日オッズ。過去レースのページに残るのは確定オッズ（発走時点）だけ。
    検証では確定オッズを前日オッズの代わりに使う。実際の前日オッズより情報量が多いぶん
    バックテストはやや楽観的に出る。運用時は前日オッズを使うので、そこは差し引いて見る。

お願い:
    アクセス間隔は最低1秒。netkeiba の利用規約を確認のうえ、個人利用の範囲で。
"""

import io
import re
import time
from datetime import date, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
JRA_CODES = set(VENUE)

BASE = "https://db.netkeiba.com"
_session = requests.Session()
_session.headers.update({"User-Agent": UA})


# ------------------------------------------------------------------ 取得
# netkeiba はページによって文字コードが違う。
#   db.netkeiba.com（過去のレース結果・馬の戦績） → EUC-JP
#   race.netkeiba.com（出馬表・オッズ）           → UTF-8
# 決め打ちすると片方が文字化けし、コース表記などが読めなくなる。
# 判定に使う語は必ず日本語にする。ASCII の語（"netkeiba" など）を混ぜると
# どの文字コードで復号しても一致してしまい、文字化けしたまま採用されてしまう。
_ENCODING_MARKERS = ("レース", "馬番", "発走", "競走成績", "着順", "騎手")


def _decode(resp):
    """日本語が最も多く読めた文字コードを採用する。"""
    best, best_score = None, 0
    for enc in ("UTF-8", "EUC-JP", "CP932"):
        try:
            text = resp.content.decode(enc, errors="replace")
        except (LookupError, AttributeError):
            continue
        score = sum(w in text for w in _ENCODING_MARKERS)
        if score > best_score:
            best, best_score = text, score
    if best is not None:
        return best
    return resp.content.decode(resp.apparent_encoding or "UTF-8", errors="replace")


def fetch(url, sleep=1.0, retries=3):
    last = None
    for i in range(retries):
        try:
            r = _session.get(url, timeout=25)
            if r.status_code == 200:
                text = _decode(r)
                time.sleep(sleep)
                return text
            last = f"status {r.status_code}"
        except Exception as e:  # 通信エラーは待って再試行
            last = str(e)
        time.sleep(sleep * (2 ** i) + 1)
    raise RuntimeError(f"取得失敗 {url}: {last}")


def racing_dates(start: date, end: date):
    """JRAは土日と祝日月曜の開催。候補日だけ見に行く。"""
    d, out = start, []
    while d <= end:
        if d.weekday() in (5, 6, 0):   # 土日月
            out.append(d)
        d += timedelta(days=1)
    return out


def race_ids_on(d: date, sleep=1.0):
    html = fetch(f"{BASE}/race/list/{d.strftime('%Y%m%d')}/", sleep=sleep)
    ids = set(re.findall(r"/race/(\d{12})/?", html))
    return sorted(i for i in ids if i[4:6] in JRA_CODES)


# ------------------------------------------------------------------ 解析
_SEX_AGE = re.compile(r"([牡牝セせん]+)\s*(\d+)")
_WEIGHT = re.compile(r"(\d+)\(([-+]?\d+)\)")
_COURSE = re.compile(r"([芝ダ障])\s*([右左直外内]*)\s*(\d+)\s*m")


def _norm(s):
    return re.sub(r"\s+", "", str(s))


def _class_level(text):
    """条件をおおまかな階級（数値）にする。数字が大きいほど上のクラス。"""
    t = _norm(text)
    for pat, lv in [
        (r"新馬|未勝利", 1),
        (r"1勝クラス|500万下", 2),
        (r"2勝クラス|1000万下", 3),
        (r"3勝クラス|1600万下", 4),
    ]:
        if re.search(pat, t):
            return lv
    if re.search(r"G3|GIII|ＧⅢ", t): return 6
    if re.search(r"G2|GII|ＧⅡ", t): return 7
    if re.search(r"G1|GI|ＧⅠ", t): return 8
    if re.search(r"オープン|OP|リステッド|L\)", t): return 5
    return 3


def _parse_meta(soup, html):
    """コース・馬場・日付・条件を取り出す。"""
    # レース情報は data_intro ブロックにまとまっている。無ければページ全体から拾う。
    intro = soup.find("div", class_=re.compile("data_intro"))
    text = (intro or soup).get_text(" ", strip=True)
    if not _COURSE.search(text):
        text = soup.get_text(" ", strip=True)

    m = _COURSE.search(text)
    surface, turn, distance = (m.group(1), m.group(2), int(m.group(3))) if m else (None, "", None)
    if surface == "ダ":
        surface = "ダ"
    turn = "右" if "右" in turn else ("左" if "左" in turn else "直")

    going = None
    g = re.search(r"[芝ダート]+\s*:\s*(良|稍重|重|不良)", text)
    if g:
        going = g.group(1)
    weather = None
    w = re.search(r"天候\s*:\s*(\S+?)\s", text)
    if w:
        weather = w.group(1)
    post = None
    p = re.search(r"発走\s*:\s*(\d{1,2}:\d{2})", text)
    if p:
        post = p.group(1)

    dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    race_date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else None

    # 先頭の h1 はサイトロゴなので、data_intro 内の h1 を優先する
    name_el = None
    if intro:
        name_el = intro.find("h1")
    if name_el is None:
        for h in soup.find_all("h1"):
            if "netkeiba" not in h.get_text():
                name_el = h
                break
    race_name = _norm(name_el.get_text()) if name_el else ""

    cond = ""
    small = (intro or soup).find("p", class_="smalltxt")
    if small:
        cond = small.get_text(" ", strip=True)

    return {
        "surface": surface, "turn": turn, "distance": distance,
        "going_actual": going, "weather": weather, "post_time": post,
        "date": race_date, "race_name": race_name,
        "class_level": _class_level(race_name + " " + cond),
        "is_jump": surface == "障",
    }


def _row_ids(tr):
    """1行から 馬ID / 騎手ID / 調教師ID を取り出す。"""
    out = {"horse_id": None, "jockey_id": None, "trainer_id": None}
    for a in tr.find_all("a", href=True):
        h = a["href"]
        if out["horse_id"] is None and "/horse/" in h:
            m = re.search(r"/horse/(\w+)", h)
            if m: out["horse_id"] = m.group(1)
        elif out["jockey_id"] is None and "/jockey/" in h:
            m = re.search(r"/jockey/[a-z/]*?(\w+)/?$", h.rstrip("/"))
            if m: out["jockey_id"] = m.group(1)
        elif out["trainer_id"] is None and "/trainer/" in h:
            m = re.search(r"/trainer/[a-z/]*?(\w+)/?$", h.rstrip("/"))
            if m: out["trainer_id"] = m.group(1)
    return out


def parse_payouts(soup):
    """払戻テーブルから各券種の配当と当たり組み合わせを取り出す。"""
    out = {}
    place = {}
    for tbl in soup.find_all("table", class_=re.compile("pay_table")):
        for tr in tbl.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if not th or len(tds) < 2:
                continue
            kind = _norm(th.get_text())
            combos = [c for c in tds[0].get_text("\n").split("\n") if c.strip()]
            yens = [c for c in tds[1].get_text("\n").split("\n") if c.strip()]
            yens = [float(y.replace(",", "")) for y in yens if re.fullmatch(r"[\d,]+", y.strip())]
            if not combos or not yens:
                continue
            if kind == "複勝":
                for c, y in zip(combos, yens):
                    place[int(_norm(c))] = y
            elif kind == "単勝":
                out["payout_win_yen"] = yens[0]
                out["win_no"] = int(_norm(combos[0]))
            else:
                key = {"馬連": "umaren", "馬単": "umatan",
                       "三連複": "sanrenpuku", "三連単": "sanrentan",
                       "ワイド": "wide", "枠連": "wakuren"}.get(kind)
                if key and key != "wide":
                    out[f"payout_{key}"] = yens[0]
                    out[f"combo_{key}"] = _norm(combos[0])
    out["payout_place_map"] = place
    return out


def parse_race(race_id, html):
    """1レース分のHTMLを (出走馬DataFrame, 払戻dict) にする。障害・中止は None。"""
    soup = BeautifulSoup(html, "html.parser")
    meta = _parse_meta(soup, html)
    if meta["is_jump"] or meta["distance"] is None or meta["date"] is None:
        return None, None

    table = soup.find("table", class_=re.compile("race_table|nk_tb"))
    if table is None:
        return None, None
    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
    except ValueError:
        return None, None
    df.columns = [_norm(c) for c in df.columns]

    rows = [tr for tr in table.find_all("tr") if tr.find("td")]
    if len(rows) != len(df):
        return None, None
    ids = pd.DataFrame([_row_ids(tr) for tr in rows])

    def col(*names):
        # 完全一致 → 部分一致の順で探す。表記ゆれで取り逃がさないように。
        for n in names:
            if n in df.columns:
                return df[n]
        for n in names:
            for c in df.columns:
                if n in c:
                    return df[c]
        return pd.Series([None] * len(df))

    out = pd.DataFrame({
        "race_id": race_id,
        "date": meta["date"],
        "venue": VENUE[race_id[4:6]],
        "race_no": int(race_id[10:12]),
        "race_name": meta["race_name"],
        "surface": meta["surface"],
        "distance": meta["distance"],
        "turn": meta["turn"],
        "going_actual": meta["going_actual"],
        "weather": meta["weather"],
        "post_time": meta["post_time"],
        "class_level": meta["class_level"],
        "finish_raw": col("着順").astype(str),
        "frame_no": pd.to_numeric(col("枠番"), errors="coerce"),
        "horse_no": pd.to_numeric(col("馬番"), errors="coerce"),
        "horse_name": col("馬名").astype(str).map(_norm),
        "sex_age": col("性齢").astype(str),
        "weight_carried": pd.to_numeric(col("斤量"), errors="coerce"),
        "jockey_name": col("騎手").astype(str).map(_norm),
        "finish_time": col("タイム", "走破").astype(str),
        "corner_pos": col("通過", "通過順", "コーナー").astype(str),
        "last3f": pd.to_numeric(col("上り", "上がり", "後3F"), errors="coerce"),
        "odds_win_final": pd.to_numeric(col("単勝"), errors="coerce"),
        "popularity_final": pd.to_numeric(col("人気", "人 気"), errors="coerce"),
        "weight_raw": col("馬体重", "体重").astype(str),
        "trainer_name": col("調教師").astype(str).map(_norm),
        "prize": pd.to_numeric(col("賞金(万円)", "賞金"), errors="coerce").fillna(0.0),
    })
    out = pd.concat([out, ids], axis=1)

    sa = out["sex_age"].str.extract(_SEX_AGE)
    out["sex"] = sa[0].replace({"せん": "セ"})
    out["age"] = pd.to_numeric(sa[1], errors="coerce")
    wt = out["weight_raw"].str.extract(_WEIGHT)
    out["horse_weight"] = pd.to_numeric(wt[0], errors="coerce")
    out["horse_weight_diff"] = pd.to_numeric(wt[1], errors="coerce")

    # 着順: 中止・除外・失格は数値にならないので落とす
    out["finish_pos"] = pd.to_numeric(out["finish_raw"], errors="coerce")
    out = out[out["horse_no"].notna()].copy()
    if out["finish_pos"].notna().sum() < 5:
        return None, None

    pay = parse_payouts(soup)
    pmap = pay.get("payout_place_map", {})
    out["payout_win"] = out.apply(
        lambda r: pay.get("payout_win_yen", 0.0) if r["horse_no"] == pay.get("win_no") else 0.0,
        axis=1)
    out["payout_place"] = out["horse_no"].map(lambda n: pmap.get(int(n), 0.0))

    race_pay = {"race_id": race_id, **{k: v for k, v in pay.items()
                                       if k.startswith(("payout_", "combo_"))
                                       and k != "payout_place_map"}}
    return out.drop(columns=["finish_raw", "sex_age", "weight_raw"]), race_pay


# ------------------------------------------------------------------ 収集
def collect(start: date, end: date, sleep=1.0, checkpoint=None, save_every=40,
            progress=None):
    """
    期間内の全JRAレースを収集する。checkpoint を指定すると途中結果を保存し、
    再実行時は続きから再開する。
    """
    import os
    import pickle

    state = {"races": {}, "pays": {}, "done_dates": set()}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)
        print(f"再開: 済み {len(state['races'])}レース / {len(state['done_dates'])}日")

    def save():
        if checkpoint:
            with open(checkpoint, "wb") as f:
                pickle.dump(state, f)

    dates = [d for d in racing_dates(start, end) if d.isoformat() not in state["done_dates"]]
    n_new = 0
    for di, d in enumerate(dates):
        try:
            ids = race_ids_on(d, sleep=sleep)
        except Exception as e:
            print(f"  {d} 一覧取得失敗: {e}")
            continue
        for rid in ids:
            if rid in state["races"]:
                continue
            try:
                html = fetch(f"{BASE}/race/{rid}/", sleep=sleep)
                df, pay = parse_race(rid, html)
            except Exception as e:
                print(f"  {rid} 失敗: {e}")
                continue
            if df is not None:
                state["races"][rid] = df
                if pay:
                    state["pays"][rid] = pay
                n_new += 1
                if n_new % save_every == 0:
                    save()
        state["done_dates"].add(d.isoformat())
        save()
        if progress:
            progress(di + 1, len(dates), len(state["races"]))

    races = (pd.concat(state["races"].values(), ignore_index=True)
             if state["races"] else pd.DataFrame())
    pays = (pd.DataFrame(state["pays"].values())
            if state["pays"] else pd.DataFrame())
    return races, pays


# ------------------------------------------------------------------ 整形
def to_model_schema(df: pd.DataFrame) -> pd.DataFrame:
    """収集データをモデルが期待するスキーマに合わせる。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["finish_pos"].notna()].copy()

    # 前日オッズが取れないので確定オッズで代用する（README参照）
    d["odds_prev_win"] = d["odds_win_final"].fillna(d["odds_win_final"].median())
    d["going_forecast"] = d["going_actual"].fillna("良")
    d["field_size"] = d.groupby("race_id")["horse_no"].transform("size")

    for c, fill in [("frame_no", 4), ("age", 4), ("weight_carried", 55.0)]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(fill)
    d["sex"] = d["sex"].fillna("牡")
    d["jockey_id"] = d["jockey_id"].fillna("unknown")
    d["trainer_id"] = d["trainer_id"].fillna("unknown")
    d["horse_id"] = d["horse_id"].fillna(d["horse_name"])

    keep = [
        "race_id", "date", "venue", "race_no", "race_name", "surface", "distance",
        "turn", "class_level", "field_size", "going_forecast", "going_actual",
        "post_time", "horse_id", "horse_name", "horse_no", "frame_no", "age", "sex",
        "weight_carried", "jockey_id", "jockey_name", "trainer_id", "trainer_name",
        "odds_prev_win", "odds_win_final", "popularity_final", "finish_pos",
        "finish_time", "last3f", "corner_pos", "pace_first3f", "pace_last3f", "horse_weight", "horse_weight_diff", "prize",
        "payout_win", "payout_place",
    ]
    return d[[c for c in keep if c in d.columns]].reset_index(drop=True)
