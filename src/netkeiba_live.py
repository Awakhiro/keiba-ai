"""
これから走るレースの出馬表と前日オッズを取得する。

過去結果（db.netkeiba.com）とは別のページ構造なので分けてある。

    出馬表  https://race.netkeiba.com/race/shutuba.html?race_id=...
    一覧    https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD

注意（2026年9月時点で確認）:
    出馬表ページの**オッズと人気の列は JavaScript で後から埋められる**ため、
    requests で取ると `---.-` と `**` のままになる。
    オッズは別途 API から取りにいく必要があり、ここでは複数の経路を順に試す。

    オッズが取れなくてもモデルA（的中率重視・オッズ非考慮）は完全に動く。
    モデルBと期待値の計算だけができなくなる。
    どうしても取れない場合は odds_csv で手入力を渡せる。
"""

import io
import json
import re
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup

from .netkeiba import (
    VENUE, JRA_CODES, fetch, _norm, _class_level, _SEX_AGE, _COURSE,
)

RACE = "https://race.netkeiba.com"


# ------------------------------------------------------------------ レース一覧
def parse_date(value):
    """'2026-9-19' '2026/9/19' '20260919' などの表記ゆれを受け付ける。"""
    if isinstance(value, date):
        return value
    t = str(value).strip()
    m = re.match(r"^(\d{4})\D+(\d{1,2})\D+(\d{1,2})$", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    digits = re.sub(r"\D", "", t)
    if len(digits) == 8:
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    raise ValueError(f"日付として読めません: {value!r}（例 2026-09-19）")


def race_ids_for(d: date, sleep=1.0):
    """
    その日の中央競馬のレースIDを集める。
    静的な一覧ページを試し、駄目なら空を返す（seed_race_id 経由の取得を使う）。
    """
    ymd = d.strftime("%Y%m%d")
    for url in (f"{RACE}/top/race_list_sub.html?kaisai_date={ymd}",
                f"{RACE}/top/race_list.html?kaisai_date={ymd}",
                f"https://db.netkeiba.com/race/list/{ymd}/"):
        try:
            html = fetch(url, sleep=sleep)
        except Exception:
            continue
        ids = sorted({i for i in re.findall(r"race_id=(\d{12})", html)
                      if i[4:6] in JRA_CODES})
        ids += sorted({i for i in re.findall(r"/race/(\d{12})/?", html)
                       if i[4:6] in JRA_CODES})
        ids = sorted(set(ids))
        if ids:
            return ids
    return []


def race_ids_from_seed(seed_race_id, sleep=1.0):
    """
    レースID を1つ渡すと、その日の全レースを取ってくる。
    出馬表ページには同開催の全Rと他場へのリンクが載っているため、
    一覧ページが取れないときの確実な代替になる。

    netkeiba アプリでその日のどれか1レースを開き、URL の race_id をコピーすればよい。
    """
    seen, found = set(), {str(seed_race_id)}
    while True:
        todo = found - seen
        if not todo:
            break
        for rid in sorted(todo):
            seen.add(rid)
            try:
                html = fetch(f"{RACE}/race/shutuba.html?race_id={rid}", sleep=sleep)
            except Exception:
                continue
            found |= {i for i in re.findall(r"race_id=(\d{12})", html)
                      if i[4:6] in JRA_CODES}
        # 他場の11Rまで辿れば十分なので、2巡で打ち切る
        if len(seen) >= 4:
            break
    return sorted(found)


# ------------------------------------------------------------------ 出馬表
def _row_ids(tr):
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


def parse_shutuba(race_id, html, race_date=None):
    """出馬表HTMLを1レース分のDataFrameにする。着順は当然入らない。"""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    m = _COURSE.search(text)
    if not m:
        return None
    surface, turn_raw, distance = m.group(1), m.group(2), int(m.group(3))
    if surface == "障":
        return None
    turn = "右" if "右" in turn_raw else ("左" if "左" in turn_raw else "直")
    # コース表記の直後にある「(左 B)」の向きも拾う
    around = re.search(r"\d+m\s*\(([右左直])", text)
    if around:
        turn = around.group(1)

    going = None
    g = re.search(r"馬場\s*:\s*(良|稍重|重|不良)", text)
    if g:
        going = g.group(1)
    post = None
    p = re.search(r"(\d{1,2}:\d{2})\s*発走", text)
    if p:
        post = p.group(1)

    name_el = soup.find("div", class_=re.compile("RaceName"))
    race_name = _norm(name_el.get_text()) if name_el else ""

    # 重賞はレース名ではなくアイコンのクラス名で表現される
    # (Icon_GradeType1 = G1, 2 = G2, 3 = G3)
    grade = None
    icon = soup.find(class_=re.compile(r"Icon_GradeType\d"))
    if icon:
        cls = " ".join(icon.get("class", []))
        gm = re.search(r"Icon_GradeType(\d)", cls)
        if gm and gm.group(1) in ("1", "2", "3"):
            grade = {"1": 8, "2": 7, "3": 6}[gm.group(1)]
    cond_el = soup.find("div", class_=re.compile("RaceData02"))
    cond = cond_el.get_text(" ", strip=True) if cond_el else text[:400]

    table = soup.find("table", class_=re.compile("Shutuba_Table|RaceTable01"))
    if table is None:
        return None

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue
        cells = [c.get_text(" ", strip=True) for c in tds]
        nums = [c for c in cells[:3] if re.fullmatch(r"\d+", _norm(c))]
        if len(nums) < 2:
            continue
        frame, horse_no = int(_norm(nums[0])), int(_norm(nums[1]))

        sa = next((c for c in cells if _SEX_AGE.fullmatch(_norm(c))), None)
        kin = next((c for c in cells
                    if re.fullmatch(r"\d{2}(\.\d)?", _norm(c))
                    and 45 <= float(_norm(c)) <= 65), None)

        a_horse = tr.find("a", href=re.compile("/horse/"))
        a_jockey = tr.find("a", href=re.compile("/jockey/"))
        a_trainer = tr.find("a", href=re.compile("/trainer/"))

        # オッズはJSで埋まるので取れないことが多い。取れていれば拾う。
        odds = None
        for c in cells[8:]:
            t = _norm(c)
            if re.fullmatch(r"\d{1,4}\.\d", t):
                v = float(t)
                if 1.0 <= v <= 9999:
                    odds = v
                    break

        rows.append({
            "frame_no": frame, "horse_no": horse_no,
            "horse_name": _norm(a_horse.get_text()) if a_horse else "",
            "sex_age": _norm(sa) if sa else "",
            "weight_carried": float(_norm(kin)) if kin else None,
            "jockey_name": _norm(a_jockey.get_text()) if a_jockey else "",
            "trainer_name": _norm(a_trainer.get_text()) if a_trainer else "",
            "odds_prev_win": odds,
            **_row_ids(tr),
        })

    if len(rows) < 3:
        return None

    df = pd.DataFrame(rows)
    sa = df["sex_age"].str.extract(_SEX_AGE)
    df["sex"] = sa[0].replace({"せん": "セ"}).fillna("牡")
    df["age"] = pd.to_numeric(sa[1], errors="coerce").fillna(4)
    df.drop(columns=["sex_age"], inplace=True)

    df["race_id"] = str(race_id)
    df["date"] = pd.to_datetime(race_date) if race_date else pd.NaT
    df["venue"] = VENUE.get(str(race_id)[4:6], "")
    df["race_no"] = int(str(race_id)[10:12])
    df["race_name"] = race_name
    df["surface"] = surface
    df["distance"] = distance
    df["turn"] = turn
    df["going_forecast"] = going or "良"
    df["post_time"] = post or ""
    df["class_level"] = grade or _class_level(race_name + " " + cond)
    df["field_size"] = len(df)
    df["finish_pos"] = np.nan   # pd.NA だと列が object 型になり学習時と型が食い違う
    return df


def fetch_shutuba(race_id, race_date=None, sleep=1.0):
    html = fetch(f"{RACE}/race/shutuba.html?race_id={race_id}", sleep=sleep)
    return parse_shutuba(race_id, html, race_date)


# ------------------------------------------------------------------ オッズ
# netkeiba のオッズは JS が下のAPIを叩いて描画している。
#   https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=N
# 返りは {"data": {"odds": {"<式別>": {"<組み合わせ>": ["オッズ", ...]}}}}
# 式別キーは 1=単勝 2=複勝 のように並ぶが、番号の割り当ては変わりうるので
# 「キーの形」と「値の個数」から中身を判定する（下の _classify）。

ODDS_API = f"{RACE}/api/api_get_jra_odds.html"


def _get_odds_json(race_id, type_no, sleep=0.5):
    for suffix in ("", "&action=init", "&action=update&locale=ja"):
        try:
            body = fetch(f"{ODDS_API}?race_id={race_id}&type={type_no}{suffix}",
                         sleep=sleep, retries=1)
        except Exception:
            continue
        try:
            js = json.loads(body)
        except json.JSONDecodeError:
            continue
        data = js.get("data") if isinstance(js, dict) else None
        odds = (data or {}).get("odds") if isinstance(data, dict) else None
        if isinstance(odds, dict) and odds:
            return odds
    return None


def _to_combo(key):
    """'0102' や '01-02' を (1, 2) にする。単勝なら (1,)。"""
    k = str(key).strip()
    if "-" in k:
        parts = k.split("-")
    elif len(k) % 2 == 0 and len(k) in (2, 4, 6):
        parts = [k[i:i + 2] for i in range(0, len(k), 2)]
    else:
        parts = [k]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def _first_float(v):
    val = v[0] if isinstance(v, (list, tuple)) and v else v
    try:
        f = float(str(val).split("-")[0])
    except (ValueError, TypeError):
        return None
    return f if f >= 1.0 else None


def _classify(table):
    """
    {組み合わせ: 値} の集合が何の式別かを、キーの形と件数から判定する。
    ワイドは1組に下限と上限の2値が入るので、そこで馬連と区別できる。
    """
    combos = list(table)
    if not combos:
        return None
    size = len(combos[0])
    if any(len(c) != size for c in combos):
        return None

    if size == 1:
        return "単勝"          # 複勝は値が範囲なので呼び出し側で判定
    # 逆順の組み合わせが存在すれば着順を区別する式別（馬単・3連単）
    ordered = any(tuple(reversed(c)) in table
                  for c in combos[:400] if tuple(reversed(c)) != c)
    if size == 2:
        return "馬単" if ordered else "馬連"
    if size == 3:
        return "3連単" if ordered else "3連複"
    return None


def fetch_all_odds(race_id, types=(1, 2, 3, 4, 5, 6, 7, 8), sleep=0.5,
                   want=("単勝", "馬連", "馬単", "3連複", "3連単")):
    """
    取得できたオッズを式別ごとに返す。
        {"単勝": {馬番: 倍率}, "馬連": {(a,b): 倍率}, ...}
    馬番は実際の番号（ゼロ埋めを外した int）。
    """
    out = {}
    for t in types:
        odds = _get_odds_json(race_id, t, sleep=sleep)
        if not odds:
            continue
        for _shikibetsu, table in odds.items():
            if not isinstance(table, dict) or not table:
                continue
            parsed, ranged = {}, False
            for k, v in table.items():
                combo = _to_combo(k)
                f = _first_float(v)
                if combo is None or f is None:
                    continue
                # 単勝は ["33.5", "0", "11"] のように2番目が 0 で返る。
                # 0 は「上限なし」の意味なので、範囲（複勝・ワイド）とは扱わない。
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        hi = float(v[1])
                        if hi > 0 and abs(hi - f) > 1e-9:
                            ranged = True
                    except (ValueError, TypeError):
                        pass
                parsed[combo] = f
            if len(parsed) < 3:
                continue

            kind = _classify(parsed)
            if kind == "単勝":
                # 値が範囲（下限・上限）になっているものが複勝
                kind = "複勝" if ranged else "単勝"
                parsed = {c[0]: v for c, v in parsed.items()}
            elif kind == "馬連" and ranged:
                kind = "ワイド"
            if kind and kind not in out:
                out[kind] = parsed
        if all(k in out for k in want):
            break
    return out


def fetch_odds(race_id, sleep=0.5):
    """単勝オッズだけを {馬番: 倍率} で返す。取れなければ空。"""
    return fetch_all_odds(race_id, types=(1,), sleep=sleep, want=("単勝",)).get("単勝", {})


# ------------------------------------------------------------------ まとめ
def fetch_race_card(d: date, race_ids=None, seed_race_id=None, sleep=1.0,
                    with_odds=True, combo_odds=True, verbose=True):
    """
    指定日の出馬表をまとめて取得する。
    race_ids / seed_race_id のどちらも省略した場合は一覧ページから探す。

    戻り値: (出馬表DataFrame, {race_id: {式別: オッズ表}})
    combo_odds=True なら馬連・馬単・3連複・3連単の実オッズも集める。
    推定配当ではなく実配当で期待値を計算できるようになる。
    """
    d = parse_date(d)
    if race_ids is None:
        race_ids = (race_ids_from_seed(seed_race_id, sleep) if seed_race_id
                    else race_ids_for(d, sleep))
    if not race_ids:
        raise RuntimeError(
            "レースIDが取得できませんでした。netkeiba でその日のレースを1つ開き、\n"
            "URL の race_id をコピーして seed_race_id に渡してください。")

    cards, no_odds, odds_tables = [], [], {}
    for rid in race_ids:
        try:
            df = fetch_shutuba(rid, race_date=d, sleep=sleep)
        except Exception as e:
            if verbose:
                print(f"  {rid} 出馬表失敗: {e}")
            continue
        if df is None:
            continue
        if with_odds:
            types = (1, 2, 3, 4, 5, 6, 7, 8) if combo_odds else (1,)
            want = (("単勝", "馬連", "馬単", "3連複", "3連単") if combo_odds
                    else ("単勝",))
            tables = fetch_all_odds(rid, types=types, want=want)
            if tables.get("単勝"):
                df["odds_prev_win"] = (df["horse_no"].map(tables["単勝"])
                                       .fillna(df["odds_prev_win"]))
            combos = {k: v for k, v in tables.items()
                      if k in ("馬連", "馬単", "3連複", "3連単", "ワイド")}
            if combos:
                odds_tables[str(rid)] = combos
        if df["odds_prev_win"].isna().all():
            no_odds.append(rid)
        cards.append(df)
        if verbose:
            got = [k for k in ("単勝", "馬連", "馬単", "3連複", "3連単")
                   if (k == "単勝" and df["odds_prev_win"].notna().any())
                   or k in odds_tables.get(str(rid), {})]
            print(f"  {df['venue'].iloc[0]}{df['race_no'].iloc[0]}R "
                  f"{len(df)}頭  オッズ: {'/'.join(got) if got else 'なし'}")

    if not cards:
        raise RuntimeError("出馬表を1件も取得できませんでした。")
    out = pd.concat(cards, ignore_index=True)
    if verbose:
        print(f"\n{out['race_id'].nunique()}レース / {len(out)}頭")
        print(f"単勝オッズ取得: {out['odds_prev_win'].notna().mean()*100:.0f}%  "
              f"連勝式の実オッズ: {len(odds_tables)}レース")
        if no_odds:
            print(f"オッズが取れなかったレース {len(no_odds)}件: "
                  "モデルA（的中率重視）はオッズを使わないのでそのまま動きます。")
    return out, odds_tables


def apply_manual_odds(entries: pd.DataFrame, odds_text: str) -> pd.DataFrame:
    """
    オッズを手入力で補う。1行1レースで次の形式:

        202605040811: 1=2.4, 2=15.3, 3=7.8, ...

    馬番=オッズ をカンマ区切りで並べる。netkeiba アプリのオッズ画面を見ながら
    主要レースだけ入れる、という使い方でよい。
    """
    d = entries.copy()
    for line in odds_text.strip().splitlines():
        if ":" not in line:
            continue
        rid, body = line.split(":", 1)
        rid = rid.strip()
        pairs = {}
        for part in body.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=")
            try:
                pairs[int(k.strip())] = float(v.strip())
            except ValueError:
                continue
        if not pairs:
            continue
        mask = d["race_id"].astype(str) == rid
        # 指定した馬だけ上書きし、書かなかった馬の値は残す
        d.loc[mask, "odds_prev_win"] = (
            d.loc[mask, "horse_no"].map(pairs).fillna(d.loc[mask, "odds_prev_win"]))
    return d
