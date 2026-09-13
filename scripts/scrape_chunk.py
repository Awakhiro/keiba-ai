"""
時間予算を決めて収集し、期限が来たら途中で綺麗に終わるスクリプト。

GitHub Actions のジョブには実行時間の上限があるので、
「決めた分だけ集めて保存 → 次回の実行で続きから」という形にする。

    python scripts/scrape_chunk.py --start 2025-01-01 --end 2026-09-13 --minutes 300

状態:
    data/state.json      収集済みの開催日
    data/races.parquet   レース結果
    data/pays.parquet    払戻

終了コード 0 = 全期間の収集が完了 / 75 = まだ残りがある（次回に続く）
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.netkeiba import racing_dates, race_ids_on, fetch, parse_race, BASE  # noqa: E402
from src.store import save_table, load_table, exists  # noqa: E402

DATA = "data"
STATE = f"{DATA}/state.json"
RACES = f"{DATA}/races"
PAYS = f"{DATA}/pays"

EXIT_INCOMPLETE = 75


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        return set(s.get("done_dates", [])), s
    return set(), {}


def save(done, races_new, pays_new, meta):
    os.makedirs(DATA, exist_ok=True)
    if races_new:
        df = pd.concat(races_new, ignore_index=True)
        if exists(RACES):
            df = pd.concat([load_table(RACES), df], ignore_index=True)
        df = df.drop_duplicates(subset=["race_id", "horse_no"], keep="last")
        save_table(df, RACES)
    if pays_new:
        df = pd.DataFrame(pays_new)
        if exists(PAYS):
            df = pd.concat([load_table(PAYS), df], ignore_index=True)
        df = df.drop_duplicates(subset=["race_id"], keep="last")
        save_table(df, PAYS)
    with open(STATE, "w") as f:
        json.dump({"done_dates": sorted(done), **meta}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()

    start = date.fromisoformat(a.start)
    end = min(date.fromisoformat(a.end), date.today())
    deadline = time.time() + a.minutes * 60

    done, _ = load_state()
    todo = [d for d in racing_dates(start, end) if d.isoformat() not in done]
    print(f"対象 {start}〜{end} / 未収集 {len(todo)}日 / 予算 {a.minutes:.0f}分", flush=True)
    if not todo:
        print("すべて収集済み")
        save(done, [], [], {"updated_at": datetime.now().isoformat(timespec="seconds")})
        return 0

    races_new, pays_new, n = [], [], 0
    t0 = time.time()
    for d in todo:
        if time.time() > deadline:
            print("\n時間予算に達したので中断します。次回の実行で続きから。", flush=True)
            break
        try:
            ids = race_ids_on(d, sleep=a.sleep)
        except Exception as e:
            print(f"  {d} 一覧取得失敗: {e}", flush=True)
            continue

        for rid in ids:
            if time.time() > deadline:
                break
            try:
                df, pay = parse_race(rid, fetch(f"{BASE}/race/{rid}/", sleep=a.sleep))
            except Exception as e:
                print(f"  {rid} 失敗: {e}", flush=True)
                continue
            if df is not None:
                races_new.append(df)
                if pay:
                    pays_new.append(pay)
                n += 1
        else:
            done.add(d.isoformat())   # その日を最後まで回れた場合だけ完了扱い

        if len(races_new) >= 300:
            save(done, races_new, pays_new, {})
            races_new, pays_new = [], []
        el = (time.time() - t0) / 60
        print(f"  {d} 済み  累計{n:,}レース  経過{el:.0f}分", flush=True)

    save(done, races_new, pays_new,
         {"updated_at": datetime.now().isoformat(timespec="seconds"),
          "start": a.start, "end": a.end})

    remaining = [d for d in racing_dates(start, end) if d.isoformat() not in done]
    total = len(load_table(RACES)) if exists(RACES) else 0
    print(f"\n保存: 全{total:,}行 / 残り{len(remaining)}日", flush=True)
    return 0 if not remaining else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
