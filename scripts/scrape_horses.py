"""
馬の戦績ページを巡って、通過順・ペース・上がり3F を集める。

    python scripts/scrape_horses.py --minutes 300

収集済みのレースデータに出てくる馬を、出走回数の多い順に処理する。
馬1頭につき1リクエストで全出走歴ぶんが手に入るので、
レースを個別に回すより桁違いに効率がよい。

状態:
    data/horse_extra.parquet   取得できた (race_id, horse_id, 通過, ペース, 上り)
    data/horse_state.json      処理済みの馬ID

終了コード 0 = 全頭完了 / 75 = まだ残りがある（次回に続く）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.store import save_table, load_table, exists      # noqa: E402
from src.netkeiba_horse import fetch_horse_result         # noqa: E402

DATA = "data"
STATE = f"{DATA}/horse_state.json"
EXTRA = f"{DATA}/horse_extra"
EXIT_INCOMPLETE = 75


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        return set(s.get("done", [])), s
    return set(), {}


def save(done, new_rows, meta):
    os.makedirs(DATA, exist_ok=True)
    if new_rows:
        df = pd.concat(new_rows, ignore_index=True)
        if exists(EXTRA):
            df = pd.concat([load_table(EXTRA), df], ignore_index=True)
        df = df.drop_duplicates(subset=["race_id", "horse_id"], keep="last")
        save_table(df, EXTRA)
    with open(STATE, "w") as f:
        json.dump({"done": sorted(done), **meta}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--min-starts", type=int, default=1,
                    help="この回数以上出走している馬だけを対象にする")
    a = ap.parse_args()

    races = load_table("data/races")
    counts = (races["horse_id"].astype(str).value_counts())
    counts = counts[counts >= a.min_starts]

    done, _ = load_state()
    todo = [h for h in counts.index if h not in done and h not in ("nan", "None", "")]
    print(f"対象 {len(counts):,}頭 / 未処理 {len(todo):,}頭 / 予算 {a.minutes:.0f}分",
          flush=True)
    if not todo:
        print("すべて取得済み")
        save(done, [], {"updated_at": datetime.now().isoformat(timespec="seconds")})
        return 0

    deadline = time.time() + a.minutes * 60
    new_rows, n, t0 = [], 0, time.time()
    for hid in todo:
        if time.time() > deadline:
            print("\n時間予算に達したので中断します。次回の実行で続きから。", flush=True)
            break
        try:
            df = fetch_horse_result(hid, sleep=a.sleep)
        except Exception as e:
            print(f"  {hid} 失敗: {e}", flush=True)
            continue
        done.add(hid)
        if not df.empty:
            new_rows.append(df)
        n += 1
        if n % 100 == 0:
            save(done, new_rows, {})
            new_rows = []
            el = (time.time() - t0) / 60
            rate = n / max(el, 1e-6)
            left = (len(todo) - n) / max(rate, 1e-6)
            print(f"  {n:,}/{len(todo):,}頭  経過{el:.0f}分  残り約{left:.0f}分", flush=True)

    save(done, new_rows, {"updated_at": datetime.now().isoformat(timespec="seconds")})
    remaining = [h for h in counts.index if h not in done]
    total = len(load_table(EXTRA)) if exists(EXTRA) else 0
    print(f"\n保存: {total:,}行 / 残り {len(remaining):,}頭", flush=True)
    return 0 if not remaining else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
