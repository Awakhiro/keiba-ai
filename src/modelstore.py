"""
学習済みモデルを保存して使い回す。

毎回学習し直す必要はない。学習データが増えていなければ結果も同じなので、
データの中身を指紋にして、変わっていなければ保存済みのものを読む。

効果は時間だけではない。同じ日のうちにモデルが変わらなくなるので、
朝と昼で予想が食い違うことがなくなる。オッズが動いたから変わったのか、
モデルが変わったからなのか、という区別がつくようになる。

    from src.modelstore import load_or_train
    predictor, trained = load_or_train(history)
"""

import hashlib
import json
import os
import pickle

import pandas as pd

STORE_DIR = "data/models"
META = "meta.json"


def _config_tag():
    """
    モデルの作り方（中穴・穴の学習条件）。これが変わったら保存済みは使えない。
    データが同じでも、条件を変えたのに古いモデルを読んでしまうのを防ぐ。
    """
    try:
        from .predict import KeibaPredictor as K
        return f"pop{K.UPSET_POPULARITY}_odds{K.LONGSHOT_ODDS:g}"
    except Exception:
        return "unknown"


def data_fingerprint(history: pd.DataFrame) -> str:
    """
    学習データとモデル条件の指紋。行数・レース数・期間・最終レースID、
    それに中穴・穴の学習条件から作る。これが同じなら学習しても同じモデルになる。
    """
    try:
        parts = [
            str(len(history)),
            str(history["race_id"].nunique()),
            str(pd.to_datetime(history["date"]).min().date()),
            str(pd.to_datetime(history["date"]).max().date()),
            str(sorted(history["race_id"].astype(str))[-1]),
        ]
    except Exception:
        parts = [str(len(history))]
    parts.append(_config_tag())
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _paths(store_dir=STORE_DIR):
    return os.path.join(store_dir, META), os.path.join(store_dir, "models.pkl")


# 保存形式の版。モデルの構成を変えたら上げる。
# 2: 本命・中穴・穴の3段構成。穴モデルの学習に払戻データを渡すようにした。
MODEL_VERSION = 2


def load_saved(history, store_dir=STORE_DIR, version=MODEL_VERSION):
    """
    保存済みモデルがあり、学習データが変わっていなければ読み込む。
    無ければ None。
    """
    meta_p, pkl_p = _paths(store_dir)
    if not (os.path.exists(meta_p) and os.path.exists(pkl_p)):
        return None
    try:
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if meta.get("version") != version:
        return None
    fp = data_fingerprint(history)
    if meta.get("fingerprint") != fp:
        return None

    try:
        with open(pkl_p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save(predictor, history, store_dir=STORE_DIR, version=MODEL_VERSION, extra=None):
    meta_p, pkl_p = _paths(store_dir)
    os.makedirs(store_dir, exist_ok=True)
    with open(pkl_p, "wb") as f:
        pickle.dump(predictor, f)
    meta = {
        "version": version,
        "fingerprint": data_fingerprint(history),
        "rows": int(len(history)),
        "races": int(history["race_id"].nunique()),
        "until": str(pd.to_datetime(history["date"]).max().date()),
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        **(extra or {}),
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def load_or_train(history, store_dir=STORE_DIR, force=False, verbose=True):
    """
    保存済みがあれば読み、無ければ学習して保存する。
    戻り値: (予想器, 学習したかどうか)
    """
    from .predict import KeibaPredictor

    if not force:
        saved = load_saved(history, store_dir)
        if saved is not None:
            if verbose:
                meta_p, _ = _paths(store_dir)
                with open(meta_p, encoding="utf-8") as f:
                    m = json.load(f)
                print(f"学習済みモデルを再利用します"
                      f"（{m['races']:,}レースで学習 / {m['saved_at']}）")
            return saved, False

    if verbose:
        print(f"学習データが変わったので学習します"
              f"（{history['race_id'].nunique():,}レース）", flush=True)
    # 穴モデルは「馬連が高配当だったレース」を払戻データから選ぶので、必ず渡す
    payouts = None
    try:
        from .store import load_table, exists
        if exists("data/pays"):
            payouts = load_table("data/pays")
    except Exception:
        payouts = None
    p = KeibaPredictor().train(history, payouts=payouts)
    save(p, history, store_dir)
    return p, True


def describe(store_dir=STORE_DIR):
    """保存済みモデルの情報。無ければ None。"""
    meta_p, _ = _paths(store_dir)
    if not os.path.exists(meta_p):
        return None
    with open(meta_p, encoding="utf-8") as f:
        return json.load(f)
