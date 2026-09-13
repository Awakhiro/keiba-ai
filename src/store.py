"""
表の保存と読み込み。

pyarrow があれば parquet、無ければ csv.gz を使う。
GitHub Actions では parquet（速くて小さい）、手元で pyarrow を入れていない場合も
そのまま動くようにするための薄いラッパ。
"""

import os

import pandas as pd

try:
    import pyarrow  # noqa: F401
    EXT = ".parquet"
except ImportError:
    EXT = ".csv.gz"


def path_for(stem: str) -> str:
    """既にあるファイルを優先し、無ければ既定の拡張子を返す。"""
    for ext in (".parquet", ".csv.gz"):
        if os.path.exists(stem + ext):
            return stem + ext
    return stem + EXT


def save_table(df: pd.DataFrame, stem: str) -> str:
    p = path_for(stem)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if p.endswith(".parquet"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False, compression="gzip")
    return p


def load_table(stem: str) -> pd.DataFrame:
    p = path_for(stem)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)

    # CSV は型が失われる。ID列は文字列で読まないと
    # race_id が数値化して突合できず、jockey_id の先頭ゼロ("00666")も消える。
    head = pd.read_csv(p, compression="gzip", nrows=0)
    dtypes = {c: str for c in head.columns if c.endswith("_id") or c == "combo"}
    dtypes.update({c: str for c in head.columns if c.startswith("combo_")})
    return pd.read_csv(p, compression="gzip", dtype=dtypes)


def exists(stem: str) -> bool:
    return os.path.exists(path_for(stem))
