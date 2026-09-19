"""
parquet_loader.py

Reads the one remaining local file this pipeline still loads directly as a
plain parquet file:

    data/output/product_embeddings.parquet   <- built by build_product_embeddings.py
                                               (sql/01, 02, 03 feed that script,
                                               not this one directly — embeddings
                                               don't exist in your warehouse yet)

Basket loading (the household x tpnb x week export) no longer goes through
this module — basket_store.build_baskets_table() reads it directly from its
parquet files via DuckDB's read_parquet(), streaming/aggregating straight
off disk with no separate load step at all. See basket_store.py and
duckdb_manager.py.

No live warehouse connection required — this only reads local files. Update
PRODUCT_EMBEDDINGS_PARQUET below to wherever you've saved the download, or
pass a path directly to load_product_embeddings().
"""

from pathlib import Path

import numpy as np
import pandas as pd
import config

PRODUCT_EMBEDDINGS_PARQUET = Path(config.out("product_embeddings.parquet"))

REQUIRED_PRODUCT_COLS = {"tpnb", "embedding"}


def _parse_embedding_cell(x) -> np.ndarray:
    """
    Handles both shapes the downloaded embedding column can arrive in:
      - already a list/array (parquet preserved a nested/array type)
      - a delimited string, e.g. '0.0123,-0.451,...' (VARCHAR export — the
        more likely case coming out of a web query-tool download)
    """
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=np.float32)
    return np.array([float(v) for v in str(x).split(",")], dtype=np.float32)


def load_product_embeddings(path: Path = PRODUCT_EMBEDDINGS_PARQUET) -> pd.DataFrame:
    """
    Returns a DataFrame shaped exactly like graph_main.py's `product_df_2`:
        tpnb, embedding (np.ndarray)

    Product-level, bounded by catalog size (tens/hundreds of thousands of
    rows) — this has never been the source of this pipeline's memory
    issues, so it's still loaded directly with pd.read_parquet(), no
    streaming/Postgres treatment needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run build_product_embeddings.py first (that needs "
            f"sql/01, 02, and 03 run on your warehouse and downloaded as parquet), which "
            f"produces this file (or pass the real path into load_product_embeddings())."
        )

    df = pd.read_parquet(path)

    missing = REQUIRED_PRODUCT_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. Check that the SQL query's "
            f"column aliases survived the export — rename the columns here if not, "
            f"e.g. df = df.rename(columns={{'exported_name': 'expected_name'}})."
        )

    df["tpnb"] = df["tpnb"].astype(str)
    df["embedding"] = df["embedding"].apply(_parse_embedding_cell)

    print(f"Loaded product_df_2 from {path}: {len(df):,} products")
    return df
