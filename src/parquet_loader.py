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

    # str, deliberately and load-bearingly. The basket side is cast to VARCHAR
    # in basket_store.build_baskets_table() to meet this; before 2026-09-30 it
    # was not, the two key spaces never met, and every product silently got an
    # all-zero embedding vector. See CLAUDE.md section 4b.
    df["tpnb"] = df["tpnb"].astype(str)
    df["embedding"] = df["embedding"].apply(_parse_embedding_cell)

    if len(df):
        widths = df["embedding"].map(len)
        emb_dim = int(widths.iloc[0])
        if not (widths == emb_dim).all():
            # A ragged embedding column is a truncated or malformed export row.
            # Caught here rather than as a numpy broadcast error deep inside
            # prepare_globals' emb_matrix assignment, or a vstack failure inside
            # the sub-clustering, neither of which names the offending product.
            counts = widths.value_counts()
            odd = df.loc[widths != emb_dim, "tpnb"].head(5).tolist()
            raise ValueError(
                f"{path} has embeddings of inconsistent width — every product must "
                f"have the same number of floats.\n"
                f"  widths found (width: n_products): {counts.to_dict()}\n"
                f"  first few products at an unexpected width: {odd}\n"
                f"  If the column was exported as a delimited VARCHAR, a truncated "
                f"row or an embedded delimiter is the usual cause. Re-export, or "
                f"rebuild with build_product_embeddings.py."
            )
        if emb_dim == 0:
            raise ValueError(
                f"{path} has zero-length embeddings for every product — the "
                f"`embedding` column carried no numbers. Check the export format."
            )
        print(f"Loaded product_df_2 from {path}: {len(df):,} products x {emb_dim} dims")
    else:
        print(f"Loaded product_df_2 from {path}: EMPTY")
    return df
