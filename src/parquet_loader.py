"""
parquet_loader.py

Reads two parquet files:

    data/output/product_embeddings.parquet   <- built by build_product_embeddings.py
                                               (sql/01, 02, 03 feed that script,
                                               not this one directly — embeddings
                                               don't exist in your warehouse yet)
    data/household_tpnb_period_agg.parquet <- sql/04_household_tpnb_period_agg.sql,
                                               run manually and downloaded as parquet

No live warehouse connection required — this only reads local files. Update
the two path constants below to wherever you've saved the downloads, or pass
paths directly to the two load_* functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Update these two paths to your downloaded files
# ─────────────────────────────────────────────

PRODUCT_EMBEDDINGS_PARQUET = Path("../data/output/product_embeddings.parquet")
HOUSEHOLD_TPNB_PERIOD_PARQUET   = Path("../data/ns_household_tpnb_period_agg_train")

REQUIRED_PRODUCT_COLS   = {"tpnb", "embedding"}
REQUIRED_HOUSEHOLD_COLS = {"household_number", "tpnb", "year_number", "period_number", "quantity"}


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


def load_household_tpnb_period(path: Path = HOUSEHOLD_TPNB_PERIOD_PARQUET) -> pd.DataFrame:
    """
    Returns a DataFrame shaped exactly like graph_main.py's `tpnb_x_hh`:
        household_number, tpnb, year_number, period_number, quantity
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run sql/04_household_tpnb_period_agg.sql on the "
            f"your workspace, download the result as parquet, and save it here "
            f"(or pass the real path into load_household_tpnb_period())."
        )

    df = pd.read_parquet(path)

    missing = REQUIRED_HOUSEHOLD_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. Check that the SQL query's "
            f"column aliases survived the export — rename the columns here if not."
        )

    df["tpnb"] = df["tpnb"].astype(str)

    print(f"Loaded tpnb_x_hh from {path}: {len(df):,} household x tpnb x period rows")
    return df