"""
parquet_loader.py

Reads two parquet files:

    data/output/product_embeddings.parquet   <- built by build_product_embeddings.py
                                               (sql/01, 02, 03 feed that script,
                                               not this one directly — embeddings
                                               don't exist in your warehouse yet)
    data/ns_household_tpnb_week_agg_train  <- ns_household_tpnb_week_agg_train.sql,
                                               run manually and downloaded as parquet

BASKET GRAIN — WEEK: this table (and the "basket" it feeds) is at
household x tpnb x WEEK grain, not a true single-visit basket — see the
comment at the top of ns_household_tpnb_week_agg_train.sql for why (none of
the tables available in this warehouse carry a transaction/order identifier;
week is the finest grain the data supports).

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
HOUSEHOLD_TPNB_WEEK_PARQUET    = Path("../data/ns_household_tpnb_week_agg_train")

REQUIRED_PRODUCT_COLS   = {"tpnb", "embedding"}
REQUIRED_HOUSEHOLD_COLS = {"household_number", "tpnb", "year_number", "period_number", "week_number", "quantity"}


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


def load_household_tpnb_week(path: Path = HOUSEHOLD_TPNB_WEEK_PARQUET) -> pd.DataFrame:
    """
    Returns a DataFrame shaped exactly like graph_main.py's `tpnb_x_hh`:
        household_number, tpnb, year_number, period_number, week_number, quantity

    Loads the WHOLE table into memory in one shot via pd.read_parquet() — fine
    for a small/sampled export, but at real data scale (week grain routinely
    runs into billions of rows, since it no longer sums multiple weeks
    together the way the old period-grain export did) a single load like
    this can fail outright — pyarrow needs one big contiguous allocation for
    the conversion to pandas, and that can exceed what's available even on a
    machine with plenty of total RAM. Use
    stream_build_baskets_and_units_avg() instead for real data — it never
    materializes this table at all.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run ns_household_tpnb_week_agg_train.sql on "
            f"your workspace, download the result as parquet, and save it here "
            f"(or pass the real path into load_household_tpnb_week())."
        )

    df = pd.read_parquet(path)

    missing = REQUIRED_HOUSEHOLD_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. Check that the SQL query's "
            f"column aliases survived the export — rename the columns here if not."
        )

    df["tpnb"] = df["tpnb"].astype(str)

    print(f"Loaded tpnb_x_hh from {path}: {len(df):,} household x tpnb x week rows")
    return df


def stream_build_baskets_and_units_avg(
    path: Path = HOUSEHOLD_TPNB_WEEK_PARQUET,
    min_basket_products: int = 2,
    batch_size: int = 20_000_000,
):
    """
    Streams the household x tpnb x week export in bounded-size batches and
    builds `baskets` (and per-product average units) directly from the
    stream — the raw flat table is NEVER materialized as one object, only
    one batch (bounded by `batch_size` rows) at a time, plus the growing
    aggregated result (which is much smaller than the raw table, since it's
    already grouped).

    This replaces the old pattern of load_household_tpnb_week() followed by
    a groupby in pipeline_main.py — that pattern needs the ENTIRE raw table
    in memory before any grouping can happen, which is exactly what fails
    at real data scale (see load_household_tpnb_week()'s docstring).

    Correctness note: the same (household_number, week) basket's rows can
    legitimately land in different batches (nothing guarantees the export
    is sorted by household), so partial per-batch aggregations are merged
    into a single running accumulator across the whole stream — no basket
    is ever finalized from just one batch's worth of its rows.

    Returns
    -------
    baskets : DataFrame [household_number, year_week_number, basket_id, products, units]
        Same shape pipeline_main.py's old groupby produced — nothing
        downstream of this needs to change.
    product_units_avg : dict  tpnb -> mean quantity per household x tpnb x week row
    """
    import pyarrow.dataset as pa_dataset
    from collections import defaultdict

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run ns_household_tpnb_week_agg_train.sql on "
            f"your workspace, download the result as parquet, and save it here."
        )

    dataset = pa_dataset.dataset(str(path), format="parquet")
    missing = REQUIRED_HOUSEHOLD_COLS - set(dataset.schema.names)
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. Check that the SQL query's "
            f"column aliases survived the export."
        )
    required_cols = list(REQUIRED_HOUSEHOLD_COLS)

    basket_products = defaultdict(list)   # (household_number, year_week_number) -> [tpnb, ...]
    basket_units    = defaultdict(list)   # same key -> [quantity, ...]
    tpnb_qty_sum    = defaultdict(float)
    tpnb_qty_count  = defaultdict(int)

    n_rows, n_batches = 0, 0
    for record_batch in dataset.to_batches(columns=required_cols, batch_size=batch_size):
        df = record_batch.to_pandas()
        n_batches += 1
        n_rows += len(df)
        df["tpnb"] = df["tpnb"].astype(str)
        df["year_week_number"] = df["year_number"] * 100 + df["week_number"]

        # Streaming mean accumulation for product_units_avg — sum/count per
        # tpnb, divided only once at the very end.
        qty_stats = df.groupby("tpnb")["quantity"].agg(["sum", "count"])
        for tpnb, s, c in zip(qty_stats.index, qty_stats["sum"], qty_stats["count"]):
            tpnb_qty_sum[tpnb]   += s
            tpnb_qty_count[tpnb] += c

        # Per-batch basket aggregation (vectorized pandas groupby — fast),
        # then merged into the growing accumulator dict (bounded by unique
        # baskets touched, not raw row count).
        batch_baskets = (
            df.groupby(["household_number", "year_week_number"])
              .agg(products=("tpnb", list), units=("quantity", list))
        )
        for key, row in zip(batch_baskets.index, batch_baskets.itertuples(index=False)):
            basket_products[key].extend(row.products)
            basket_units[key].extend(row.units)

        del df, qty_stats, batch_baskets
        print(f"  batch {n_batches}: {n_rows:,} rows streamed so far, "
              f"{len(basket_products):,} distinct baskets so far")

    print(f"Streamed {n_rows:,} household x tpnb x week rows across {n_batches} batches "
          f"(from {path})")

    all_keys = list(basket_products.keys())
    before_filter = len(all_keys)
    keys = [k for k in all_keys if len(basket_products[k]) >= min_basket_products]

    baskets = pd.DataFrame({
        "household_number": [k[0] for k in keys],
        "year_week_number": [k[1] for k in keys],
        "products":         [basket_products[k] for k in keys],
        "units":            [basket_units[k] for k in keys],
    })
    baskets["basket_id"] = (
        baskets["household_number"].astype(str) + "_" +
        baskets["year_week_number"].astype(str)
    )
    print(f"  Full baskets: {len(baskets):,} "
          f"({before_filter - len(baskets):,} dropped with < {min_basket_products} products)")

    product_units_avg = {t: tpnb_qty_sum[t] / tpnb_qty_count[t] for t in tpnb_qty_sum}

    return baskets, product_units_avg