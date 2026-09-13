"""
basket_store.py

The shared basket-storage layer — used by BOTH pipeline_main.py (training)
and score_new_baskets.py (scoring), via a `dataset_tag` argument
("train" / "score") that keeps their tables separate in the same local
DuckDB database (see duckdb_manager.py). This is the fix for the recurring
memory bug in this pipeline: `baskets` was always held as one Python object
for the whole script, and at real week-grain scale that's bigger than
available RAM. Basket storage now lives in DuckDB; nothing in this file
ever holds the full basket population in Python memory — every function
here either streams bounded chunks, or returns a result whose size is
bounded by something OTHER than basket count (catalog size, chunk size, or
the training-sample size).

DuckDB detail worth knowing: it can query parquet files DIRECTLY
(`read_parquet(...)`), streaming/aggregating straight off disk — so unlike
the earlier Postgres-based design, there's no separate "load raw rows into
a staging table" step at all. build_baskets_table() reads the raw export
straight from its parquet files and produces the basket table in one pass.

Grain reminder: a "basket" here is everything one household bought in one
WEEK (household_number x year_week_number) — see
data/ns_household_tpnb_week_agg_train.sql for why this, not a true
single-visit basket, is the finest grain this warehouse supports.

Every function here takes `con` — the shared DuckDB connection object from
duckdb_manager.get_connection() — not a URI to reconnect with (DuckDB has
no server/URI model; the connection IS the open database file handle).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_RAW_COLS = {"household_number", "tpnb", "year_number", "period_number", "week_number", "quantity"}
BASKET_COLS = ["household_number", "year_week_number", "basket_id", "products", "units"]

# Lowercase-only: kept simple and consistent, even though DuckDB (unlike
# Postgres's to_regclass()) doesn't have the same case-folding gotcha —
# still avoids any ambiguity about identifier quoting.
_TAG_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _validate_tag(dataset_tag: str):
    if not _TAG_RE.match(dataset_tag):
        raise ValueError(
            f"dataset_tag must be a plain lowercase identifier matching "
            f"{_TAG_RE.pattern!r} (it's used to build table names) — got {dataset_tag!r}"
        )


def _qi(name: str) -> str:
    """Quote a (validated) identifier for safe interpolation into SQL text."""
    return f'"{name}"'


def _seed_to_duckdb_seed(seed: int) -> float:
    """DuckDB's setseed(), like Postgres's, wants a float in [-1, 1]."""
    return (seed % 2000 - 1000) / 1000.0


def _glob_for(path) -> str:
    """A parquet path can be a single file or a folder of part-files (the
    Databricks/Spark export shape) — read_parquet() needs a glob for the
    latter, a plain path for the former."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return str(path / "*.parquet") if path.is_dir() else str(path)


def _fetch_basket_range(con, baskets_table: str, lo: int, hi: int) -> pd.DataFrame:
    return con.execute(
        f"SELECT household_number, year_week_number, basket_id, products, units "
        f"FROM {_qi(baskets_table)} WHERE basket_seq BETWEEN ? AND ? ORDER BY basket_seq",
        [lo, hi],
    ).df()


# ─────────────────────────────────────────────
# AGGREGATE — build the per-basket table directly from the raw parquet export
# ─────────────────────────────────────────────

def build_baskets_table(con, parquet_path, dataset_tag: str, min_basket_products: int = 2):
    """
    Reads the household x tpnb x week export straight from its parquet
    files via read_parquet() and builds the per-basket table in one SQL
    pass — DuckDB streams/aggregates directly off disk, so the raw flat
    table is never materialized as a Python object OR as an intermediate
    DuckDB table. This replaces the old Python dict-accumulator (and the
    Postgres design's separate staging-table load step) entirely.

    Returns (n_baskets_total, product_units_avg, product_uniques) — all
    three are bounded by catalog size or a single COUNT(*), never by
    basket count.
    """
    _validate_tag(dataset_tag)
    glob_path = _glob_for(parquet_path)

    # Sanity-check the export has the columns we need before spending time
    # scanning it — read_parquet's schema is available via a 0-row query.
    schema_cols = set(con.execute(f"SELECT * FROM read_parquet(?) LIMIT 0", [glob_path]).df().columns)
    missing = REQUIRED_RAW_COLS - schema_cols
    if missing:
        raise ValueError(f"{parquet_path} is missing columns {missing}.")

    baskets_table = f"baskets_{dataset_tag}"
    con.execute(f"DROP TABLE IF EXISTS {_qi(baskets_table)}")
    con.execute(
        f"""
        CREATE TABLE {_qi(baskets_table)} AS
        SELECT ROW_NUMBER() OVER (ORDER BY household_number, year_week_number) AS basket_seq,
               household_number, year_week_number,
               (CAST(household_number AS VARCHAR) || '_' || CAST(year_week_number AS VARCHAR)) AS basket_id,
               list(tpnb) AS products,
               list(quantity) AS units,
               CASE
                   WHEN len(list(tpnb)) <= 1  THEN 0
                   WHEN len(list(tpnb)) <= 5  THEN 1
                   WHEN len(list(tpnb)) <= 15 THEN 2
                   WHEN len(list(tpnb)) <= 50 THEN 3
                   ELSE 4
               END AS size_bucket
        FROM (
            SELECT household_number,
                   year_number * 100 + week_number AS year_week_number,
                   tpnb, quantity
            FROM read_parquet(?)
            WHERE household_number IS NOT NULL AND tpnb IS NOT NULL
        ) t
        GROUP BY household_number, year_week_number
        HAVING len(list(tpnb)) >= ?
        """,
        [glob_path, min_basket_products],
    )

    con.execute(f'CREATE UNIQUE INDEX "idx_{dataset_tag}_seq" ON {_qi(baskets_table)} (basket_seq)')
    con.execute(f'CREATE INDEX "idx_{dataset_tag}_bucket" ON {_qi(baskets_table)} (size_bucket)')
    con.execute(f'CREATE INDEX "idx_{dataset_tag}_bid" ON {_qi(baskets_table)} (basket_id)')

    n_baskets_total = con.execute(f"SELECT COUNT(*) FROM {_qi(baskets_table)}").fetchone()[0]

    units_avg_rows = con.execute(
        "SELECT tpnb, AVG(quantity) FROM read_parquet(?) GROUP BY tpnb", [glob_path]
    ).fetchall()
    product_units_avg = {r[0]: float(r[1]) for r in units_avg_rows}

    uniq_rows = con.execute("SELECT DISTINCT tpnb FROM read_parquet(?) ORDER BY tpnb", [glob_path]).fetchall()
    product_uniques = np.array([r[0] for r in uniq_rows])

    print(f"  {baskets_table}: {n_baskets_total:,} baskets, "
          f"{len(product_uniques):,} distinct products (read directly from parquet)")
    return n_baskets_total, product_units_avg, product_uniques


# ─────────────────────────────────────────────
# STREAM — bounded-size chunks for the co-purchase matrix loop
# ─────────────────────────────────────────────

def stream_basket_chunks(con, dataset_tag: str, chunk_size: int, start_seq: int = 1):
    """
    Generator yielding one DataFrame (<= chunk_size rows) at a time, in
    basket_seq order. Used by pipeline_main.py's co-purchase chunk loop,
    which already has its own restart mechanism (the fingerprinted
    copurchase_sparse checkpoint) — this only needs to be a plain
    sequential stream.
    """
    _validate_tag(dataset_tag)
    baskets_table = f"baskets_{dataset_tag}"
    max_seq = con.execute(f"SELECT MAX(basket_seq) FROM {_qi(baskets_table)}").fetchone()[0] or 0

    lo = start_seq
    while lo <= max_seq:
        hi = min(lo + chunk_size - 1, max_seq)
        df = _fetch_basket_range(con, baskets_table, lo, hi)
        if len(df) > 0:
            yield df
        lo = hi + 1


# ─────────────────────────────────────────────
# SAMPLE — stratified-by-size training sample, computed in SQL
# ─────────────────────────────────────────────

def sample_training_baskets(con, dataset_tag: str, n_samples: int, seed: int) -> pd.DataFrame:
    """
    Same stratified-by-basket-size logic GraphBuilder.sample_baskets used to
    do in-memory (pd.cut over the whole `baskets` DataFrame) — computed here
    via the size_bucket column build_baskets_table() already persisted, with
    an index on it, so each stratum's random draw
    (`WHERE size_bucket = ? ORDER BY random() LIMIT ?`) only sorts that
    stratum's rows, not the whole population. Returns ONE DataFrame
    (n_samples rows) — the only basket-shaped object that stays fully
    resident in Python memory anywhere in this design.
    """
    _validate_tag(dataset_tag)
    baskets_table = f"baskets_{dataset_tag}"

    con.execute("SELECT setseed(?)", [_seed_to_duckdb_seed(seed)])
    stratum_counts = dict(
        con.execute(f"SELECT size_bucket, COUNT(*) FROM {_qi(baskets_table)} GROUP BY size_bucket").fetchall()
    )
    total = sum(stratum_counts.values())
    if total == 0:
        print("  WARNING: baskets table is empty — nothing to sample.")
        return pd.DataFrame(columns=BASKET_COLS)

    parts = []
    for bucket, count in stratum_counts.items():
        if count == 0:
            continue
        quota = min(count, max(1, round(n_samples * count / total)))
        df = con.execute(
            f"SELECT household_number, year_week_number, basket_id, products, units "
            f"FROM {_qi(baskets_table)} WHERE size_bucket = ? ORDER BY random() LIMIT ?",
            [bucket, quota],
        ).df()
        parts.append(df)

    sampled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BASKET_COLS)
    if len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(f"  Sampled {len(sampled):,} baskets across {len(stratum_counts)} "
          f"basket-size strata (via DuckDB, indexed per-stratum random sort)")
    return sampled


# ─────────────────────────────────────────────
# RESTARTABLE INFERENCE CHUNK QUEUE
#
# A DuckDB table, not just an in-memory generator — this is what makes a
# long inference run resumable after a crash: already-'complete' chunks are
# never redone.
#
# IMPORTANT LIMITATION vs. the earlier Postgres design: DuckDB has no
# row-level locking (no FOR UPDATE SKIP LOCKED). claim_next_chunk() below
# is safe for this pipeline's actual usage — one process claiming chunks
# sequentially — but is NOT safe if ever called concurrently from multiple
# processes/threads at once (two callers could both see and claim the same
# "pending" chunk in a race). If true multi-machine parallel inference is
# ever built, it would need a different coordination mechanism than DuckDB
# can provide on its own (e.g. a small separate lock file per chunk, or a
# lightweight external coordinator) — this is an honest trade-off of
# switching away from Postgres, not something worked around here.
# ─────────────────────────────────────────────

def ensure_inference_chunk_plan(con, dataset_tag: str, chunk_size: int):
    """
    Creates and populates the chunk-plan table ONCE per dataset_tag. If it
    already exists (a resumed run after a crash), it's left untouched so
    already-complete chunks are never redone.
    """
    _validate_tag(dataset_tag)
    baskets_table = f"baskets_{dataset_tag}"
    chunks_table = f"inference_chunks_{dataset_tag}"

    already_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [chunks_table]
    ).fetchone()[0] > 0
    if already_exists:
        print(f"  {chunks_table} already exists — reusing as-is "
              f"(resuming any pending/incomplete chunks rather than restarting)")
        return

    max_seq = con.execute(f"SELECT MAX(basket_seq) FROM {_qi(baskets_table)}").fetchone()[0] or 0

    con.execute(
        f"""
        CREATE TABLE {_qi(chunks_table)} (
            chunk_id   INTEGER PRIMARY KEY,
            seq_lo     BIGINT NOT NULL,
            seq_hi     BIGINT NOT NULL,
            status     VARCHAR NOT NULL DEFAULT 'pending',
            worker_id  VARCHAR,
            updated_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )

    rows, chunk_id, lo = [], 0, 1
    while lo <= max_seq:
        hi = min(lo + chunk_size - 1, max_seq)
        rows.append((chunk_id, lo, hi))
        chunk_id += 1
        lo = hi + 1

    if rows:
        con.executemany(f"INSERT INTO {_qi(chunks_table)} (chunk_id, seq_lo, seq_hi) VALUES (?, ?, ?)", rows)
    print(f"  {chunks_table}: {len(rows):,} chunks planned ({chunk_size:,} baskets each)")


def claim_next_chunk(con, dataset_tag: str, worker_id: str, stale_after_seconds: int = 3600):
    """
    Claims one pending (or crashed-and-stale) chunk. See the module-level
    note above about this NOT being safe for concurrent multi-process
    callers — correct only for this pipeline's actual single-process usage.
    Returns None when there's nothing left to claim.
    """
    _validate_tag(dataset_tag)
    chunks_table = f"inference_chunks_{dataset_tag}"

    # stale_after_seconds is always an internal int from our own code, never
    # user input, so inlining it into the INTERVAL literal is safe.
    row = con.execute(
        f"""
        UPDATE {_qi(chunks_table)} SET status = 'running', worker_id = ?, updated_at = now()
        WHERE chunk_id = (
            SELECT chunk_id FROM {_qi(chunks_table)}
            WHERE status = 'pending'
               OR (status = 'running' AND updated_at < now() - INTERVAL '{int(stale_after_seconds)} seconds')
            ORDER BY chunk_id
            LIMIT 1
        )
        RETURNING chunk_id, seq_lo, seq_hi
        """,
        [worker_id],
    ).fetchone()
    if row is None:
        return None
    return {"chunk_id": row[0], "seq_lo": row[1], "seq_hi": row[2]}


def mark_chunk_complete(con, dataset_tag: str, chunk_id: int):
    _validate_tag(dataset_tag)
    chunks_table = f"inference_chunks_{dataset_tag}"
    con.execute(f"UPDATE {_qi(chunks_table)} SET status = 'complete', updated_at = now() WHERE chunk_id = ?",
                [chunk_id])


def get_basket_range(con, dataset_tag: str, seq_lo: int, seq_hi: int) -> pd.DataFrame:
    """Fetch exactly one claimed chunk's basket rows."""
    _validate_tag(dataset_tag)
    baskets_table = f"baskets_{dataset_tag}"
    return _fetch_basket_range(con, baskets_table, seq_lo, seq_hi)


def count_baskets(con, dataset_tag: str) -> int:
    _validate_tag(dataset_tag)
    baskets_table = f"baskets_{dataset_tag}"
    return con.execute(f"SELECT COUNT(*) FROM {_qi(baskets_table)}").fetchone()[0]


def drop_all(con, dataset_tag: str):
    """Drops every table this module creates for a given dataset_tag — for
    fully resetting a tag (e.g. between independent test runs)."""
    _validate_tag(dataset_tag)
    for table in (f"baskets_{dataset_tag}", f"inference_chunks_{dataset_tag}"):
        con.execute(f"DROP TABLE IF EXISTS {_qi(table)}")


# ─────────────────────────────────────────────
# EXCLUDE ALREADY-EMBEDDED BASKETS (used by score_new_baskets.py)
# ─────────────────────────────────────────────

def exclude_existing_basket_ids(con, dataset_tag: str, existing_ids_parquet_path):
    """
    Removes rows from baskets_{tag} whose basket_id already appears in
    `existing_ids_parquet_path` (e.g. the training run's
    basket_gnn_embeddings.parquet) — the scoring-time equivalent of "skip
    baskets we've already embedded". DuckDB reads that parquet file
    directly for the anti-join; the existing-id set is never materialized
    as a Python object, however large the training population is.
    """
    _validate_tag(dataset_tag)
    existing_ids_parquet_path = Path(existing_ids_parquet_path)
    if not existing_ids_parquet_path.exists():
        print(f"  {existing_ids_parquet_path} not found — skipping already-embedded exclusion "
              f"(treating every basket in baskets_{dataset_tag} as new).")
        return

    baskets_table = f"baskets_{dataset_tag}"
    glob_path = _glob_for(existing_ids_parquet_path)

    n_before = con.execute(f"SELECT COUNT(*) FROM {_qi(baskets_table)}").fetchone()[0]
    con.execute(
        f"DELETE FROM {_qi(baskets_table)} WHERE basket_id IN "
        f"(SELECT basket_id FROM read_parquet(?))",
        [glob_path],
    )
    n_after = con.execute(f"SELECT COUNT(*) FROM {_qi(baskets_table)}").fetchone()[0]
    print(f"  Excluded {n_before - n_after:,} already-embedded baskets from {baskets_table}")
