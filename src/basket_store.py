"""
basket_store.py

The shared basket-storage layer — used by BOTH pipeline_main.py (training)
and score_new_baskets.py (scoring), via a `dataset_tag` argument
("train" / "score") that keeps their tables separate in the same local
Postgres instance (see pg_manager.py). This is the fix for the recurring
memory bug in this pipeline: `baskets` was always held as one Python object
for the whole script, and at real week-grain scale that's bigger than
available RAM. Basket storage now lives in Postgres; nothing in this file
ever holds the full basket population in Python memory — every function
here either streams bounded chunks, or returns a result whose size is
bounded by something OTHER than basket count (catalog size, chunk size, or
the training-sample size).

Grain reminder: a "basket" here is everything one household bought in one
WEEK (household_number x year_week_number) — see
data/ns_household_tpnb_week_agg_train.sql for why this, not a true
single-visit basket, is the finest grain this warehouse supports.

Requires `psycopg2-binary` and a running Postgres connection URI from
pg_manager.get_connection_uri().
"""

import io
import re

import numpy as np
import pandas as pd

REQUIRED_RAW_COLS = ["household_number", "tpnb", "year_number", "period_number",
                     "week_number", "quantity"]

BASKET_COLS = ["household_number", "year_week_number", "basket_id", "products", "units"]

# Session-level tuning — these only affect QUERY SPEED, never correctness:
# Postgres spills hash aggregates / sorts to disk automatically past whatever
# work_mem is set to, so raising this is a performance knob, not a memory
# ceiling anything can silently overflow. No arbitrary cap here on purpose.
PG_WORK_MEM = "2GB"
PG_MAINTENANCE_WORK_MEM = "2GB"

# Lowercase-only on purpose: table names built from this are passed
# unquoted to to_regclass() in ensure_inference_chunk_plan(), which folds
# unquoted identifiers to lowercase per standard SQL rules — a mixed-case
# tag would make sql.Identifier()-created tables (case-preserving) invisible
# to that existence check. Restricting to lowercase avoids the mismatch
# entirely rather than requiring every lookup site to remember to quote.
_TAG_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _validate_tag(dataset_tag: str):
    if not _TAG_RE.match(dataset_tag):
        raise ValueError(
            f"dataset_tag must be a plain lowercase identifier matching "
            f"{_TAG_RE.pattern!r} (it's used to build table names, and is "
            f"looked up case-foldingly elsewhere) — got {dataset_tag!r}"
        )


def _seed_to_pg_seed(seed: int) -> float:
    """Postgres setseed() wants a float in [-1, 1] — deterministic map from an int seed."""
    return (seed % 2000 - 1000) / 1000.0


def _connect(conn_uri):
    import psycopg2
    return psycopg2.connect(conn_uri)


def _fetch_basket_range(conn, baskets_table: str, lo: int, hi: int) -> pd.DataFrame:
    from psycopg2 import sql
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT household_number, year_week_number, basket_id, products, units "
                "FROM {} WHERE basket_seq BETWEEN %s AND %s ORDER BY basket_seq"
            ).format(sql.Identifier(baskets_table)),
            (lo, hi),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=BASKET_COLS)


# ─────────────────────────────────────────────
# LOAD — stream the raw warehouse export straight into Postgres
# ─────────────────────────────────────────────

def load_raw_export_to_postgres(conn_uri, parquet_path, dataset_tag: str,
                                  batch_size: int = 20_000_000):
    """
    Streams the household x tpnb x week export (a Databricks/Spark
    folder-of-part-files export) via pyarrow.dataset.to_batches() — same
    bounded-batch read this pipeline already used in the retired
    parquet_loader.stream_build_baskets_and_units_avg() — but instead of
    accumulating into Python dicts, pipes each batch straight into a
    Postgres staging table via COPY. Only one batch (bounded by
    `batch_size` rows) is ever in Python memory at a time; nothing
    accumulates client-side across batches at all.
    """
    _validate_tag(dataset_tag)
    from pathlib import Path
    import pyarrow.dataset as pa_dataset
    from psycopg2 import sql

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found — run the household x tpnb x week export SQL "
            f"on your warehouse, download the result as parquet, and save it here."
        )

    dataset = pa_dataset.dataset(str(parquet_path), format="parquet")
    missing = set(REQUIRED_RAW_COLS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"{parquet_path} is missing columns {missing}.")

    staging_table = f"staging_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(staging_table)))
            cur.execute(sql.SQL("""
                CREATE TABLE {} (
                    household_number BIGINT,
                    tpnb TEXT,
                    year_number INTEGER,
                    period_number INTEGER,
                    week_number INTEGER,
                    quantity DOUBLE PRECISION
                )
            """).format(sql.Identifier(staging_table)))
        conn.commit()

        copy_sql = sql.SQL(
            "COPY {} (household_number, tpnb, year_number, period_number, week_number, quantity) "
            "FROM STDIN WITH (FORMAT csv)"
        ).format(sql.Identifier(staging_table)).as_string(conn)

        n_rows, n_batches = 0, 0
        for record_batch in dataset.to_batches(columns=REQUIRED_RAW_COLS, batch_size=batch_size):
            df = record_batch.to_pandas()
            df["tpnb"] = df["tpnb"].astype(str)
            buf = io.StringIO()
            df[REQUIRED_RAW_COLS].to_csv(buf, index=False, header=False)
            buf.seek(0)
            with conn.cursor() as cur:
                cur.copy_expert(copy_sql, buf)
            conn.commit()

            n_batches += 1
            n_rows += len(df)
            del df, buf
            print(f"  batch {n_batches}: {n_rows:,} rows loaded into Postgres "
                  f"({staging_table}) so far")

        print(f"Loaded {n_rows:,} household x tpnb x week rows across {n_batches} batches "
              f"into {staging_table} (from {parquet_path})")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# AGGREGATE — build the per-basket table, in Postgres, once
# ─────────────────────────────────────────────

def build_baskets_table(conn_uri, dataset_tag: str, min_basket_products: int = 2):
    """
    Replaces the old Python dict-accumulator basket-building step with one
    SQL pass. Postgres's own disk-spilling hash aggregate handles the
    "finalize a basket only once every one of its rows has been seen"
    correctness requirement natively — no assumption about row order in the
    staging table, no need to hold partial per-basket accumulators in Python.

    Returns (n_baskets_total, product_units_avg, product_uniques) — all three
    are bounded by catalog size or a single COUNT(*), never by basket count.
    """
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    staging_table = f"staging_{dataset_tag}"
    baskets_table = f"baskets_{dataset_tag}"

    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET work_mem = %s"), (PG_WORK_MEM,))
            cur.execute(sql.SQL("SET maintenance_work_mem = %s"), (PG_MAINTENANCE_WORK_MEM,))

            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(baskets_table)))
            cur.execute(sql.SQL("""
                CREATE TABLE {baskets} AS
                SELECT ROW_NUMBER() OVER (ORDER BY household_number, year_week_number) AS basket_seq,
                       household_number, year_week_number,
                       (household_number::text || '_' || year_week_number::text) AS basket_id,
                       array_agg(tpnb) AS products,
                       array_agg(quantity) AS units,
                       CASE
                           WHEN array_length(array_agg(tpnb), 1) <= 1  THEN 0
                           WHEN array_length(array_agg(tpnb), 1) <= 5  THEN 1
                           WHEN array_length(array_agg(tpnb), 1) <= 15 THEN 2
                           WHEN array_length(array_agg(tpnb), 1) <= 50 THEN 3
                           ELSE 4
                       END AS size_bucket
                FROM (
                    SELECT household_number,
                           year_number * 100 + week_number AS year_week_number,
                           tpnb, quantity
                    FROM {staging}
                ) t
                GROUP BY household_number, year_week_number
                HAVING array_length(array_agg(tpnb), 1) >= %(min_basket_products)s
            """).format(baskets=sql.Identifier(baskets_table), staging=sql.Identifier(staging_table)),
            {"min_basket_products": min_basket_products})

            cur.execute(sql.SQL("CREATE UNIQUE INDEX ON {} (basket_seq)").format(sql.Identifier(baskets_table)))
            cur.execute(sql.SQL("CREATE INDEX ON {} (size_bucket)").format(sql.Identifier(baskets_table)))
            cur.execute(sql.SQL("CREATE INDEX ON {} (basket_id)").format(sql.Identifier(baskets_table)))

            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(baskets_table)))
            n_baskets_total = cur.fetchone()[0]

            cur.execute(sql.SQL("SELECT tpnb, AVG(quantity) FROM {} GROUP BY tpnb")
                        .format(sql.Identifier(staging_table)))
            product_units_avg = {row[0]: float(row[1]) for row in cur.fetchall()}

            cur.execute(sql.SQL("SELECT DISTINCT tpnb FROM {} ORDER BY tpnb")
                        .format(sql.Identifier(staging_table)))
            product_uniques = np.array([row[0] for row in cur.fetchall()])

            cur.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(staging_table)))
        conn.commit()

        print(f"  {baskets_table}: {n_baskets_total:,} baskets, "
              f"{len(product_uniques):,} distinct products (staging table dropped)")
        return n_baskets_total, product_units_avg, product_uniques
    finally:
        conn.close()


# ─────────────────────────────────────────────
# STREAM — bounded-size chunks for the co-purchase matrix loop
# ─────────────────────────────────────────────

def stream_basket_chunks(conn_uri, dataset_tag: str, chunk_size: int, start_seq: int = 1):
    """
    Generator yielding one DataFrame (<= chunk_size rows) at a time, in
    basket_seq order. Used by pipeline_main.py's co-purchase chunk loop,
    which already has its own restart mechanism (the fingerprinted
    copurchase_sparse checkpoint) — this only needs to be a plain sequential
    stream, not a claimed queue.
    """
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    baskets_table = f"baskets_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT MAX(basket_seq) FROM {}").format(sql.Identifier(baskets_table)))
            max_seq = cur.fetchone()[0] or 0

        lo = start_seq
        while lo <= max_seq:
            hi = min(lo + chunk_size - 1, max_seq)
            df = _fetch_basket_range(conn, baskets_table, lo, hi)
            if len(df) > 0:
                yield df
            lo = hi + 1
    finally:
        conn.close()


# ─────────────────────────────────────────────
# SAMPLE — stratified-by-size training sample, computed in SQL
# ─────────────────────────────────────────────

def sample_training_baskets(conn_uri, dataset_tag: str, n_samples: int, seed: int) -> pd.DataFrame:
    """
    Same stratified-by-basket-size logic GraphBuilder.sample_baskets used to
    do in-memory (pd.cut over the whole `baskets` DataFrame) — computed here
    via the size_bucket column build_baskets_table() already persisted, with
    an index on it, so each stratum's random draw
    (`WHERE size_bucket = %s ORDER BY random() LIMIT %s`) only sorts that
    stratum's rows, not the whole population. Returns ONE DataFrame
    (n_samples rows) — the only basket-shaped object that stays fully
    resident in Python memory anywhere in this design, by construction far
    smaller than the full basket population.
    """
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    baskets_table = f"baskets_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET work_mem = %s"), (PG_WORK_MEM,))
            cur.execute("SELECT setseed(%s)", (_seed_to_pg_seed(seed),))
            cur.execute(sql.SQL("SELECT size_bucket, COUNT(*) FROM {} GROUP BY size_bucket")
                        .format(sql.Identifier(baskets_table)))
            stratum_counts = dict(cur.fetchall())

        total = sum(stratum_counts.values())
        if total == 0:
            print("  WARNING: baskets table is empty — nothing to sample.")
            return pd.DataFrame(columns=BASKET_COLS)

        parts = []
        with conn.cursor() as cur:
            for bucket, count in stratum_counts.items():
                if count == 0:
                    continue
                quota = min(count, max(1, round(n_samples * count / total)))
                cur.execute(sql.SQL(
                    "SELECT household_number, year_week_number, basket_id, products, units "
                    "FROM {} WHERE size_bucket = %s ORDER BY random() LIMIT %s"
                ).format(sql.Identifier(baskets_table)), (bucket, quota))
                rows = cur.fetchall()
                parts.append(pd.DataFrame(rows, columns=BASKET_COLS))

        sampled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BASKET_COLS)
        if len(sampled) > n_samples:
            sampled = sampled.sample(n=n_samples, random_state=seed).reset_index(drop=True)

        print(f"  Sampled {len(sampled):,} baskets across {len(stratum_counts)} "
              f"basket-size strata (via Postgres, indexed per-stratum random sort)")
        return sampled
    finally:
        conn.close()


# ─────────────────────────────────────────────
# RESTARTABLE INFERENCE CHUNK QUEUE
# A Postgres table, not just an in-memory generator — this is what makes a
# long inference run resumable after a crash: already-'complete' chunks are
# never redone, and a chunk left 'running' by a machine that crashed becomes
# claimable again automatically after `stale_after_seconds`.
# ─────────────────────────────────────────────

def ensure_inference_chunk_plan(conn_uri, dataset_tag: str, chunk_size: int):
    """
    Creates and populates the chunk-plan table ONCE per dataset_tag. If it
    already exists (a resumed run after a crash), it's left untouched so
    already-complete chunks are never redone.
    """
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    baskets_table = f"baskets_{dataset_tag}"
    chunks_table = f"inference_chunks_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (chunks_table,))
            already_exists = cur.fetchone()[0] is not None
            if already_exists:
                print(f"  {chunks_table} already exists — reusing as-is "
                      f"(resuming any pending/incomplete chunks rather than restarting)")
                return

            cur.execute(sql.SQL("SELECT MAX(basket_seq) FROM {}").format(sql.Identifier(baskets_table)))
            max_seq = cur.fetchone()[0] or 0

            cur.execute(sql.SQL("""
                CREATE TABLE {} (
                    chunk_id   INTEGER PRIMARY KEY,
                    seq_lo     BIGINT NOT NULL,
                    seq_hi     BIGINT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'pending',
                    worker_id  TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """).format(sql.Identifier(chunks_table)))

            rows, chunk_id, lo = [], 0, 1
            while lo <= max_seq:
                hi = min(lo + chunk_size - 1, max_seq)
                rows.append((chunk_id, lo, hi))
                chunk_id += 1
                lo = hi + 1

            if rows:
                insert_sql = sql.SQL(
                    "INSERT INTO {} (chunk_id, seq_lo, seq_hi) VALUES (%s, %s, %s)"
                ).format(sql.Identifier(chunks_table)).as_string(conn)
                with conn.cursor() as cur2:
                    cur2.executemany(insert_sql, rows)
        conn.commit()
        print(f"  {chunks_table}: {len(rows):,} chunks planned ({chunk_size:,} baskets each)")
    finally:
        conn.close()


def claim_next_chunk(conn_uri, dataset_tag: str, worker_id: str, stale_after_seconds: int = 3600):
    """
    Atomically claims one pending (or crashed-and-stale) chunk via
    `FOR UPDATE SKIP LOCKED` — the standard Postgres claimed-work-queue
    pattern. Returns None when there's nothing left to claim.
    """
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    chunks_table = f"inference_chunks_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("""
                UPDATE {t} SET status = 'running', worker_id = %(worker_id)s, updated_at = now()
                WHERE chunk_id = (
                    SELECT chunk_id FROM {t}
                    WHERE status = 'pending'
                       OR (status = 'running' AND updated_at < now() - (%(stale)s || ' seconds')::interval)
                    ORDER BY chunk_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING chunk_id, seq_lo, seq_hi
            """).format(t=sql.Identifier(chunks_table)),
            {"worker_id": worker_id, "stale": stale_after_seconds})
            row = cur.fetchone()
        conn.commit()
        if row is None:
            return None
        return {"chunk_id": row[0], "seq_lo": row[1], "seq_hi": row[2]}
    finally:
        conn.close()


def mark_chunk_complete(conn_uri, dataset_tag: str, chunk_id: int):
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    chunks_table = f"inference_chunks_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {} SET status = 'complete', updated_at = now() WHERE chunk_id = %s")
                        .format(sql.Identifier(chunks_table)), (chunk_id,))
        conn.commit()
    finally:
        conn.close()


def get_basket_range(conn_uri, dataset_tag: str, seq_lo: int, seq_hi: int) -> pd.DataFrame:
    """Fetch exactly one claimed chunk's basket rows."""
    _validate_tag(dataset_tag)
    baskets_table = f"baskets_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        return _fetch_basket_range(conn, baskets_table, seq_lo, seq_hi)
    finally:
        conn.close()


def drop_all(conn_uri, dataset_tag: str):
    """
    Drops every table this module creates for a given dataset_tag (staging,
    baskets, inference chunk-plan) — for fully resetting a tag (e.g. between
    independent test runs, or when the source export/basket definition
    changed and you want a clean rebuild rather than relying on the
    per-table staleness guards elsewhere).
    """
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            for table in (f"staging_{dataset_tag}", f"baskets_{dataset_tag}",
                          f"inference_chunks_{dataset_tag}"):
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))
        conn.commit()
    finally:
        conn.close()


def count_baskets(conn_uri, dataset_tag: str) -> int:
    _validate_tag(dataset_tag)
    from psycopg2 import sql

    baskets_table = f"baskets_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(baskets_table)))
            return cur.fetchone()[0]
    finally:
        conn.close()


# ─────────────────────────────────────────────
# EXCLUDE ALREADY-EMBEDDED BASKETS (used by score_new_baskets.py)
# ─────────────────────────────────────────────

def exclude_existing_basket_ids(conn_uri, dataset_tag: str, existing_ids_parquet_path,
                                  batch_size: int = 5_000_000):
    """
    Removes rows from baskets_{tag} whose basket_id already appears in
    `existing_ids_parquet_path` (e.g. the training run's
    basket_gnn_embeddings.parquet) — the scoring-time equivalent of "skip
    baskets we've already embedded". Streams ONLY the basket_id column
    (never the embedding vectors) in bounded batches straight into a
    throwaway Postgres table, then removes matches via one SQL anti-join.
    The existing-id set is never materialized as a Python object — at real
    scale it could be the entire training basket population (tens of
    millions of rows), which is exactly the kind of object this whole
    redesign exists to avoid holding in memory.
    """
    _validate_tag(dataset_tag)
    from pathlib import Path
    import pyarrow.dataset as pa_dataset
    from psycopg2 import sql

    existing_ids_parquet_path = Path(existing_ids_parquet_path)
    if not existing_ids_parquet_path.exists():
        print(f"  {existing_ids_parquet_path} not found — skipping already-embedded exclusion "
              f"(treating every basket in baskets_{dataset_tag} as new).")
        return

    baskets_table = f"baskets_{dataset_tag}"
    existing_table = f"existing_ids_{dataset_tag}"
    conn = _connect(conn_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(existing_table)))
            cur.execute(sql.SQL("CREATE TABLE {} (basket_id TEXT)").format(sql.Identifier(existing_table)))
        conn.commit()

        copy_sql = sql.SQL("COPY {} (basket_id) FROM STDIN WITH (FORMAT csv)") \
            .format(sql.Identifier(existing_table)).as_string(conn)

        dataset = pa_dataset.dataset(str(existing_ids_parquet_path), format="parquet")
        n_rows = 0
        for record_batch in dataset.to_batches(columns=["basket_id"], batch_size=batch_size):
            df = record_batch.to_pandas()
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False)
            buf.seek(0)
            with conn.cursor() as cur:
                cur.copy_expert(copy_sql, buf)
            conn.commit()
            n_rows += len(df)
            del df, buf

        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE INDEX ON {} (basket_id)").format(sql.Identifier(existing_table)))
            cur.execute(sql.SQL(
                "DELETE FROM {b} WHERE basket_id IN (SELECT basket_id FROM {e})"
            ).format(b=sql.Identifier(baskets_table), e=sql.Identifier(existing_table)))
            n_deleted = cur.rowcount
            cur.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(existing_table)))
        conn.commit()

        print(f"  Excluded {n_deleted:,} already-embedded baskets from {baskets_table} "
              f"(checked against {n_rows:,} existing basket_ids)")
    finally:
        conn.close()
