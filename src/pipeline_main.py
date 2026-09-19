"""
pipeline_main.py

End-to-end pipeline:  your warehouse (manual export)  ->  GNN basket embeddings  ->  need-state clustering

    Stage 0   Read the two parquet files exported from your workspace, build the
              per-basket table in DuckDB directly from the household x tpnb x week
              export (parquet_loader.py for product embeddings,
              duckdb_manager.py + basket_store.py for baskets)
    Stage 1   Build the co-purchase matrix, train the GNN, embed every basket
    Stage 2   Cluster the resulting basket embeddings — BOTH Leiden and GMM    (cluster_basket_embeddings.py)
    Stage 3   Save need-state output as parquet for manual reload into your warehouse

THEME-FREE: no theme, category, or product_theme concept feeds this
pipeline anywhere, upstream or downstream. This supersedes an earlier
version of this file that worked around GraphBuilder.py's old
product_theme requirement by grouping products into arbitrary fixed-size
chunks labeled like themes — that requirement doesn't exist anymore.
GraphBuilder.py's prepare_globals() now clusters products GLOBALLY, across
the whole catalog, with no pre-grouping of any kind, and takes no
product_theme argument at all. See REFACTOR_NOTES.md for the full account
of what changed in GraphBuilder.py and why.

Need-states are found across the WHOLE basket — split_basket_by_theme.py
is not used anywhere in this pipeline (it's fully decommissioned, not just
unused — see REFACTOR_NOTES.md).

BASKET GRAIN — WEEK, NOT A TRUE SINGLE-VISIT BASKET: a "basket" here is
everything one household bought in one WEEK (household_number x
year_week_number), not one shopping trip. None of the tables available in
this warehouse carry a transaction/order/checkout identifier, so a true
single-visit basket can't be built from what's available — week is the
finest grain the data supports. See the comment at the top of
data/ns_household_tpnb_week_agg_train.sql for the full reasoning, and
data/TABLE_REFERENCE.md for what each source table actually contains.

BASKET STORAGE — DUCKDB, NOT A PYTHON DATAFRAME: `baskets` used to be
held as one Python object for this entire script's run, which at real
week-grain scale (tens of millions of baskets) is bigger than available
RAM. Basket storage now lives in a local, embedded DuckDB database
(duckdb_manager.py opens it — no server process, no privileged setup step;
basket_store.py owns the aggregate/stream/sample functions, and reads the
raw export straight from its parquet files via DuckDB's read_parquet()
rather than loading it into Python first). Nothing in this file ever holds
the full basket population in memory — Stage 1's co-purchase loop streams
bounded chunks, the training sample is a small SQL-drawn subset, and
inference streams the full population from DuckDB in restartable chunks
(see GraphBuilder.run_inference — restartable for a single process; DuckDB
has no row-level locking, so this isn't safe for concurrent processes).

(An earlier version of this used a self-contained Postgres instance via
`pgserver` instead of DuckDB — dropped because `initdb`'s privileged
directory-permission step was blocked outright on the target machine, even
running as Administrator, on every local drive tried. DuckDB needs no such
privileged step — it just opens an ordinary file.)

BREAKING CHANGE — cached artifact schema: GraphBuilder.py's node feature
layout changed (in_dim = emb_dim + 4, was + 5) and its sub-clustering
algorithm changed (global MiniBatchKMeans, was per-theme KMeans). The basket
grain also changed (household x WEEK, was household x PERIOD), which
invalidates everything derived from basket composition. Delete these under
data/output/ before running this version against any changed data or basket
definition:
    basket_gnn_model.pt, copurchase_sparse.npz,
    product_id_to_index.pkl, product_units_avg.pkl,
    basket_gnn_embeddings.parquet, gmm_basket_model.pkl,
    basket_need_state_clusters.parquet, embeddings_chunk_*.parquet,
    training_graphs.lmdb/ (and its .manifest.json)
product_subclusters.pkl and product_embeddings.parquet do NOT need deleting
for a basket-grain change — both are derived purely from product text
attributes, never from basket/purchase data. The co-purchase-matrix build's
own checkpoint (copurchase_sparse.checkpoint.npz + .progress.txt) and the
LMDB training-graph cache are both self-protecting: they fingerprint what
they were built from and rebuild automatically on mismatch, rather than
silently resuming into stale data. The DuckDB `baskets_train` table and
`inference_chunks_train` chunk-plan table are NOT fingerprinted — if you
change the source export or basket definition, drop them manually
(`basket_store.drop_all(con, "train")`) or just delete
data/pipeline.duckdb entirely to start clean.
There is no product_theme.pkl to delete — that file is never written by
this version, and never read by score_new_baskets.py either.

There's no live database connection to your WAREHOUSE anywhere in this file
(the DuckDB database it does use is a private, local, embedded file
managed by duckdb_manager.py — not your company's data warehouse). Before
running this, you need to have already run build_product_embeddings.py
(product embeddings don't exist yet, so that script builds them from
scratch — see its own docstring).

Run this from inside the src/ folder, with a sibling data/ folder holding
your downloaded tables:

    project/
    ├── data/
    │   ├── ns_item_lookup_tpna/                    (folder — Spark export)
    │   ├── ns_tpnb_to_tpna_mapping/                (folder)
    │   ├── ns_household_tpnb_week_agg_train/       (folder)
    │   ├── ns_household_tpnb_week_agg_score/       (folder — used later by score_new_baskets.py)
    │   ├── pipeline.duckdb                         (the local embedded DuckDB database file)
    │   └── output/                                 (everything this pipeline WRITES lands here —
    │                                                 product_embeddings.parquet, caches, the
    │                                                 trained model, embeddings, need-state output)
    └── src/
        ├── build_product_embeddings.py
        ├── pipeline_main.py
        └── ... (the rest)

    cd src
    python build_product_embeddings.py
    python pipeline_main.py

Stage 2 runs both Leiden and GMM clustering and reports how much they agree,
since need-states here are decided by comparing/combining both methods
rather than picking one. GMM_N_COMPONENTS below is a placeholder — swap it
for your team's real best-K selection once that logic is available; in the
meantime cluster_basket_embeddings.select_k_via_bic() gives a reasonable BIC
sweep to eyeball.

Run:  python pipeline_main.py [--worker-id NAME]
"""

import argparse
import os
import pickle
import gc

import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import parquet_loader
import duckdb_manager
import basket_store
from GNN_Train import train_and_embed
from cluster_basket_embeddings import (
    build_basket_knn_graph,
    run_leiden_on_basket_graph,
    cluster_basket_embeddings_gmm,
    compare_leiden_gmm,
    GMM_MODEL_PATH,
)
import need_state_graph
import config

# Drop degenerate 1-item baskets.
MIN_BASKET_PRODUCTS = config.MIN_BASKET_PRODUCTS

# Placeholder K for GMM — replace with your real best_k_value.py logic once
# shared. cluster_basket_embeddings.select_k_via_bic() sweeps a range and
# reports BIC/AIC per K if you want to eyeball it first.
GMM_N_COMPONENTS = config.GMM_N_COMPONENTS

# Co-purchase matrix is built in row-chunks of this many baskets at a time,
# with the running result checkpointed to disk after every chunk. This
# bounds peak memory during that step and survives a crash/restart without
# losing already-completed work — it does NOT change the result: X.T @ X is
# exactly additive over row-disjoint chunks of X (a basket only contributes
# co-purchase pairs to itself, never across baskets, and each chunk is a
# disjoint basket_seq range from the DuckDB baskets table).
# Lower this if you still see memory pressure; raise it for fewer, faster
# chunks once you've confirmed headroom.
COPURCHASE_CHUNK_BASKETS = config.COPURCHASE_CHUNK_BASKETS

# Chunk size for the restartable inference pass over the full basket
# population (see GraphBuilder.run_inference) — independent of the
# co-purchase chunk size above.
INFERENCE_CHUNK_BASKETS = config.INFERENCE_CHUNK_BASKETS

# All pipeline-produced artifacts (caches, models, embeddings, cluster
# output) live under this one folder — nothing gets written to the working
# directory. Input SQL exports you downloaded by hand stay under data/
# directly (they're inputs, not outputs of this script).
OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRODUCT_EMBEDDINGS_PARQUET  = config.out("product_embeddings.parquet")
HOUSEHOLD_TPNB_WEEK_PARQUET = config.HOUSEHOLD_TPNB_WEEK_TRAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-id", default=None,
        help="Label for this process in the restartable inference chunk queue "
             "(defaults to hostname-pid). DuckDB's chunk queue is only safe for "
             "one process at a time (see basket_store.claim_next_chunk) — this "
             "is mostly a diagnostic label for single-machine runs.",
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────
    # Cache-staleness guard
    # ─────────────────────────────────────────────
    # GraphBuilder.py caches product sub-clusters to a FIXED filename, reused
    # as-is on any future run regardless of whether the underlying data OR
    # CODE changed. If you're running this for the first time after a
    # feature-layout change and this file exists from before, IT WILL BE
    # LOADED AS-IS. The LMDB training-graph cache doesn't have this problem —
    # it checks a manifest (graph builder version, seed, sample size, in_dim)
    # and rebuilds automatically on mismatch, see lmdb_graph_cache.py.
    stale_cache = os.path.join(OUTPUT_DIR, "product_subclusters.pkl")
    if os.path.exists(stale_cache):
        print(f"WARNING: {stale_cache} exists and will be REUSED AS-IS by GraphBuilder "
              f"(cached by fixed filename, not by data or code fingerprint). If this is "
              f"left over from before a feature-layout change, or from a different "
              f"warehouse export, DELETE IT before continuing.")

    # ─────────────────────────────────────────────
    # Stage 0: read the manually-downloaded warehouse exports, build the
    # per-basket table in the local embedded DuckDB database
    # ─────────────────────────────────────────────
    print("[Stage 0] Reading warehouse exports...")
    product_df_2 = parquet_loader.load_product_embeddings(PRODUCT_EMBEDDINGS_PARQUET)

    con = duckdb_manager.get_connection()
    print("Building the per-basket table in DuckDB (reads the household x tpnb x week "
          "export directly from its parquet files — never held as one Python object, "
          "no separate load step needed)...")
    n_baskets_total, product_units_avg, product_uniques = basket_store.build_baskets_table(
        con, HOUSEHOLD_TPNB_WEEK_PARQUET, config.TRAIN_DATASET_TAG, min_basket_products=MIN_BASKET_PRODUCTS,
    )

    BASKET_COUNT_WARN_THRESHOLD = config.BASKET_COUNT_WARN_THRESHOLD
    if n_baskets_total > BASKET_COUNT_WARN_THRESHOLD:
        print(f"  NOTE: {n_baskets_total:,} baskets — this now lives in DuckDB, not Python "
              f"memory, so basket count alone no longer risks an OOM crash the way it used "
              f"to. If this looks unexpectedly large, check "
              f"ns_household_tpnb_week_agg_train.sql's MOD(household_number, N) = 0 filter is "
              f"actually being applied to the data you downloaded.")

    product_id_to_index = {pid: idx for idx, pid in enumerate(product_uniques)}
    n_products_cp = len(product_uniques)
    print(f"  {n_products_cp:,} distinct products across all baskets")

    # ─────────────────────────────────────────────
    # Stage 1a: co-purchase matrix — chunked + checkpointed, streamed from
    # DuckDB in row-disjoint basket_seq ranges
    # ─────────────────────────────────────────────
    print("\n[Stage 1] Building co-purchase matrix "
          f"(chunked {COPURCHASE_CHUNK_BASKETS:,} baskets at a time, checkpointed to disk)...")

    _CP_CHECKPOINT = os.path.join(OUTPUT_DIR, "copurchase_sparse.checkpoint.npz")
    _CP_PROGRESS   = os.path.join(OUTPUT_DIR, "copurchase_sparse.progress.txt")

    n_chunks = (n_baskets_total + COPURCHASE_CHUNK_BASKETS - 1) // COPURCHASE_CHUNK_BASKETS

    start_chunk = 0
    if os.path.exists(_CP_CHECKPOINT) and os.path.exists(_CP_PROGRESS):
        try:
            _progress_fields = open(_CP_PROGRESS).read().split(",")
            _ckpt_chunk = int(_progress_fields[0])
            _ckpt_fingerprint = tuple(_progress_fields[1:3]) if len(_progress_fields) >= 3 else None
        except (ValueError, IndexError):
            _ckpt_chunk = 0
            _ckpt_fingerprint = None
        _current_fingerprint = (str(n_baskets_total), str(n_products_cp))

        if _ckpt_fingerprint != _current_fingerprint:
            print(f"  IGNORING stale checkpoint at {_CP_CHECKPOINT}: it was built for "
                  f"n_baskets={_ckpt_fingerprint[0] if _ckpt_fingerprint else '?'}, "
                  f"n_products={_ckpt_fingerprint[1] if _ckpt_fingerprint else '?'}, but this run has "
                  f"n_baskets={n_baskets_total:,}, n_products={n_products_cp:,} — almost certainly "
                  f"a different dataset or basket grain. Rebuilding from scratch rather than risking "
                  f"silently blending stale partial results into this run.")
            copurchase_sparse = csr_matrix((n_products_cp, n_products_cp), dtype=np.int32)
        else:
            copurchase_sparse = sp.load_npz(_CP_CHECKPOINT).tocsr()
            start_chunk = _ckpt_chunk
            print(f"  Resuming from checkpoint: {start_chunk}/{n_chunks} chunks already done "
                  f"(nnz so far={copurchase_sparse.nnz:,}) — delete {_CP_CHECKPOINT} and "
                  f"{_CP_PROGRESS} if you want this step to start over from scratch instead.")
    else:
        copurchase_sparse = csr_matrix((n_products_cp, n_products_cp), dtype=np.int32)

    chunk_stream = basket_store.stream_basket_chunks(
        con, "train", COPURCHASE_CHUNK_BASKETS,
        start_seq=start_chunk * COPURCHASE_CHUNK_BASKETS + 1,
    )
    for chunk_i, chunk_df in zip(range(start_chunk, n_chunks), chunk_stream):
        hi = min((chunk_i + 1) * COPURCHASE_CHUNK_BASKETS, n_baskets_total)

        # Explode only this chunk's "products" column (not the whole row, which
        # would needlessly duplicate every other column — units, household_number,
        # etc. — across every exploded row, the way the old single-shot version did).
        chunk_items = chunk_df["products"].explode()
        if len(chunk_items) == 0:
            continue
        local_basket_codes, _ = pd.factorize(chunk_items.index, sort=False)
        product_idx = chunk_items.map(product_id_to_index).to_numpy()

        X_chunk = csr_matrix(
            (np.ones(len(chunk_items), dtype=np.int32), (local_basket_codes, product_idx)),
            shape=(local_basket_codes.max() + 1, n_products_cp),
        )
        copurchase_sparse = (copurchase_sparse + (X_chunk.T @ X_chunk)).tocsr()
        del chunk_df, chunk_items, local_basket_codes, product_idx, X_chunk
        gc.collect()

        tmp_ckpt = _CP_CHECKPOINT + ".writing.npz"
        sp.save_npz(tmp_ckpt, copurchase_sparse)
        os.replace(tmp_ckpt, _CP_CHECKPOINT)   # atomic — never leaves a truncated checkpoint

        tmp_progress = _CP_PROGRESS + ".tmp"
        with open(tmp_progress, "w") as f:
            f.write(f"{chunk_i + 1},{n_baskets_total},{n_products_cp}")
        os.replace(tmp_progress, _CP_PROGRESS)   # atomic — never leaves an empty/truncated progress file

        print(f"  chunk {chunk_i + 1}/{n_chunks}  ({hi:,}/{n_baskets_total:,} baskets)  "
              f"running nnz={copurchase_sparse.nnz:,}")

    if os.path.exists(_CP_CHECKPOINT):
        os.remove(_CP_CHECKPOINT)
    if os.path.exists(_CP_PROGRESS):
        os.remove(_CP_PROGRESS)
    print(f"  Co-purchase matrix complete: {n_products_cp:,} x {n_products_cp:,}, "
          f"nnz={copurchase_sparse.nnz:,}")

    product_embedding = dict(zip(product_df_2["tpnb"], product_df_2["embedding"]))

    # train_and_embed() only saves the model — it does NOT persist these
    # inputs, but score_new_baskets.py (scoring new baskets against the
    # trained model later) needs exactly these files on disk under exactly
    # these names. Save them now while they're in scope. No product_theme to
    # save — that concept doesn't exist in this pipeline.
    print("Saving inference-time artifacts (needed later to score new baskets)...")
    sp.save_npz(os.path.join(OUTPUT_DIR, "copurchase_sparse.npz"), copurchase_sparse.tocsr())
    with open(os.path.join(OUTPUT_DIR, "product_id_to_index.pkl"), "wb") as f:
        pickle.dump(product_id_to_index, f)
    with open(os.path.join(OUTPUT_DIR, "product_units_avg.pkl"), "wb") as f:
        pickle.dump(product_units_avg, f)
    print(f"  Saved copurchase_sparse.npz, product_id_to_index.pkl, product_units_avg.pkl to {OUTPUT_DIR}")

    # ─────────────────────────────────────────────
    # Stage 1b: train the GNN, embed every basket (restartable, chunked
    # inference — see GraphBuilder.run_inference)
    # ─────────────────────────────────────────────
    print("\n[Stage 1] Training GNN + embedding baskets...")
    basket_gnn_embeddings = train_and_embed(
        con                  = con,
        dataset_tag          = "train",
        product_embedding    = product_embedding,
        product_id_to_index  = product_id_to_index,
        copurchase_sparse    = copurchase_sparse,
        product_units_avg    = product_units_avg,
        worker_id            = args.worker_id,
    )

    # ─────────────────────────────────────────────
    # Stage 2: cluster the basket embeddings into need-states — Leiden AND GMM
    # ─────────────────────────────────────────────

    print("\n[Stage 2a] Leiden clustering...")
    # Called as two steps rather than via cluster_basket_embeddings(), which
    # builds `edges` internally and then drops it on return. The basket-level
    # edge list IS the cross-need-state structure — every edge whose endpoints
    # land in different communities is a boundary between two need-states — and
    # Stage 2.5 below needs it. Same graph, same Leiden call, same result;
    # the edge list just stays in scope now.
    basket_edges   = build_basket_knn_graph(basket_gnn_embeddings)
    leiden_clusters = run_leiden_on_basket_graph(basket_edges)
    print(f"\nNeed-state clusters found: {leiden_clusters['need_state_cluster'].nunique()}")

    print("\n[Stage 2b] GMM clustering (comparison / combination method)...")
    gmm_clusters = cluster_basket_embeddings_gmm(basket_gnn_embeddings, n_components=GMM_N_COMPONENTS)

    print("\n[Stage 2c] Comparing the two methods...")
    compare_leiden_gmm(leiden_clusters, gmm_clusters)

    # Both label sets kept side by side — need_state_cluster (Leiden) and
    # need_state_cluster_gmm (GMM) — rather than collapsing to one, since the
    # real decision (compare vs. combine, and how) isn't settled yet.
    need_state_clusters = leiden_clusters.merge(gmm_clusters, on="basket_id", how="outer")
    need_state_clusters_path = os.path.join(OUTPUT_DIR, "basket_need_state_clusters.parquet")
    need_state_clusters.to_parquet(need_state_clusters_path, index=False)
    print(f"  Saved {need_state_clusters_path} ({len(need_state_clusters):,} rows, "
          f"columns: need_state_cluster [Leiden], need_state_cluster_gmm [GMM])")

    # ─────────────────────────────────────────────
    # Stage 2.5: need-state GRAPHS — how need-states relate to each other
    # ─────────────────────────────────────────────
    # Stage 2 produces need-state LABELS. This produces the EDGES between
    # need-states, which the pipeline previously computed and discarded. Two
    # different edge sets, for two different questions (see need_state_graph.py):
    #   adjacency  — which need-states border each other (undirected, static)
    #   transitions — where households actually go next (directed, time-ordered)
    # Only the second can answer "what journey can this customer take".
    print("\n[Stage 2.5] Building need-state graphs...")
    _gmm_model = joblib.load(GMM_MODEL_PATH) if os.path.exists(GMM_MODEL_PATH) else None
    need_state_graph.build_and_save_all(
        edges                 = basket_edges,
        leiden_clusters       = leiden_clusters,
        basket_gnn_embeddings = basket_gnn_embeddings,
        gmm                   = _gmm_model,
    )

    # ─────────────────────────────────────────────
    # Stage 3: reload into your warehouse (manual)
    # ─────────────────────────────────────────────
    # No live write-back connection here, same as Stage 0. Two options:
    #   1. Run sql/06_write_back_need_states.sql once to create the landing table,
    #      then upload data/output/basket_need_state_clusters.parquet through your
    #      workspace web tool's import feature.
    #   2. If your web tool doesn't support parquet import, re-save as CSV first:
    #        need_state_clusters.to_csv(os.path.join(OUTPUT_DIR, "basket_need_state_clusters.csv"), index=False)

    print(f"\n[Stage 3] Skipped — reload {need_state_clusters_path} into "
          "your warehouse manually (see sql/06_write_back_need_states.sql).")

    print("\nPipeline complete.")

    # Themes, if wanted at all, are formed AFTER this point — by profiling the
    # need-states this run produced (e.g. summarizing each need_state_cluster's
    # dominant products/hierarchy) — never fed back in as an input. That
    # profiling step isn't implemented in this repository yet.


if __name__ == "__main__":
    # Windows `spawn` re-imports and re-executes the launching module in
    # every DataLoader worker process — without this guard, enabling
    # GNN_Train.py's NUM_WORKERS > 0 on Windows would re-run this entire
    # script (Stage 0, the co-purchase matrix build, everything) inside every
    # worker. This guard is what makes it safe to turn workers on.
    main()
