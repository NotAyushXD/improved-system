"""
baseline_mean_embedding.py

The control the GNN has never been measured against.

WHY THIS EXISTS
───────────────
GNN_Train's loss reconstructs the per-graph MEAN of the raw node features
(batch_graph_targets). With product embeddings actually reaching the nodes,
emb_dim of the in_dim = emb_dim + 3 target dimensions are the basket's mean
product-embedding vector — so MSE is overwhelmingly dominated by it, and the
optimum the model is pushed toward is "compress the basket's average product
vector into OUT_DIM dims".

Note what that says about the graph: both GINEConv layers mix in neighbour
information that is NOT in the target, so message passing is closer to an
obstacle to this loss than an aid. The co-purchase edges and their two
features barely enter the gradient.

Which makes one question worth answering before tuning anything else:

    does the GNN beat simply averaging the product embeddings per basket?

This script computes that baseline. No training, no LMDB cache, no inference
pass — one streaming pass over the basket table. Its output is shaped exactly
like basket_gnn_embeddings.parquet (basket_id, gnn_embedding), so every
downstream step runs against it unchanged and the comparison is like for like:
same kNN, same Leiden, same profiling, same permutation null.

If the GNN cannot clear this on `excess` lift (observed minus its own null),
Stage 1 is elaborate machinery for a mean. If it can, you finally know what
the graph is buying and by how much — which nothing currently measures.

WHAT IT DOES
────────────
    basket -> its products              baskets_<tag> in DuckDB
    product -> 384-dim vector           product_embeddings.parquet
    basket vector                       mean of its products' vectors
    -> PCA to OUT_DIM (default 64)      so the dimensionality matches the GNN's
                                        and the kNN search is comparable

Products with no embedding are EXCLUDED from the mean rather than counted as
zero — a zero vector dragged toward the origin would make basket vectors a
function of how many unembedded products a basket happened to contain. Both
counts are reported.

The mean is over RAW embedding vectors, not L2-normalised ones, because that
is what the GNN sees: build_one_graph puts emb_matrix rows into node features
untouched, and global_mean_pool averages what the convolutions make of them.
Normalising per product first would be a different (defensible) baseline, and
a different experiment.

USAGE
─────
    python -u baseline_mean_embedding.py
    python -u baseline_mean_embedding.py --limit 500000      # quick smoke run

Then, exactly as for the GNN embeddings but tagged so nothing collides:

    python -u cluster_basket_embeddings.py --build-edges \\
        --embeddings ../data/output/basket_mean_embeddings.parquet --tag meanemb
    python -u cluster_leiden_networkit.py \\
        --edges ../data/output/basket_knn_edges_k10_onedir_meanemb.parquet \\
        --embeddings ../data/output/basket_mean_embeddings.parquet --resolution 1.5
    python -u profile_need_states.py \\
        --clusters ../data/output/basket_need_state_clusters_k10_onedir_meanemb_r1p5.parquet
    python -u evaluate_run.py \\
        --clusters ../data/output/basket_need_state_clusters_k10_onedir_meanemb_r1p5.parquet \\
        --note "control: mean product embedding, no GNN"

And its own permutation null, because the null moves with cluster size — see
BASKET_BANDING_DESIGN.md §5. Borrowing another run's floor misleads by up to
0.06 lift, which is a large share of the signal being measured.
"""

import argparse
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

import basket_store
import config
import duckdb_manager
import parquet_loader
from cluster_basket_embeddings import fmt_duration, progress_step

OUTPUT_DIR = config.OUTPUT_DIR
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "basket_mean_embeddings.parquet")

# Rows the PCA is fitted on. 64 components out of 384 dimensions needs nowhere
# near the full population, and a full-population fit would mean holding
# 57.1M x 384 float32 (~88GB) purely to learn 64 directions.
PCA_FIT_SAMPLE = 300_000


def _basket_means(chunk_df, emb_matrix, pid2idx):
    """
    Mean product-embedding vector per basket, for one chunk.

    Two vectorisations, both load-bearing at this scale:

    * The product -> row lookup goes through pandas .explode() + .map() rather
      than a nested Python loop. The full population is ~1.06B basket-product
      rows; a dict lookup per row in interpreter code is minutes of pure
      overhead for something pandas does in C.
    * The averaging is a sparse incidence matrix times the embedding matrix,
      one scipy matmul per chunk. It also gives baskets with no embedded
      product an all-zero row for free, where np.add.reduceat — the other
      obvious vectorisation — mishandles exactly that empty-segment case.

    Products absent from pid2idx have no embedding and are DROPPED, not
    averaged in as zeros: a zero vector would pull the basket toward the
    origin in proportion to how many unembedded products it happened to hold,
    making the basket vector partly a function of catalog coverage.

    Returns (means, n_products_used) — the second is per basket, so the caller
    can count and report the baskets that contributed nothing.
    """
    n = len(chunk_df)
    if n == 0:
        return np.zeros((0, emb_matrix.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    # Positional index, so exploded row labels ARE basket offsets within the
    # chunk. DuckDB's .df() already gives a RangeIndex, but .iloc slicing for
    # --limit happens upstream and this must not depend on that having kept it.
    exploded = chunk_df["products"].reset_index(drop=True).explode()
    codes = exploded.map(pid2idx)
    keep = codes.notna().to_numpy()

    rows = exploded.index.to_numpy()[keep].astype(np.int64)
    cols = codes.to_numpy()[keep].astype(np.int64)

    n_used = np.bincount(rows, minlength=n).astype(np.int64)
    if len(rows) == 0:
        return np.zeros((n, emb_matrix.shape[1]), dtype=np.float32), n_used

    incidence = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, emb_matrix.shape[0]),
    )
    sums = incidence @ emb_matrix                      # (n, emb_dim)
    return sums / np.maximum(n_used, 1)[:, None].astype(np.float32), n_used


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-tag", default=config.TRAIN_DATASET_TAG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--out-dim", type=int, default=config.OUT_DIM,
                    help="PCA components, so the vector width matches the GNN's "
                         f"(default PIPELINE_OUT_DIM = {config.OUT_DIM})")
    ap.add_argument("--chunk", type=int, default=config.INFERENCE_CHUNK_BASKETS)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="stop after N baskets — a smoke run, NOT a result. The "
                         "kNN graph over a truncated basket_seq prefix is not a "
                         "subgraph of anything meaningful.")
    args = ap.parse_args()

    started = time.perf_counter()
    con = duckdb_manager.get_connection()
    tag = args.dataset_tag

    # ── Product vectors, keyed exactly as the basket table keys them ─────
    product_df = parquet_loader.load_product_embeddings()
    pid2idx = {p: i for i, p in enumerate(product_df["tpnb"].tolist())}
    emb_matrix = np.vstack(product_df["embedding"].values).astype(np.float32)
    emb_dim = emb_matrix.shape[1]
    print(f"  {len(pid2idx):,} products x {emb_dim} dims")

    n_baskets = basket_store.count_baskets(con, tag)
    if args.limit:
        n_baskets = min(n_baskets, args.limit)
        print(f"  SMOKE RUN: {n_baskets:,} baskets only — not a result")
    if n_baskets == 0:
        raise SystemExit(f"baskets_{tag} is empty — run pipeline_main.py Stage 0 first.")
    print(f"  {n_baskets:,} baskets in baskets_{tag}")

    # ── Pass 1: fit the PCA on a bounded sample ──────────────────────────
    # Stratified by basket size (the sampler already does this), which is if
    # anything better than uniform here: it guarantees both tiny and huge
    # baskets inform the components rather than letting whichever size class
    # is most numerous dictate all 64 directions.
    with progress_step(f"fitting PCA({args.out_dim}) on a {PCA_FIT_SAMPLE:,}-basket sample", 1, 3):
        sample = basket_store.sample_training_baskets(
            con, tag, n_samples=PCA_FIT_SAMPLE, seed=config.SEED)
        sample_means, _ = _basket_means(sample, emb_matrix, pid2idx)
        pca = PCA(n_components=args.out_dim, random_state=config.SEED)
        pca.fit(sample_means)
        del sample, sample_means
        gc.collect()
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA keeps {explained:.1%} of the variance in {args.out_dim} of {emb_dim} dims")

    # ── Pass 2: stream every basket ──────────────────────────────────────
    # Per-chunk parquet then concat, the same shape GraphBuilder.run_inference /
    # merge_inference_output use, so peak memory is one chunk plus the final
    # frame rather than the whole population twice.
    chunk_dir = os.path.join(OUTPUT_DIR, f"_meanemb_chunks_{tag}")
    os.makedirs(chunk_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(chunk_dir, "*.parquet")):
        os.remove(stale)

    n_seen = n_empty = 0
    total_products = total_used = 0
    with progress_step(f"embedding {n_baskets:,} baskets", 2, 3):
        for chunk_i, chunk_df in enumerate(
                basket_store.stream_basket_chunks(con, tag, args.chunk)):
            if args.limit and n_seen >= args.limit:
                break
            if args.limit and n_seen + len(chunk_df) > args.limit:
                chunk_df = chunk_df.iloc[: args.limit - n_seen]

            means, n_used = _basket_means(chunk_df, emb_matrix, pid2idx)
            z = pca.transform(means).astype(np.float32)

            pd.DataFrame({
                "basket_id": chunk_df["basket_id"].tolist(),
                "gnn_embedding": list(z),
            }).to_parquet(os.path.join(chunk_dir, f"chunk_{chunk_i:06d}.parquet"),
                          index=False)

            n_seen += len(chunk_df)
            n_empty += int((n_used == 0).sum())
            total_used += int(n_used.sum())
            total_products += int(chunk_df["products"].map(len).sum())
            print(f"  {n_seen:,}/{n_baskets:,} baskets "
                  f"({n_seen / n_baskets:.1%})", flush=True)
            del chunk_df, means, z
            gc.collect()

    # Counted and printed, per the rule that anything dropped gets a number
    # against it. A large n_empty means product_embeddings.parquet does not
    # cover this basket population, and the control is measuring the coverage
    # gap rather than the method.
    dropped = total_products - total_used
    print(f"  products averaged: {total_used:,} of {total_products:,} "
          f"({dropped:,} had no embedding and were excluded from their basket's mean)")
    if n_empty:
        print(f"  WARNING: {n_empty:,} baskets ({n_empty / max(n_seen, 1):.2%}) had NO "
              f"embedded product at all and got a zero vector. They will cluster "
              f"together as an artifact, not as a need-state.")

    with progress_step("merging and writing", 3, 3):
        paths = sorted(glob.glob(os.path.join(chunk_dir, "*.parquet")))
        merged = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        n_dup = int(merged["basket_id"].duplicated().sum())
        if n_dup:
            raise ValueError(f"{n_dup:,} duplicate basket_id(s) across {len(paths)} "
                             f"chunk files under {chunk_dir} — refusing to write.")
        merged.to_parquet(args.out, index=False)
        for p in paths:
            os.remove(p)
        os.rmdir(chunk_dir)

    print(f"\nSaved {args.out} ({len(merged):,} baskets x {args.out_dim} dims) "
          f"in {fmt_duration(time.perf_counter() - started)}")
    print("\nThis file is shaped exactly like basket_gnn_embeddings.parquet, so the "
          "rest of Stage 2 runs against it unchanged:")
    edges = os.path.join(
        OUTPUT_DIR,
        f"basket_knn_edges_k{config.BASKET_KNN_K}_"
        f"{'mutual' if config.USE_MUTUAL_KNN else 'onedir'}_meanemb.parquet")
    labels = edges.replace("basket_knn_edges", "basket_need_state_clusters").replace(
        ".parquet", f"_r{str(float(config.LEIDEN_RESOLUTION)).replace('.', 'p')}.parquet")
    print(f"  python -u cluster_basket_embeddings.py --build-edges "
          f"--embeddings {args.out} --tag meanemb")
    print(f"  python -u cluster_leiden_networkit.py --edges {edges} "
          f"--embeddings {args.out} --resolution {config.LEIDEN_RESOLUTION}")
    print(f"  python -u profile_need_states.py --clusters {labels}")
    print(f"  python -u evaluate_run.py --clusters {labels} "
          f"--note \"control: mean product embedding, no GNN\"")
    print("\nCompare on `excess` (observed minus its OWN permutation null), not on "
          "raw median_max_lift — the null moves with cluster size, see "
          "BASKET_BANDING_DESIGN.md section 5.")


if __name__ == "__main__":
    main()
