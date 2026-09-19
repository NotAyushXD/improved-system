"""
GraphBuilder.py

Builds per-basket graphs for the need-state GNN and runs inductive
inference over all baskets. This is a theme-free rewrite of the original
v4 graph builder (optimized_graph_builder_v4.py) — see REFACTOR_NOTES.md
in this same folder for the full account of what changed and why.

Summary of the theme-free architecture:
  - Product sub-clustering runs ONCE, GLOBALLY, across the whole product
    catalog — there is no pre-grouping of products by theme, category, or
    any other label before clustering. This is a real change in kind, not a
    relabeling: earlier iterations of this file used theme-grouped
    clustering, and an intermediate iteration used arbitrary fixed-size
    product chunks as a stand-in for themes — both are gone. Clustering
    here sees the entire catalog as one population.
  - Basket sampling is stratified by basket SIZE only — now computed in SQL
    against the DuckDB baskets table (basket_store.sample_training_baskets),
    not in-memory in this file; see that module.
  - Edge features are derived purely from co-purchase strength — the old
    same_theme edge flag is replaced by each edge's strength relative to
    its source node's own strongest co-purchase link, which is itself a
    purely co-purchase-derived signal.
  - There is no theme_ids array, no theme_score node feature, and no
    product_theme parameter anywhere in this file.
  - Node feature layout is now: [product_embedding] + [cp_score] +
    [sub_cluster_id] + [distinctiveness] + [log_units]  ->  in_dim = D + 4
    (previously D + 5, with the removed dimension being theme_score).
  - Training-graph construction (build once) lives in lmdb_graph_cache.py,
    caching to LMDB rather than one big in-memory list. Inference
    (run_inference / _embed_basket_chunk, below) builds a REAL per-basket
    graph for every basket — reusing build_one_graph, the exact same
    function training's LMDB cache uses — and runs it through the model's
    full encode() pipeline (node_encoder -> conv1 -> conv2 ->
    global_mean_pool -> proj). An earlier version of this function skipped
    both graph convolutions and used a different meaning for one feature
    slot than training did — a real train/score mismatch, independent of
    themes, fixed by construction: there is only one graph-building
    function, used everywhere.

IMPORTANT — this is a breaking change to cached artifacts. in_dim changed
(D+5 -> D+4) and the sub-clustering algorithm changed (per-theme KMeans ->
global MiniBatchKMeans), so every previously-cached file is stale and
incompatible:
    product_subclusters.pkl, basket_gnn_model.pt, copurchase_sparse.npz,
    product_id_to_index.pkl, product_units_avg.pkl
Delete these, and any LMDB training-graph cache directory, before running
anything against this version (the LMDB cache also self-checks a manifest
and rebuilds automatically on mismatch — see lmdb_graph_cache.py). There is
no product_theme.pkl anymore — that concept no longer exists. There is no
training_graphs.pkl anymore either — that was joblib-dumped Python list is
replaced by the LMDB cache.
"""

import os
import gc
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import scipy.sparse as sp
import joblib
from numba import njit
import numba
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TOP_K                = config.TOP_K
SEED                 = config.SEED

# Bumped any time build_one_graph()'s feature/edge layout changes (the same
# class of change that moved in_dim from D+5 to D+4). Compared against a
# saved manifest by lmdb_graph_cache.load_or_build_lmdb_cache() so a cached
# LMDB training-graph set built under a different feature layout is never
# silently reused.
# 3: fixed the edge relative-strength denominator (was an arbitrary neighbour
#    rather than the node's strongest link, for any node with <= TOP_K
#    neighbours). Edge feature [1] changes value, so every graph cached under
#    version 2 is stale — this bump forces the LMDB cache to rebuild instead of
#    training on the old edge scaling.
GRAPH_BUILDER_VERSION = 3

# All pipeline-produced artifacts (caches, models, embeddings) live under
# this folder rather than scattered in the working directory. Created here
# so any of this module's write sites can assume it already exists.
OUTPUT_DIR           = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global product sub-clustering (whole catalog, no theme/category
# pre-grouping of any kind). K is selected from these candidates via
# silhouette score on a sample — see _best_k_global(). Tune this range for
# your catalog size; a few hundred products call for a small handful of
# candidates near the low end, a few hundred thousand call for something
# like what's here.
SUBCL_K_CANDIDATES   = config.SUBCL_K_CANDIDATES
SUBCL_N_INIT         = config.SUBCL_N_INIT
SUBCL_BATCH_SIZE      = config.SUBCL_BATCH_SIZE
SUBCL_SIL_SAMPLE      = config.SUBCL_SIL_SAMPLE   # silhouette evaluated on a sample, not the whole catalog
SUBCL_CACHE_PATH      = os.path.join(OUTPUT_DIR, "product_subclusters.pkl")  # cached after first run


# ─────────────────────────────────────────────
# NUMBA EDGE KERNEL
# Edge features are purely co-purchase-derived: log co-purchase strength,
# and each edge's strength relative to its source node's own strongest
# co-purchase link. No theme/category input of any kind.
# ─────────────────────────────────────────────

@njit(cache=True)
def _build_edges_numba(indptr, indices, data, top_k):
    n         = len(indptr) - 1
    max_edges = n * top_k
    src_buf         = np.empty(max_edges, dtype=np.int64)
    dst_buf         = np.empty(max_edges, dtype=np.int64)
    logval_buf      = np.empty(max_edges, dtype=np.float32)
    relstrength_buf = np.empty(max_edges, dtype=np.float32)
    count = 0

    for i in range(n):
        rs, re  = indptr[i], indptr[i + 1]
        row_len = re - rs
        if row_len == 0:
            continue

        tmp_cols = np.empty(row_len, dtype=np.int64)
        tmp_vals = np.empty(row_len, dtype=np.float32)
        k = 0
        for jj in range(rs, re):
            j = indices[jj]
            if j != i:
                tmp_cols[k] = j
                tmp_vals[k] = data[jj]
                k += 1
        if k == 0:
            continue

        tmp_cols = tmp_cols[:k]
        tmp_vals = tmp_vals[:k]

        if k > top_k:
            selected = np.empty(top_k, dtype=np.int64)
            used     = np.zeros(k, dtype=numba.boolean)
            for s in range(top_k):
                best_val, best_idx = -1.0, -1
                for jj in range(k):
                    if not used[jj] and tmp_vals[jj] > best_val:
                        best_val = tmp_vals[jj]
                        best_idx = jj
                selected[s]    = best_idx
                used[best_idx] = True
            final_cols = tmp_cols[selected]
            final_vals = tmp_vals[selected]
        else:
            final_cols = tmp_cols
            final_vals = tmp_vals

        # Each selected edge is expressed as a fraction of this node's
        # STRONGEST co-purchase link — a purely co-purchase-derived
        # relative-strength signal, replacing the old same_theme edge flag.
        #
        # This used to read `final_vals[0]`, on the reasoning that the
        # selection loop above emits the largest value first. That is true
        # ONLY in the `k > top_k` branch, where the greedy loop runs. In the
        # `else` branch — every node with <= top_k neighbours, i.e. every
        # basket of <= 11 products, which is a large share of week-grain
        # baskets — final_vals is still in CSR COLUMN order, so final_vals[0]
        # was an arbitrary neighbour rather than the strongest one. With
        # co-purchase counts of 30/90/60 that produced relative strengths of
        # 1.0/3.0/2.0 instead of 0.33/1.0/0.67: the documented invariant
        # (every value in (0, 1], exactly one equal to 1.0) was violated, and
        # small and large baskets had their edge features on different scales.
        #
        # Scanning for the max explicitly is correct in both branches. Written
        # as a loop rather than final_vals.max() deliberately — it is trivially
        # verifiable under numba's typing with no reliance on array-method
        # support, and it compiles to the same thing.
        max_val_for_node = final_vals[0]
        for s in range(1, len(final_vals)):
            if final_vals[s] > max_val_for_node:
                max_val_for_node = final_vals[s]
        if max_val_for_node <= 0:
            max_val_for_node = 1.0

        for s in range(len(final_cols)):
            j = final_cols[s]
            src_buf[count]         = i
            dst_buf[count]         = j
            logval_buf[count]      = np.log1p(final_vals[s])
            relstrength_buf[count] = final_vals[s] / max_val_for_node
            count += 1

    return src_buf[:count], dst_buf[:count], logval_buf[:count], relstrength_buf[:count]


def _warmup_numba():
    _build_edges_numba(
        np.array([0, 2, 3], dtype=np.int64),
        np.array([1, 0, 0], dtype=np.int64),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        10,
    )


# ─────────────────────────────────────────────
# PRODUCT SUB-CLUSTERING — GLOBAL, WHOLE CATALOG
# No theme/category/chunk pre-grouping of any kind. One clustering pass
# across every embedded product.
# ─────────────────────────────────────────────

def _distances_to_centroids(X, centroids):
    """
    Memory-safe (n, k) distance matrix via the algebraic expansion
    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c — O(n*k) memory. A naive
    X[:, None, :] - centroids[None, :, :] broadcast is O(n*k*d), which at
    whole-catalog scale (n in the hundreds of thousands) is multiple
    gigabytes for a single temporary array; this is a couple hundred MB at
    most for the same inputs.
    """
    X         = X.astype(np.float32, copy=False)
    centroids = centroids.astype(np.float32, copy=False)
    x_sq   = np.sum(X ** 2, axis=1, keepdims=True)   # (n, 1)
    c_sq   = np.sum(centroids ** 2, axis=1)            # (k,)
    cross    = X @ centroids.T                                         # (n, k)
    two      = np.float32(2.0)   # explicit dtype — a plain Python float here can
                                  # silently upcast a float32 array to float64 on
                                  # some numpy versions, doubling this array's size
    sq_dists = np.maximum(x_sq + c_sq[None, :] - two * cross, np.float32(0.0))
    return np.sqrt(sq_dists, dtype=np.float32)


def _best_k_global(X, k_candidates, n_init, batch_size, sil_sample, seed):
    """
    Picks K for global product sub-clustering via silhouette score,
    evaluated on a fixed-size sample (silhouette's own cost grows with the
    sample it's given — sklearn's sample_size parameter bounds this the
    same way the old per-theme version did, just applied once across the
    whole catalog instead of many small per-theme calls).
    """
    best_k, best_sil = k_candidates[0], -1.0
    for k in k_candidates:
        if k >= len(X):
            continue
        km = MiniBatchKMeans(n_clusters=k, n_init=n_init, batch_size=batch_size,
                             random_state=seed, max_iter=100)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X, labels, metric="euclidean",
                               sample_size=min(sil_sample, len(X)), random_state=seed)
        print(f"    k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k


def build_product_subclusters(product_embedding):
    """
    One global clustering pass over every embedded product — no theme,
    category, or arbitrary chunk grouping beforehand. Uses MiniBatchKMeans,
    which scales to hundreds of thousands of products without the
    memory/runtime cost a full KMeans (or many full KMeans fits) needs at
    that size, plus the memory-safe distance computation above for the
    distinctiveness calculation.

    Parameters
    ----------
    product_embedding : dict  tpnb -> embedding vector

    Returns
    -------
    product_subcluster     : dict  tpnb -> normalised sub-cluster id float [0,1]
    product_distinctiveness: dict  tpnb -> float [0,1]
    best_k                  : int  (for reference/logging)
    """
    # Fingerprint guard. This cache used to be keyed by FILENAME ALONE and
    # reused as-is on any future run — pipeline_main.py printed a warning
    # about it and relied on the operator to delete the file by hand. Now that
    # SUBCL_* and SEED are env-tunable (config.py), that is no longer good
    # enough: changing a value in .env would silently reuse sub-clusters built
    # under the old settings. The fingerprint is stored inside the cache and
    # compared here, so a settings change rebuilds automatically.
    expected_fp = config.subcluster_fingerprint()
    if os.path.exists(SUBCL_CACHE_PATH):
        print(f"  Loading cached product sub-clusters from {SUBCL_CACHE_PATH}...")
        try:
            cached = joblib.load(SUBCL_CACHE_PATH)
            if isinstance(cached, dict) and cached.get("fingerprint") == expected_fp:
                return cached["product_subcluster"], cached["product_distinctiveness"], cached["best_k"]
            if isinstance(cached, dict):
                print(f"  IGNORING cached sub-clusters: built with "
                      f"{cached.get('fingerprint')}, this run wants {expected_fp} — "
                      f"rebuilding rather than mixing settings.")
            else:
                print(f"  IGNORING cached sub-clusters at {SUBCL_CACHE_PATH}: written by an "
                      f"older version with no fingerprint, so what settings produced it "
                      f"cannot be verified. Rebuilding from scratch.")
        except Exception as e:
            print(f"  Cached file exists but failed to load ({type(e).__name__}: {e}) — "
                  f"treating it as stale and rebuilding from scratch.")

    print("  Building GLOBAL product sub-clusters (no theme/category pre-grouping)...")

    tpnbs = list(product_embedding.keys())
    X = np.vstack([product_embedding[t] for t in tpnbs]).astype(np.float32)
    X = normalize(X, norm="l2")

    print(f"  Selecting k across the whole catalog ({len(tpnbs):,} products)...")
    best_k = _best_k_global(
        X, SUBCL_K_CANDIDATES, SUBCL_N_INIT, SUBCL_BATCH_SIZE, SUBCL_SIL_SAMPLE, SEED,
    )
    print(f"  Selected k={best_k}")

    print(f"  Fitting final clustering (k={best_k}, n_init={SUBCL_N_INIT})...")
    km = MiniBatchKMeans(n_clusters=best_k, n_init=SUBCL_N_INIT,
                         batch_size=SUBCL_BATCH_SIZE, random_state=SEED, max_iter=200)
    labels = km.fit_predict(X)
    print(f"  Final fit done. Computing distinctiveness ({len(X):,} x {best_k})...")

    centroids       = km.cluster_centers_
    all_dists       = _distances_to_centroids(X, centroids)          # (n, k), memory-safe
    assigned_dists  = all_dists[np.arange(len(X)), labels]
    max_dists       = all_dists.max(axis=1)
    distinctiveness = 1.0 - (assigned_dists / (max_dists + 1e-8))
    print(f"  Distinctiveness computed.")

    product_subcluster      = {}
    product_distinctiveness = {}
    for i, t in enumerate(tpnbs):
        product_subcluster[t]      = float(labels[i]) / max(best_k - 1, 1)
        product_distinctiveness[t] = float(distinctiveness[i])

    result = (product_subcluster, product_distinctiveness, best_k)
    # Stored as a dict carrying the fingerprint, so the loader above can tell
    # what settings produced it rather than trusting the filename.
    payload = {
        "fingerprint": expected_fp,
        "product_subcluster": product_subcluster,
        "product_distinctiveness": product_distinctiveness,
        "best_k": best_k,
    }
    tmp_path = SUBCL_CACHE_PATH + ".tmp"
    try:
        joblib.dump(payload, tmp_path, compress=0)
        os.replace(tmp_path, SUBCL_CACHE_PATH)   # atomic — never leaves a truncated file
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    print(f"  Sub-clusters cached to {SUBCL_CACHE_PATH}")
    return result


# ─────────────────────────────────────────────
# PREPARE GLOBALS
# ─────────────────────────────────────────────

def prepare_globals(product_embedding, product_id_to_index,
                    copurchase_sparse, product_units_avg):
    """
    Converts all dicts to numpy arrays.

    in_dim = emb_dim + 4
        [emb_dim]  product embedding
        [+1]       co-purchase score
        [+1]       sub_cluster_id  (normalised — from GLOBAL clustering, no
                                     theme/category pre-grouping)
        [+1]       distinctiveness
        [+1]       log_units       (filled per basket at graph build time)

    No theme/category parameter anywhere in this function's signature —
    need-states are discovered downstream of basket embeddings; nothing
    here segments products by category first.
    """
    print("  Building embedding matrix...")
    n_products = len(product_id_to_index)
    sample_emb = next(iter(product_embedding.values()))
    emb_dim    = len(sample_emb)

    emb_matrix = np.zeros((n_products, emb_dim), dtype=np.float32)
    for product, idx in product_id_to_index.items():
        if product in product_embedding:
            emb_matrix[idx] = product_embedding[product]

    print("  Building product sub-clusters (global, no theme/category pre-grouping)...")
    product_subcluster, product_distinctiveness, best_k = \
        build_product_subclusters(product_embedding)
    print(f"  Global sub-clusters: k={best_k}")

    subcluster_arr      = np.zeros(n_products, dtype=np.float32)
    distinctiveness_arr = np.zeros(n_products, dtype=np.float32)
    avg_units_arr        = np.ones(n_products, dtype=np.float32)

    for product, idx in product_id_to_index.items():
        subcluster_arr[idx]      = product_subcluster.get(product, 0.0)
        distinctiveness_arr[idx] = product_distinctiveness.get(product, 0.5)
        avg_units_arr[idx]       = product_units_avg.get(product, 1.0)

    # No whole-matrix dtype cast here on purpose: .astype(np.float32) on a
    # different dtype always allocates a brand-new copy of the ENTIRE
    # matrix (at real scale, tens of GB) held alongside the original —
    # and it's unnecessary, since _basket_dense_cp_submatrix() already
    # casts to float32 on its own tiny per-basket slice. .tocsr() alone is
    # a no-op (returns self, no copy) when the matrix is already CSR, which
    # it is here.
    print("  Using co-purchase matrix as CSR (no whole-matrix dtype copy)...")
    csr = copurchase_sparse.tocsr()

    # _basket_dense_cp_submatrix() binary-searches each row's column indices,
    # which requires them SORTED and DE-DUPLICATED. tocsr() normally produces
    # that already, so both calls below are usually no-ops — but "usually" is
    # not good enough when the failure mode is silently reading the wrong
    # co-purchase counts into every node feature and edge in the pipeline.
    # Both operate in place; neither copies the matrix.
    if not csr.has_canonical_format:
        print("  Normalising co-purchase matrix to canonical CSR "
              "(sorted, de-duplicated indices) — required by the per-basket "
              "submatrix lookup...")
        csr.sum_duplicates()
    csr.sort_indices()

    return dict(
        emb_matrix          = emb_matrix,
        emb_dim             = emb_dim,
        in_dim              = emb_dim + 4,   # embedding + 4 extra features — no theme_score
        subcluster_arr      = subcluster_arr,
        distinctiveness_arr = distinctiveness_arr,
        avg_units_arr       = avg_units_arr,
        csr                 = csr,
        product_id_to_index = product_id_to_index,
    )


# ─────────────────────────────────────────────
# Stratified-by-size basket sampling now lives in basket_store.py
# (sample_training_baskets) — computed via SQL against the DuckDB baskets
# table instead of in-memory pd.cut over a full `baskets` DataFrame, since
# that DataFrame no longer exists as one in-memory object anywhere in this
# pipeline. See basket_store.sample_training_baskets for the equivalent
# logic (same stratification bins, same intent).
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# PER-BASKET DENSE CO-PURCHASE SUBMATRIX
#
# There is deliberately no whole-sample dense co-purchase matrix anywhere in
# this file anymore. An earlier version pre-built ONE dense matrix covering
# every product touched anywhere across the whole training sample — but at
# real catalog sizes, a random sample of tens/hundreds of thousands of
# baskets typically touches nearly the ENTIRE catalog (a basic
# coupon-collector argument: covering a catalog of size N needs on the
# order of N*ln(N) product draws, far fewer than a large training sample
# actually produces), so that matrix tends toward catalog_size^2 * 4 bytes —
# well over 100GB for a ~200k-product catalog — no matter how carefully the
# sample is chosen. There is no safe fixed size threshold to fall back on,
# so instead this is solved by never needing that matrix at all: every
# basket only ever needs co-purchase counts between ITS OWN products, so
# each basket gets its own small submatrix, sliced from the sparse GLOBAL
# co-purchase matrix on the fly. Its size is bounded by that one basket's
# product count squared — independent of catalog size — and is used
# identically by both lmdb_graph_cache.build_lmdb_cache() (training) and
# _embed_basket_chunk() (inference/scoring), so both remain guaranteed to
# compute this the same way.
# ─────────────────────────────────────────────

def _basket_dense_cp_submatrix(products, csr, pid2idx):
    """
    Extracts the (n x n) co-purchase submatrix for ONE basket's products.

    PERFORMANCE — this is the hottest function in the pipeline; it runs once
    per basket, so at 57M baskets a millisecond here is 16 hours of wall clock.

    The previous implementation was:

        csr[global_idx_arr][:, global_idx_arr].todense()

    which is correct but pathologically slow at catalog scale. `csr[rows]`
    materialises the ENTIRE rows first — every nonzero in them — and only then
    does `[:, cols]` narrow to the basket's own products. A row for a popular
    product (milk, bread) co-occurs with a large fraction of the catalog, so a
    single row can carry 100k+ nonzeros; a 30-item basket therefore copied
    millions of entries to keep 900 of them. Measured at ~20ms per basket,
    which is ~9 days of inference over a 57M-basket population.

    This version instead binary-searches each row's sorted column indices for
    just the basket's own products: n searchsorted calls over n targets each,
    independent of how many nonzeros the row actually has. Same numbers, same
    dtype, same zeros — see test_pipeline.check_graph_primitives(), which
    asserts the two implementations agree exactly.

    REQUIRES canonical CSR (sorted, de-duplicated column indices per row).
    prepare_globals() guarantees that via sum_duplicates()/sort_indices().
    """
    global_idx_arr = np.array(
        sorted({pid2idx[p] for p in products if p in pid2idx}), dtype=np.int64
    )
    n = len(global_idx_arr)
    if n == 0:
        return {}, np.zeros((0, 0), dtype=np.float32)
    local_idx_map = {int(g): l for l, g in enumerate(global_idx_arr)}

    indptr, indices, data = csr.indptr, csr.indices, csr.data
    # searchsorted compares against the index array's own dtype; matching it
    # here avoids an int32->int64 upcast of the (large) row slice on every call.
    targets = global_idx_arr.astype(indices.dtype, copy=False)

    dense_cp = np.zeros((n, n), dtype=np.float32)
    for li in range(n):
        gi = global_idx_arr[li]
        start, end = indptr[gi], indptr[gi + 1]
        if start == end:
            continue                      # product co-occurs with nothing
        row_cols = indices[start:end]     # sorted — canonical CSR
        pos = np.searchsorted(row_cols, targets)
        np.clip(pos, 0, len(row_cols) - 1, out=pos)
        hit = row_cols[pos] == targets    # exact match, not just insertion point
        if hit.any():
            dense_cp[li, hit] = data[start:end][pos[hit]]

    return local_idx_map, dense_cp


# ─────────────────────────────────────────────
# BUILD ONE GRAPH  (volume-weighted, sub-cluster + distinctiveness features)
# Used identically by both training-graph construction (lmdb_graph_cache.py)
# and inference (_embed_basket_chunk below), so training and scoring can
# never drift apart in what features they compute.
# ─────────────────────────────────────────────

def _minmax(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr) if mx > 0 else np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def build_one_graph(
    products, units, basket_id,
    emb_matrix, emb_dim,
    subcluster_arr, distinctiveness_arr,
    dense_cp, local_idx,
    product_id_to_index,
):
    """
    Node feature layout (in_dim = emb_dim + 4):
      [0:emb_dim]    product embedding
      [emb_dim]      cp_score
      [emb_dim+1]    sub_cluster_id  (normalised 0-1, from global clustering)
      [emb_dim+2]    distinctiveness
      [emb_dim+3]    log_units

    No theme/category input anywhere in this function.
    """
    in_dim = emb_dim + 4

    # Deduplicate — keep first occurrence, sum units for duplicates
    seen      = {}
    unit_map  = {}
    for p, u in zip(products, units):
        if p not in seen and p in product_id_to_index:
            gi = product_id_to_index[p]
            if gi in local_idx:
                seen[p]     = gi
                unit_map[p] = float(u) if u is not None else 1.0
            elif p not in seen:
                # product not in the dense submatrix passed in — skip
                pass
        elif p in seen:
            unit_map[p] = unit_map.get(p, 1.0) + (float(u) if u is not None else 0.0)

    products_clean = list(seen.keys())
    n              = len(products_clean)

    if n == 0:
        x = torch.zeros((1, in_dim), dtype=torch.float)
        return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 2), dtype=torch.float), basket_id=basket_id)

    global_idx_arr = np.array([seen[p] for p in products_clean], dtype=np.int64)
    local_idx_arr  = np.array([local_idx[gi] for gi in global_idx_arr], dtype=np.int64)
    units_arr      = np.array([unit_map[p] for p in products_clean], dtype=np.float32)
    log_units      = np.log1p(units_arr)

    # ── Node features ──
    emb      = emb_matrix[global_idx_arr]          # (n, emb_dim)
    subcl    = subcluster_arr[global_idx_arr]       # (n,) normalised [0,1]
    distinct = distinctiveness_arr[global_idx_arr]  # (n,) [0,1]

    # Co-purchase node score (from the dense submatrix passed in — training
    # passes a training-sample-scoped submatrix; inference passes a small,
    # freshly built basket-scoped one — see embed_all_baskets_fast)
    if n >= 2:
        sub      = dense_cp[np.ix_(local_idx_arr, local_idx_arr)]
        row_sums = sub.sum(axis=1)
        diag     = sub.diagonal()
        cp       = np.log1p((row_sums - diag) / (n - 1)).astype(np.float32)
        cp       = _minmax(cp)
    else:
        cp = np.zeros(n, dtype=np.float32)

    # Stack all features: [emb | cp | subcluster | distinctiveness | log_units]
    feat = np.concatenate([
        emb,
        cp[:, None],
        subcl[:, None],
        distinct[:, None],
        log_units[:, None],
    ], axis=1).astype(np.float32)

    # Sanitize — replace NaN/inf with 0 so bad products never poison a training batch
    np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    x = torch.tensor(feat, dtype=torch.float)

    # ── Edges — purely co-purchase-derived, see _build_edges_numba ──
    if n > 1:
        sub_csr = sp.csr_matrix(dense_cp[np.ix_(local_idx_arr, local_idx_arr)])
        src, dst, logv, rel = _build_edges_numba(
            sub_csr.indptr.astype(np.int64),
            sub_csr.indices.astype(np.int64),
            sub_csr.data.astype(np.float32),
            TOP_K,
        )
        if len(src) > 0:
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
            edge_attr  = torch.tensor(np.stack([logv, rel], axis=1), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 2), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 2), dtype=torch.float)

    data           = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.basket_id = basket_id
    return data


# ─────────────────────────────────────────────
# Training-graph construction (build once, cache to LMDB) now lives in
# lmdb_graph_cache.py (build_lmdb_cache / load_or_build_lmdb_cache) — it
# calls _basket_dense_cp_submatrix + build_one_graph above, exactly as this
# module's old build_training_graphs() did, but writes each graph straight
# to LMDB instead of accumulating a Python list (~20-25GB at real sample
# sizes) that then got joblib-dumped to disk. See that module's docstring.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# INDUCTIVE INFERENCE — restartable, bounded-memory, one chunk at a time
#
# _embed_basket_chunk builds a REAL per-basket graph for every basket in ONE
# chunk — reusing build_one_graph, the exact same function training's LMDB
# cache uses — and runs it through the model's full encode() pipeline
# (node_encoder -> conv1 -> conv2 -> global_mean_pool -> proj). This is the
# same graph-then-encode logic the old single-shot embed_all_baskets_fast()
# used, just scoped to whatever one chunk run_inference() hands it.
#
# run_inference is the new outer driver: it claims chunks one at a time from
# a DuckDB work queue (basket_store.claim_next_chunk), so a run that
# crashes partway through (a machine going down 8 hours into a long
# inference job) resumes from whatever's left pending/stale rather than
# starting over. Each chunk's embeddings are written straight to their own
# parquet file and the chunk is marked complete immediately after — nothing
# here ever holds more than one chunk's worth of baskets, graphs, or
# embeddings in memory at once, regardless of how many baskets exist in
# total.
# ─────────────────────────────────────────────

def _embed_basket_chunk(chunk_df, G, model, device, batch_size=4096):
    """
    Builds and encodes graphs for exactly the baskets in `chunk_df` — no
    accumulation across chunks happens here, that's run_inference()'s job.
    Returns (basket_ids, z_array) for this one chunk only.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    has_units    = "units" in chunk_df.columns
    all_products = chunk_df["products"].tolist()
    all_units    = chunk_df["units"].tolist() if has_units else None
    basket_ids   = (
        chunk_df["basket_id"].tolist()
        if "basket_id" in chunk_df.columns
        else list(range(len(chunk_df)))
    )

    csr     = G["csr"]
    pid2idx = G["product_id_to_index"]

    graphs = []
    for i in range(len(chunk_df)):
        products = all_products[i]
        units    = all_units[i] if all_units is not None else [1.0] * len(products)
        local_idx_map, basket_dense_cp = _basket_dense_cp_submatrix(products, csr, pid2idx)
        g = build_one_graph(
            products             = products,
            units                = units,
            basket_id            = basket_ids[i],
            emb_matrix           = G["emb_matrix"],
            emb_dim              = G["emb_dim"],
            subcluster_arr       = G["subcluster_arr"],
            distinctiveness_arr  = G["distinctiveness_arr"],
            dense_cp             = basket_dense_cp,
            local_idx            = local_idx_map,
            product_id_to_index  = pid2idx,
        )
        graphs.append(g)

    if not graphs:
        return [], np.empty((0, 0), dtype=np.float32)

    model.eval()
    z_parts = []
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch)
            z_parts.append(z.detach().cpu().numpy().astype(np.float32))

    del graphs, loader
    gc.collect()

    return basket_ids, np.vstack(z_parts)


def run_inference(con, dataset_tag, G, model, device, worker_id=None,
                    chunk_size=None, batch_size=None, output_dir=None,
                    stale_after_seconds=None):
    """
    Claims one chunk at a time from the DuckDB inference-chunk queue
    (basket_store.ensure_inference_chunk_plan / claim_next_chunk),
    embeds it via _embed_basket_chunk, writes that chunk's embeddings to
    its own parquet file under output_dir, marks the chunk complete, and
    repeats until nothing is claimable. Call merge_inference_output() once
    every chunk shows 'complete' to produce the final combined embeddings
    file.

    NOTE: this claim loop is only safe for a single process claiming chunks
    sequentially — DuckDB has no row-level locking equivalent to Postgres's
    FOR UPDATE SKIP LOCKED, so concurrent callers could race on the same
    chunk. See the note in basket_store.py's claim_next_chunk().
    """
    import socket
    import pandas as pd
    import basket_store

    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    output_dir = output_dir or OUTPUT_DIR
    chunk_size = chunk_size if chunk_size is not None else config.INFERENCE_CHUNK_BASKETS
    batch_size = batch_size if batch_size is not None else config.INFERENCE_BATCH_SIZE
    stale_after_seconds = (stale_after_seconds if stale_after_seconds is not None
                           else config.CHUNK_STALE_AFTER_SECONDS)
    os.makedirs(output_dir, exist_ok=True)
    basket_store.ensure_inference_chunk_plan(con, dataset_tag, chunk_size)

    n_done = 0
    while True:
        claimed = basket_store.claim_next_chunk(con, dataset_tag, worker_id, stale_after_seconds)
        if claimed is None:
            break

        chunk_df = basket_store.get_basket_range(con, dataset_tag, claimed["seq_lo"], claimed["seq_hi"])
        basket_ids, z_array = _embed_basket_chunk(chunk_df, G, model, device, batch_size=batch_size)

        # Namespaced by dataset_tag — training ("train") and scoring
        # ("score") runs share this same output_dir, and without the tag in
        # the filename their chunk_id numbering would collide and corrupt
        # each other's merge.
        chunk_path = os.path.join(output_dir, f"embeddings_chunk_{dataset_tag}_{claimed['chunk_id']}.parquet")
        pd.DataFrame({"basket_id": basket_ids, "gnn_embedding": list(z_array)}).to_parquet(chunk_path, index=False)

        basket_store.mark_chunk_complete(con, dataset_tag, claimed["chunk_id"])
        n_done += 1
        print(f"  chunk {claimed['chunk_id']} complete ({len(chunk_df):,} baskets) "
              f"— {n_done} chunks finished by worker {worker_id!r} this run")

        del chunk_df, basket_ids, z_array
        gc.collect()

    print(f"run_inference: no more claimable chunks for dataset_tag={dataset_tag!r} "
          f"(worker {worker_id!r} finished this run — {n_done} chunks completed by it)")


def merge_inference_output(dataset_tag, output_dir=None, final_path=None):
    """
    Concatenates every embeddings_chunk_{dataset_tag}_*.parquet file in
    output_dir into one final embeddings parquet, once every chunk shows
    'complete'. The final result (n_baskets x out_dim floats — a few GB even
    at 20M+ baskets) is small enough to hold as one DataFrame; only the much
    larger per-basket GRAPH objects were ever the thing that had to stay
    chunked.
    """
    import glob
    import pandas as pd

    output_dir = output_dir or OUTPUT_DIR
    final_path = final_path or os.path.join(output_dir, "basket_gnn_embeddings.parquet")
    chunk_paths = sorted(glob.glob(os.path.join(output_dir, f"embeddings_chunk_{dataset_tag}_*.parquet")))
    if not chunk_paths:
        raise FileNotFoundError(f"No embeddings_chunk_{dataset_tag}_*.parquet files found under "
                                 f"{output_dir} to merge.")

    merged = pd.concat([pd.read_parquet(p) for p in chunk_paths], ignore_index=True)
    merged.to_parquet(final_path, index=False)
    print(f"Merged {len(chunk_paths)} chunk files ({len(merged):,} baskets total) into {final_path}")
    return merged