"""
cluster_basket_embeddings.py

Stage 2 of the end-to-end pipeline: takes the basket-level embeddings
produced by GNN_Train.train_and_embed() (Stage 1) and clusters them directly
into need-states.

Reuses the same mutual-kNN + Leiden approach as multiview_clustering_v5.py's
View 2 / Leiden logic, but applied at BASKET grain (the GNN embedding vector)
instead of ITEM grain (text + price embedding). There's no second
"co-purchase" view here the way multiview_clustering_v5.py fuses two views:
the GNN embedding already encodes co-purchase structure internally, since it
was trained as a graph autoencoder over co-purchase-weighted basket graphs.
If you want to add a second view later (e.g. basket recency/spend
similarity), fuse it in with the same pattern as multiview_clustering_v5.py's
fuse_views().

This file also has a GMM path (cluster_basket_embeddings_gmm, select_k_via_bic,
compare_leiden_gmm), since need-states here are decided by comparing/
combining Leiden and GMM rather than picking one upfront. The GMM settings
below (K range, covariance type) are reasonable defaults, NOT a copy of your
team's real GMM_smoothening.py / best_k_value.py logic — swap them in once
you share those files.
"""

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    from pynndescent import NNDescent
    HAVE_PYNNDESCENT = True
except ImportError:
    HAVE_PYNNDESCENT = False
    from sklearn.neighbors import NearestNeighbors

import igraph as ig
import leidenalg
import joblib
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASKET_KNN_K       = config.BASKET_KNN_K   # neighbors per basket before mutual-kNN filtering
USE_MUTUAL_KNN     = config.USE_MUTUAL_KNN # an edge only counts if BOTH baskets rank each other in top-K
LEIDEN_RESOLUTION  = config.LEIDEN_RESOLUTION  # starting point — sweep before trusting, see sweep_resolution()
LEIDEN_N_ITERATIONS = config.LEIDEN_N_ITERATIONS  # leidenalg's own default is 2
SAMPLE_SEED        = config.SEED

# GMM defaults — placeholders, see module docstring above
GMM_K_MIN          = config.GMM_K_MIN
GMM_K_MAX          = config.GMM_K_MAX
GMM_K_STEP         = config.GMM_K_STEP
GMM_N_INIT         = config.GMM_N_INIT
GMM_COVARIANCE     = config.GMM_COVARIANCE   # "diag" scales to more dimensions than "full"
GMM_INIT_PARAMS    = config.GMM_INIT_PARAMS  # "kmeans" fits a whole k-means before EM starts
GMM_MAX_ITER       = config.GMM_MAX_ITER
GMM_REG_COVAR      = config.GMM_REG_COVAR    # floor on covariance diagonals

# All pipeline-produced artifacts land here, not the working directory.
OUTPUT_DIR         = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

GMM_MODEL_PATH     = os.path.join(OUTPUT_DIR, "gmm_basket_model.pkl")   # fitted model, reloaded by score_new_baskets.py

PROGRESS_HEARTBEAT_SECS = config.PROGRESS_HEARTBEAT_SECS

CACHE_BASKET_EDGES = config.CACHE_BASKET_EDGES


def _edge_cache_path(k: int, use_mutual: bool) -> str:
    """
    One cache file per (k, mutual) setting, rather than one fixed filename.

    The fingerprint manifest already REFUSES a mismatched graph, but refusing
    happens after the fact — a run at different settings had already
    overwritten the file by then. That cost the same 35-minute rebuild twice:
    a `pipeline_main` run picked up k=15/mutual=True from a reverted .env,
    rebuilt over a k=10 one-directional graph, and only then reported the
    mismatch it had just caused.

    Encoding the settings in the name makes that impossible. A wrong-config
    run builds a DIFFERENT file, leaves yours alone, and flipping between two
    configurations costs one build each instead of one build per flip.

    Only k and mutual are in the name. The other fingerprint dimensions
    (seed, basket count, embedding digest, backend) still invalidate via the
    manifest — correctly, since a graph built from retrained embeddings is
    not something you want to keep alongside the new one.
    """
    return os.path.join(
        OUTPUT_DIR,
        f"basket_knn_edges_k{int(k)}_{'mutual' if use_mutual else 'onedir'}.parquet",
    )


BASKET_EDGES_PATH  = _edge_cache_path(BASKET_KNN_K, USE_MUTUAL_KNN)

MIN_GRAPH_COVERAGE = config.MIN_GRAPH_COVERAGE

# need_state_cluster for a basket that never entered the graph. Leiden cannot
# place a vertex it never saw, so these are not "cluster 0" and not a missing
# value to be imputed — they are baskets the mutual-kNN filter left with no
# edges. An explicit sentinel keeps them countable downstream; NaN from an
# outer merge is what previously hid 28.4M of them.
UNCLUSTERED = -1

# Rows per predict_proba call when collapsing posteriors to a confidence
# column — bounds a n_baskets x n_components intermediate. Matches the batch
# size need_state_graph.build_gmm_overlap uses for the same call.
GMM_PREDICT_CHUNK  = 200_000


# ─────────────────────────────────────────────
# Progress reporting
# ─────────────────────────────────────────────
#
# Stage 2 is the one part of this pipeline that can run for hours with no
# output whatsoever. Stage 1 has a tqdm bar per basket chunk because it loops
# in Python over a known number of baskets; Stage 2's expensive parts are
# single opaque calls (the kNN search, the igraph build, each Leiden optimiser
# iteration, each GMM EM iteration) with no countable Python loop, so at 57M
# baskets a healthy run 40 minutes into Leiden and a wedged run look exactly
# the same from the terminal.
#
# A true percentage bar is therefore impossible for those calls — they expose
# no progress callback to hook. What this provides instead:
#
#   * every step announces itself before it starts and reports how long it
#     took when it ends, so "which step am I in" is always answerable;
#   * steps longer than PROGRESS_HEARTBEAT_SECS emit a "still running, N
#     elapsed" line while they run;
#   * Leiden and GMM report per ITERATION rather than only at the very end
#     (see run_leiden_on_basket_graph and cluster_basket_embeddings_gmm), which
#     is what makes the total runnable-time estimable from the first iteration.
#
# Heartbeat caveat, deliberately not hidden: it runs on a background thread, so
# it only prints while the working call has released the GIL. numpy and
# pynndescent/numba do release it; a C extension that holds it will stay silent
# until it returns. The step start/end timings and the per-iteration reporting
# do not depend on the GIL and are the load-bearing part of this.


def fmt_duration(seconds: float) -> str:
    """0 -> '0s', 95 -> '1m 35s', 3725 -> '1h 02m 05s'."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@contextmanager
def progress_step(label, index=None, total=None, heartbeat_secs=PROGRESS_HEARTBEAT_SECS):
    """
    Wraps one long step: announce it, heartbeat while it runs, time it.

        with progress_step("building igraph", 4, 5):
            ...

    prints

        [4/5] building igraph ...
              [4/5] building igraph: still running, 30s elapsed
        [4/5] building igraph: done in 47s
    """
    tag = f"[{index}/{total}] " if index is not None and total is not None else ""
    print(f"  {tag}{label} ...", flush=True)

    finished = threading.Event()
    started = time.perf_counter()

    def _heartbeat():
        while not finished.wait(heartbeat_secs):
            elapsed = fmt_duration(time.perf_counter() - started)
            print(f"        {tag}{label}: still running, {elapsed} elapsed", flush=True)

    beat = threading.Thread(target=_heartbeat, daemon=True)
    beat.start()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        finished.set()
        beat.join(timeout=1.0)
        elapsed = fmt_duration(time.perf_counter() - started)
        # A step that raised must not print a success-shaped line. This helper
        # exists to tell a hang from a crash; reporting "done in 4m" for a
        # MemoryError would defeat exactly that.
        outcome = "done in" if succeeded else "FAILED after"
        print(f"  {tag}{label}: {outcome} {elapsed}", flush=True)


# ─────────────────────────────────────────────
# kNN edge-list cache
# ─────────────────────────────────────────────
#
# The kNN graph is by far the most expensive thing Stage 2 produces — an
# approximate-NN search over the whole basket population — and it was the only
# major artifact the pipeline never wrote down. Everything upstream of it is
# already resumable (the co-purchase matrix has a fingerprinted .npz, the
# trained model has a manifest, inference has a DuckDB chunk queue), so a
# Stage 2 that died in Leiden or GMM still threw away hours of neighbour search
# and started that part over. This closes that gap.
#
# Fingerprinted rather than "file exists", for the reason the rest of this
# pipeline is: silently reusing an edge list built from different embeddings
# would cluster the old geometry while reporting the new run's settings.


def _cache_manifest_path(cache_path: str) -> str:
    return os.path.splitext(cache_path)[0] + ".manifest.json"


def _embedding_digest(basket_gnn_embeddings: pd.DataFrame, sample_rows: int = 4096) -> str:
    """
    Cheap deterministic identity for an embedding table.

    Hashes an evenly-spaced sample of rows, not all 57M of them: enough to
    notice a retrained model or a different basket population without a
    multi-GB pass. Row count alone would NOT be enough — retraining the GNN
    produces entirely different vectors for exactly the same baskets, and
    that is precisely the case where reusing a cached edge list would be
    wrong and invisible.
    """
    n = len(basket_gnn_embeddings)
    positions = np.linspace(0, n - 1, num=min(sample_rows, n), dtype=np.int64)
    sample = basket_gnn_embeddings.iloc[positions]

    digest = hashlib.sha256()
    # Cast to a single dtype so the digest describes the VALUES rather than
    # how they happened to be stored. Note this only removes the difference in
    # the widening direction: genuinely re-rounding f64 values to f32 does
    # change the digest, which errs toward a needless rebuild rather than a
    # wrong reuse — the safe direction.
    vectors = np.stack(sample["gnn_embedding"].values).astype(np.float64)
    digest.update(np.ascontiguousarray(vectors).tobytes())
    digest.update("|".join(map(str, sample["basket_id"].tolist())).encode("utf-8"))
    return digest.hexdigest()[:16]


def _edge_cache_manifest(basket_gnn_embeddings: pd.DataFrame, k: int, use_mutual: bool) -> dict:
    """Everything that changes which edges come out of build_basket_knn_graph()."""
    return {
        "k": int(k),
        "use_mutual": bool(use_mutual),
        "seed": int(SAMPLE_SEED),
        "n_baskets": int(len(basket_gnn_embeddings)),
        "embedding_digest": _embedding_digest(basket_gnn_embeddings),
        # pynndescent is approximate and sklearn is exact — same k, different graph.
        "knn_backend": "pynndescent" if HAVE_PYNNDESCENT else "sklearn",
    }


def _read_manifest(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _atomic_replace(src: str, dst: str, attempts: int = 6, delay: float = 0.25):
    """
    os.replace, retried briefly on a Windows sharing violation.

    On Windows a replace raises PermissionError (WinError 5) when anything
    else holds a handle to either path. An antivirus scanner or the search
    indexer opening a file microseconds after it is written is the usual
    cause, it is timing-dependent, and it clears on its own within a moment.
    POSIX does not behave this way, so the retry costs nothing there.

    Worth having because the alternative is losing a 35-minute edge build at
    the very last step — the parquet is written, and only the rename fails.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _write_edge_cache(edges: pd.DataFrame, cache_path: str, manifest: dict):
    """
    Parquet first, manifest second, each replaced atomically. A crash between
    the two leaves an edge list with no manifest — unusable, which is the safe
    direction. The reverse order could leave a manifest vouching for a
    half-written parquet.
    """
    tmp_edges = cache_path + ".tmp"
    edges.to_parquet(tmp_edges, index=False)
    _atomic_replace(tmp_edges, cache_path)

    manifest_path = _cache_manifest_path(cache_path)
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    _atomic_replace(tmp_manifest, manifest_path)


# ─────────────────────────────────────────────
# kNN graph over basket embeddings
# ─────────────────────────────────────────────

def build_basket_knn_graph(
    basket_gnn_embeddings: pd.DataFrame,
    k: int = BASKET_KNN_K,
    use_mutual: bool = USE_MUTUAL_KNN,
    use_cache: bool = CACHE_BASKET_EDGES,
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    basket_gnn_embeddings : DataFrame with columns 'basket_id', 'gnn_embedding'
        (gnn_embedding is a fixed-length float vector per row — the output of
        GNN_Train.train_and_embed()).
    use_cache : reuse (and write) a fingerprinted edge list at cache_path.
        Only reused when k, use_mutual, seed, basket count, kNN backend and a
        digest of the embeddings all match — see _edge_cache_manifest.

    Returns
    -------
    DataFrame with columns: basket_a, basket_b, weight

    Edge construction is fully vectorized (NumPy arrays + a sorted-array
    lookup for the mutual-kNN membership check) rather than a per-basket-pair
    Python loop. At real data scale (tens of millions of baskets x k
    neighbors), the previous version accumulated results into a plain Python
    dict keyed by (i, j) tuples — hundreds of millions of entries, each
    costing far more memory than its two ints + one float actually need, and
    a pure-Python loop over that many iterations is also far too slow to
    finish in practical time at this N. Neither of those costs depend on the
    math changing here — same neighbors, same similarities, same mutual-kNN
    rule, just computed over arrays instead of one basket-pair at a time.
    """
    KNN_STEPS = 4

    # Derived from the k/use_mutual ACTUALLY in use, not from a default bound
    # at import. A caller passing k=30 to a function whose cache_path default
    # was fixed at config's k would read and overwrite the wrong file — the
    # precise failure _edge_cache_path exists to prevent.
    if cache_path is None:
        cache_path = _edge_cache_path(k, use_mutual)

    manifest = _edge_cache_manifest(basket_gnn_embeddings, k, use_mutual) if use_cache else None
    if use_cache and os.path.exists(cache_path):
        if _read_manifest(_cache_manifest_path(cache_path)) == manifest:
            with progress_step(f"loading cached kNN edges from {cache_path}"):
                cached = pd.read_parquet(cache_path)
            print(f"  REUSING cached basket kNN graph: {len(cached):,} edges "
                  f"(k={k}, mutual={use_mutual}) — skipping the neighbour search. "
                  f"Delete {cache_path} to force a rebuild.")
            return cached
        print(f"  IGNORING {cache_path}: its fingerprint doesn't match this run "
              f"(k, mutual-kNN, seed, basket count, kNN backend or the embeddings "
              f"themselves changed) — rebuilding rather than clustering a stale graph.")

    # .to_numpy() rather than .tolist(): these ids are only ever used by
    # position, as a numpy array, and at 57M baskets the list round-trip
    # materialises 57M boxed Python objects for no reason.
    basket_id_arr = basket_gnn_embeddings["basket_id"].to_numpy()
    n = len(basket_id_arr)

    with progress_step(f"normalising {n:,} embeddings", 1, KNN_STEPS):
        # copy=False normalises in place. Inference writes embeddings as
        # float32 (GraphBuilder._embed_basket_chunk), so at 57M x 64 this
        # array is ~13.6GB and sklearn's default copy=True would hold two of
        # them at once — that pair IS the "~29GB peak" the project notes call
        # the reason Stage 2 may not fit. np.stack() above has just built a
        # fresh array that nothing else references, so normalising it in place
        # is safe: there is no caller-visible array to preserve.
        X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values), copy=False)

    if HAVE_PYNNDESCENT:
        print(f"  kNN method: pynndescent (approximate) — k={k}, n={n:,} baskets")
        # verbose=True is pynndescent's own internal progress output (RP-forest
        # construction, then each NN-descent iteration). It is the only
        # inside-the-call visibility available for what is normally Stage 2's
        # single longest step.
        with progress_step(f"kNN search, k={k} over {n:,} baskets", 2, KNN_STEPS):
            index = NNDescent(X, n_neighbors=k + 1, metric="cosine",
                              random_state=SAMPLE_SEED, verbose=True)
            indices, distances = index.neighbor_graph
    else:
        fallback_msg = (
            f"kNN method: sklearn NearestNeighbors (EXACT/brute-force under cosine metric) — "
            f"k={k}, n={n:,} baskets. pynndescent not installed — at real data scale (millions "
            f"of baskets) this is computationally infeasible, not just slow. Install pynndescent "
            f"(`pip install pynndescent`) before running this at full scale."
        )
        if n > 100_000:
            print("=" * 70)
            print(f"  !!! WARNING !!!  {fallback_msg}")
            print(f"  n={n:,} is large enough that this will likely never finish in "
                  f"practical time — this is not a slow-but-safe fallback at this scale.")
            print("=" * 70)
        else:
            print(f"  kNN method: {fallback_msg}")
        with progress_step(f"exact kNN search, k={k} over {n:,} baskets", 2, KNN_STEPS):
            nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
            nn.fit(X)
            distances, indices = nn.kneighbors(X)

    with progress_step(f"selecting {'mutual-' if use_mutual else ''}kNN edges", 3, KNN_STEPS):
        # Drop each row's own self-match (its nearest "neighbor" is itself, at
        # column 0), keep the k candidate neighbor columns, and flatten into one
        # (src, dst, similarity) edge per directed pair — same neighbors the old
        # per-basket loop iterated, just as arrays instead of a Python loop body.
        neighbor_idx  = indices[:, 1:k + 1].astype(np.int64)
        neighbor_dist = distances[:, 1:k + 1]

        src = np.repeat(np.arange(n, dtype=np.int64), neighbor_idx.shape[1])
        dst = neighbor_idx.reshape(-1)
        sim = (1 - neighbor_dist).reshape(-1).astype(np.float64)

        # Drop any self-match that slipped in at a position other than 0
        # (e.g. duplicate embeddings) — same as the old loop's `j != i` — and
        # any slot the approximate search could not fill.
        #
        # The unfilled slots matter more than they look. pynndescent pads a
        # row it cannot complete with an out-of-range index and an infinite
        # distance; that is what its "Failed to correctly find n_neighbors for
        # some samples" warning means, and it fires on every full-scale run
        # here. An out-of-range dst makes the key below alias a DIFFERENT
        # legitimate pair: src*n + n is exactly (src+1)*n + 0, so a padded row
        # from basket src can answer the reverse-edge lookup for the real pair
        # (0, src+1). Those rows were previously removed further down by the
        # `avg_sim > 0` filter (padding has similarity -inf), which is to say
        # the aliasing was survivable only by accident. Removing them here
        # keeps the key space honest.
        in_range = (dst >= 0) & (dst < n)
        usable = (dst != src) & in_range
        n_padded = int((~in_range).sum())
        if n_padded:
            print(f"  Dropped {n_padded:,} unfilled neighbour slots "
                  f"({n_padded / len(dst):.2%} of candidates) — the approximate "
                  f"search could not find {k} neighbours for every basket")
        src, dst, sim = src[usable], dst[usable], sim[usable]

        if use_mutual:
            # Mutual-kNN check: "does the reverse edge (j, i) also exist among
            # the edges above" — done via one integer key per directed pair and
            # a sorted search, instead of hundreds of millions of dict lookups.
            #
            # Computed inside this branch, not before it. The one-directional
            # path below never reads rev_exists or rev_sim, and at full
            # population without the mutual filter there are ~857M directed
            # pairs: the argsort alone allocates a second int64 array that size
            # and the surrounding key/position arrays several more. That was
            # tens of GB and a long sort spent on values nothing would consume.
            keys = src * n + dst
            order = np.argsort(keys)
            sorted_keys = keys[order]

            rev_keys   = dst * n + src
            pos        = np.clip(np.searchsorted(sorted_keys, rev_keys), 0, len(sorted_keys) - 1)
            rev_exists = sorted_keys[pos] == rev_keys
            rev_sim    = sim[order[pos]]
            del keys, order, sorted_keys, rev_keys, pos

            # Reverse direction must exist (mutual), and each undirected pair is
            # kept exactly once via src < dst — same as the old `i < j` check.
            avg_sim = (sim + rev_sim) / 2.0
            keep = rev_exists & (src < dst) & (avg_sim > 0)
            edge_a, edge_b, edge_sim = src[keep], dst[keep], avg_sim[keep]
        else:
            keep = sim > 0
            edge_a = np.minimum(src[keep], dst[keep])
            edge_b = np.maximum(src[keep], dst[keep])
            edge_sim = sim[keep]

    with progress_step(f"assembling edge table ({len(edge_a):,} candidate edges)", 4, KNN_STEPS):
        # Deduplicate on the integer vertex indices rather than on the string
        # ids. basket_id is the basket grain key — one row per basket, so
        # index -> id is 1:1 and the two give the same result — but
        # drop_duplicates over two object columns hashes ~193M Python strings,
        # where one int64 key per edge is hashed in C. Filtering first also
        # means the id columns are only materialised for surviving edges.
        # Same int64 key trick as the mutual check above: safe because
        # (n-1)*n + (n-1) is ~3.3e15 against int64's ~9.2e18.
        pair_key = edge_a.astype(np.int64, copy=False) * n + edge_b
        _, first_seen = np.unique(pair_key, return_index=True)
        # np.unique orders by key; sorting the positions restores original row
        # order, which is what drop_duplicates(keep="first") preserved.
        first_seen.sort()
        del pair_key

        kept_a, kept_b = edge_a[first_seen], edge_b[first_seen]

        # Counted here, while the endpoints are still integer indices: a
        # one-byte-per-basket mask is ~57MB at full scale, where uniquing the
        # 193M string ids after the fact is the expensive thing this whole
        # function already goes out of its way to avoid.
        in_graph = np.zeros(n, dtype=bool)
        in_graph[kept_a] = True
        in_graph[kept_b] = True
        n_in_graph = int(in_graph.sum())
        del in_graph

        edges = pd.DataFrame({
            "basket_a": basket_id_arr[kept_a],
            "basket_b": basket_id_arr[kept_b],
            "similarity": edge_sim[first_seen],
        })
        edges["weight"] = minmax_scale(edges["similarity"])

    # Visible before _build_igraph/Leiden ever touch this — the kNN graph
    # itself (not just building it) is the next likely memory bottleneck at
    # full basket-population scale (potentially 1B+ edges), flagged here
    # rather than silently discovered when igraph/leidenalg choke on it.
    # Estimated from a sample, not measured with memory_usage(deep=True) over
    # the whole frame: deep=True walks every Python string in both id columns
    # via sys.getsizeof, which at ~96M rows is a couple of hundred million
    # interpreter-level calls spent on a single log line.
    sample = edges.head(10_000)
    bytes_per_row = sample.memory_usage(deep=True).sum() / max(len(sample), 1)
    est_mb = bytes_per_row * len(edges) / 1e6
    print(f"Basket kNN graph ({'mutual' if use_mutual else 'one-directional'}): "
          f"{len(edges):,} edges over {n:,} baskets "
          f"(~{est_mb:,.0f} MB as a DataFrame, before igraph's own edge-list overhead)")

    # Coverage, not just edge count. A basket with no surviving edge is not a
    # vertex at all (the vertex set is derived from the edge endpoints), so it
    # gets no community — Leiden has nothing to place. Printed on every build
    # because this was invisible: a k=15 mutual run over 57.1M baskets built a
    # 28.7M-vertex graph and nothing in the log said the other half was gone.
    isolated = n - n_in_graph
    print(f"  Coverage: {n_in_graph:,} of {n:,} baskets have at least one edge "
          f"({n_in_graph / n:.1%})")
    if isolated:
        print(f"  {isolated:,} baskets ({isolated / n:.1%}) have NO edge and cannot be "
              f"assigned a Leiden community. Raise PIPELINE_BASKET_KNN_K or set "
              f"PIPELINE_USE_MUTUAL_KNN=false to cover more of the population; "
              f"cluster_basket_embeddings.coverage_probe() measures the trade-off "
              f"without running Leiden.")

    result = edges[["basket_a", "basket_b", "weight"]]

    if use_cache:
        # Written before Leiden and GMM run, precisely because those are the
        # two steps expected to fail for memory at full scale — the whole
        # point is that the next attempt does not repeat the neighbour search.
        with progress_step(f"caching edge list to {cache_path}"):
            _write_edge_cache(result, cache_path, manifest)
        print(f"  Cached {len(result):,} edges — a rerun will reuse this instead of "
              f"redoing the kNN search.")

    return result


# ─────────────────────────────────────────────
# Leiden
# ─────────────────────────────────────────────

def _build_igraph(edges: pd.DataFrame):
    """
    Map basket ids onto contiguous 0..n-1 vertex indices and build the graph.

    Vectorised deliberately, and it is the same argument build_basket_knn_graph
    makes above. The obvious way to write this — sorted(set(a) | set(b)), a
    {basket_id: index} dict, then a list comprehension of (i, j) tuples — is a
    pure-Python pass over every vertex and every edge, and it lands at the
    worst possible point in the run: at ~57M vertices / ~96M edges it
    materialises a 57M-entry dict plus ~96M two-tuples, each tuple and each
    boxed index a separate object, which is tens of GB of interpreter overhead
    on top of the ~6GB edge table, on the machine least able to afford it. It
    also sits between two prints with nothing in between, so a run that dies
    here is indistinguishable from a run that hung inside Leiden.

    pd.factorize(..., sort=True) produces the identical vertex ordering that
    sorted(set(...)) does and the identical index for each vertex, over arrays
    instead of Python objects. Concatenating both endpoint columns and
    factorizing once (rather than per column) is what guarantees both endpoints
    resolve against the same index.
    """
    n_edges = len(edges)

    with progress_step(f"indexing vertices across {n_edges:,} edges", 1, 2):
        codes, baskets = pd.factorize(
            pd.concat([edges["basket_a"], edges["basket_b"]], ignore_index=True),
            sort=True,
        )
        # factorize codes nulls as -1 and leaves them out of `baskets`, where
        # the sorted(set(...)) build this replaced would have kept them as real
        # vertices. igraph would read -1 as "last vertex" or reject it, so a
        # null id has to stop the run rather than quietly re-point an edge.
        if len(codes) and codes.min() < 0:
            raise ValueError(
                f"{int((codes < 0).sum()):,} null basket id(s) in the edge list — "
                f"these cannot be graph vertices. Fix them upstream rather than "
                f"silently dropping or remapping the affected edges."
            )
        edge_array = np.column_stack([codes[:n_edges], codes[n_edges:]])

    with progress_step(f"building igraph ({len(baskets):,} vertices, {n_edges:,} edges)", 2, 2):
        g = ig.Graph(n=len(baskets))
        g.add_edges(edge_array)
        g.es["weight"] = edges["weight"].to_numpy()

    return g, baskets


def _mutual_coverage_mask(indices: np.ndarray, distances: np.ndarray, k: int) -> np.ndarray:
    """
    Boolean mask: which rows keep at least one MUTUAL neighbour among their
    first k, i.e. which baskets would still be vertices under mutual-kNN.

    `indices` / `distances` are pynndescent's or sklearn's neighbour tables
    INCLUDING each row's self-match at column 0, exactly as
    build_basket_knn_graph receives them.

    NOT every entry in `indices` is a vertex. When the approximate search
    cannot fill a row — the "Failed to correctly find n_neighbors for some
    samples" warning pynndescent emits at this scale — it pads the row with
    out-of-range sentinels and infinite distances. Following one of those as
    an index raises IndexError (observed: index 57,115,804 in a table of
    57,115,804 rows), so they are masked out here rather than assumed away.

    Column-at-a-time rather than the sorted-key join build_basket_knn_graph
    uses, because this only needs the vertex set and not the edges. Each
    iteration gathers one (n, k) table, so peak memory is ~n*k*itemsize rather
    than the several multiples of n*k the key sort needs — that difference is
    what makes probing k=50 over 57M baskets affordable.

    Remaining difference from production, small and deliberate: the real rule
    keeps a pair when the AVERAGE of the two directions' similarities is > 0,
    where this requires each direction to be > 0 on its own. They differ only
    when one direction is positive and the other negative by more — which the
    sentinel handling above already covers, since those are -inf.
    """
    n = indices.shape[0]
    neighbours = indices[:, 1:k + 1]
    similarities = 1.0 - distances[:, 1:k + 1]

    # Compared in the neighbour table's own dtype (int32 from pynndescent)
    # rather than promoting to int64: the gather below is the single largest
    # allocation in this function and widening it would double that for
    # nothing. Basket counts are nowhere near int32's 2.1B ceiling.
    self_id = np.arange(n, dtype=indices.dtype)[:, None]

    valid = (neighbours >= 0) & (neighbours < n) & (similarities > 0)
    # Invalid slots still have to point somewhere for the gather; row 0 is
    # arbitrary and safe because `valid` masks the result out afterwards.
    # Copying preserves the int32 dtype that np.where would widen.
    safe = neighbours.copy()
    safe[~valid] = 0

    has_mutual = np.zeros(n, dtype=bool)
    for column in range(neighbours.shape[1]):
        j = safe[:, column]
        # j's own neighbour list, gathered per row; does it list me back, in a
        # slot that is itself a real neighbour rather than padding?
        listed_back = ((safe[j] == self_id) & valid[j]).any(axis=1)
        # A self-match sitting at a column other than 0 would otherwise count
        # as its own mutual neighbour — build_basket_knn_graph drops those
        # too (the `dst != src` filter).
        has_mutual |= listed_back & valid[:, column] & (j != self_id[:, 0])
    return has_mutual


def coverage_probe(
    basket_gnn_embeddings: pd.DataFrame,
    k_values=(10, 15, 20, 30, 50),
) -> pd.DataFrame:
    """
    How much of the basket population survives mutual-kNN, as a function of k.

    Answers "which k do I need" with ONE neighbour search instead of one full
    Stage 2 per candidate: the search runs at max(k_values), and every smaller
    k is evaluated as a prefix of the same neighbour table. At full scale that
    is ~30 minutes total rather than ~30 minutes per setting, and it never
    builds an edge table, writes a cache, or runs Leiden.

    Only the mutual case is swept. With use_mutual=False every basket emits k
    directed edges by construction, so coverage is 100% by definition and
    there is nothing to measure — the question there is edge count and Leiden
    runtime, not coverage.

    Returns
    -------
    DataFrame: k, n_covered, n_isolated, coverage
    """
    k_values = sorted(set(int(k) for k in k_values))
    k_max = k_values[-1]

    n = len(basket_gnn_embeddings)
    with progress_step(f"normalising {n:,} embeddings", 1, 3):
        X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values), copy=False)

    with progress_step(f"kNN search, k={k_max} over {n:,} baskets", 2, 3):
        if HAVE_PYNNDESCENT:
            index = NNDescent(X, n_neighbors=k_max + 1, metric="cosine",
                              random_state=SAMPLE_SEED, verbose=True)
            indices, distances = index.neighbor_graph
        else:
            nn = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine")
            nn.fit(X)
            distances, indices = nn.kneighbors(X)
    del X

    rows = []
    with progress_step(f"evaluating coverage at k={k_values}", 3, 3):
        for k in k_values:
            covered = int(_mutual_coverage_mask(indices, distances, k).sum())
            rows.append({
                "k": k,
                "n_covered": covered,
                "n_isolated": n - covered,
                "coverage": covered / n if n else 1.0,
            })
            print(f"    k={k:>3}: {covered:,} of {n:,} covered "
                  f"({covered / n:.1%}), {n - covered:,} isolated")

    result = pd.DataFrame(rows)
    print()
    # Not exact, and not a bound in either direction. Every k here is read off
    # ONE search configured for k_max neighbours, but production runs its own
    # search at its own k — and an approximate index explores differently when
    # asked for 51 neighbours than when asked for 16, so the first 15 columns
    # of a k=50 search are not the 15 columns a k=15 search would return.
    # Measured gap: this reported 41.0% at k=15 where a real k=15 build
    # covered 50.2%. Trust the SHAPE of the curve (where the returns flatten),
    # not the absolute numbers.
    print("Mutual-kNN coverage by k (approximate — each k is read off a single "
          f"k={k_max} search, not its own):")
    print(result.to_string(index=False))
    print(f"\nPIPELINE_MIN_GRAPH_COVERAGE is currently {MIN_GRAPH_COVERAGE:.2f}. "
          f"If no k here clears it, use PIPELINE_USE_MUTUAL_KNN=false — that "
          f"covers every basket by construction, at the cost of a much denser "
          f"graph and a slower Leiden.")
    return result


def diagnose_connectivity(edges: pd.DataFrame) -> pd.Series:
    """Same check as multiview_clustering_v5.py — run before trusting a resolution sweep."""
    g, baskets = _build_igraph(edges)
    with progress_step(f"finding connected components over {g.vcount():,} vertices"):
        components = g.connected_components(mode="weak")
        sizes = pd.Series([len(c) for c in components])

    print(f"Connected components: {len(sizes)}")
    print(sizes.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))
    print(f"Largest component: {sizes.max()} baskets ({sizes.max() / len(baskets):.1%} of graph)")
    return sizes


def sweep_resolution(edges: pd.DataFrame, resolutions=(0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """
    Builds the graph once, reports n_clusters/modularity/size distribution
    across resolutions — does NOT auto-pick. Read the table, set
    LEIDEN_RESOLUTION, then call cluster_basket_embeddings() for the final run.
    """
    g, baskets = _build_igraph(edges)

    results = []
    # No tqdm bar here on purpose: the per-resolution summary line below is
    # already the progress report, and a bar being redrawn underneath printed
    # lines just shreds both.
    for i, r in enumerate(resolutions, start=1):
        print(f"  resolution {i}/{len(resolutions)}: {r:.2f}")
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=r, seed=SAMPLE_SEED,
        )
        sizes = pd.Series(partition.membership).value_counts()
        modularity = g.modularity(partition.membership, weights="weight")
        results.append({
            "resolution": r, "n_clusters": len(sizes), "modularity": modularity,
            "min_size": sizes.min(), "median_size": sizes.median(), "max_size": sizes.max(),
        })
        print(f"resolution={r:.2f}  n_clusters={len(sizes)}  modularity={modularity:.4f}  "
              f"size min/median/max={sizes.min()}/{sizes.median():.0f}/{sizes.max()}")

    return pd.DataFrame(results)


def _check_graph_coverage(baskets, all_basket_ids, min_coverage: float):
    """
    Refuse to spend hours of Leiden on a graph that is missing most of its
    baskets. Returns the de-duplicated universe of basket ids.

    Checked BEFORE the optimiser starts, deliberately. The failure this exists
    for cost a full run: mutual-kNN at k=15 over 57,115,804 baskets produced a
    28,692,907-vertex graph, Leiden ran for hours on that half, and the
    shortfall only became visible as NaN in the output parquet — by which
    point the cheap fix (raise k, or drop the mutual filter, then redo a
    ~30-minute neighbour search) had cost a day.
    """
    universe = pd.Index(pd.unique(np.asarray(all_basket_ids)))
    n_total = len(universe)
    n_in_graph = len(baskets)
    coverage = n_in_graph / n_total if n_total else 1.0

    print(f"  Graph coverage: {n_in_graph:,} of {n_total:,} baskets "
          f"({coverage:.1%}), {n_total - n_in_graph:,} with no edge")

    if coverage < min_coverage:
        raise ValueError(
            f"kNN graph covers only {coverage:.1%} of baskets "
            f"({n_in_graph:,} of {n_total:,}) — below PIPELINE_MIN_GRAPH_COVERAGE="
            f"{min_coverage:.2f}. The {n_total - n_in_graph:,} baskets with no edge "
            f"cannot be given a Leiden community, so clustering now would silently "
            f"produce labels for a subset of the population.\n"
            f"  Fix the graph:  raise PIPELINE_BASKET_KNN_K, or set "
            f"PIPELINE_USE_MUTUAL_KNN=false\n"
            f"  Measure first:  cluster_basket_embeddings.coverage_probe() reports "
            f"coverage per k without running Leiden\n"
            f"  Accept it:      set PIPELINE_MIN_GRAPH_COVERAGE=0 to cluster the "
            f"covered subset anyway (uncovered baskets get need_state_cluster="
            f"{UNCLUSTERED})"
        )
    return universe


def run_leiden_on_basket_graph(
    edges: pd.DataFrame,
    resolution: float = LEIDEN_RESOLUTION,
    n_iterations: int = LEIDEN_N_ITERATIONS,
    all_basket_ids=None,
    min_coverage: float = MIN_GRAPH_COVERAGE,
) -> pd.DataFrame:
    """
    Leiden over the basket kNN graph.

    all_basket_ids : the full basket population this graph was built from.
        Pass it. Without it there is no way to tell a graph covering every
        basket from one covering half of them — the edge list alone only
        knows about baskets that survived the mutual-kNN filter. When given,
        coverage is checked against min_coverage before the optimiser runs,
        and every basket appears in the returned frame: those absent from the
        graph get need_state_cluster=-1 rather than being dropped and later
        resurfacing as NaN in an outer merge.

    Drives the optimiser one iteration at a time instead of handing the whole
    thing to leidenalg.find_partition(). This is NOT a change of algorithm:
    find_partition() is a thin wrapper that builds the partition object, seeds
    an Optimiser, and calls optimise_partition() n_iterations times — exactly
    what happens below, same partition type, same weights, same seed, same
    optimiser instance, same iteration count. The only difference is that
    control returns to Python between iterations, which is what allows each one
    to report its cluster count, quality and duration.

    That matters at full scale because one iteration over ~57M vertices is
    itself a long opaque call: knowing iteration 1 took 18 minutes is what
    makes the total estimable, and a heartbeat alone cannot tell you that (a C
    extension holding the GIL starves the heartbeat thread — see the progress
    reporting notes at the top of this module).

    Stops early if an iteration yields no improvement. That is leidenalg's own
    convergence criterion (its n_iterations=-1 mode means "until no
    improvement"); a non-improving iteration leaves the partition unchanged, so
    this returns the same labels, just without burning another full pass.
    """
    g, baskets = _build_igraph(edges)

    universe = None
    if all_basket_ids is not None:
        universe = _check_graph_coverage(baskets, all_basket_ids, min_coverage)

    partition = leidenalg.RBConfigurationVertexPartition(
        g, weights="weight", resolution_parameter=resolution,
    )
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(SAMPLE_SEED)

    print(f"  Leiden: up to {n_iterations} optimiser iteration(s) over "
          f"{g.vcount():,} vertices / {g.ecount():,} edges, resolution={resolution}")

    for iteration in range(1, n_iterations + 1):
        with progress_step(f"Leiden iteration {iteration}", iteration, n_iterations):
            improvement = optimiser.optimise_partition(partition, n_iterations=1)
        print(f"        -> {len(partition):,} clusters, "
              f"quality={partition.quality():,.4f}, improvement={improvement:,.6g}")
        if not improvement:
            print(f"        -> converged (no improvement), stopping after "
                  f"iteration {iteration} of {n_iterations}")
            break

    clustered = pd.DataFrame({"basket_id": baskets, "need_state_cluster": partition.membership})
    if universe is None:
        return clustered

    # Every basket comes back, labelled or explicitly UNCLUSTERED. Returning
    # only the clustered ones is what let the shortfall travel downstream as
    # NaN: pipeline_main outer-merges this with the GMM labels, which DO cover
    # the whole population, so the row count looked right while half the
    # Leiden column was empty.
    result = pd.DataFrame({"basket_id": universe}).merge(clustered, on="basket_id", how="left")
    n_missing = int(result["need_state_cluster"].isna().sum())
    result["need_state_cluster"] = (
        result["need_state_cluster"].fillna(UNCLUSTERED).astype(np.int64)
    )
    if n_missing:
        print(f"  {n_missing:,} baskets had no vertex in the graph — "
              f"need_state_cluster={UNCLUSTERED} (not a cluster; no edges to place them by)")
    return result


def cluster_basket_embeddings(
    basket_gnn_embeddings: pd.DataFrame,
    resolution: float = LEIDEN_RESOLUTION,
) -> pd.DataFrame:
    """Convenience wrapper: build graph -> Leiden -> per-basket need-state cluster."""
    started = time.perf_counter()
    edges = build_basket_knn_graph(basket_gnn_embeddings)
    clusters = run_leiden_on_basket_graph(
        edges,
        resolution=resolution,
        all_basket_ids=basket_gnn_embeddings["basket_id"].to_numpy(),
    )
    print(f"  Leiden clustering total: {fmt_duration(time.perf_counter() - started)}")

    real = clusters[clusters["need_state_cluster"] != UNCLUSTERED]
    print(f"\nNeed-state clusters found: {real['need_state_cluster'].nunique()}")
    print(real["need_state_cluster"].value_counts().describe())
    return clusters


# ─────────────────────────────────────────────
# Assigning NEW baskets to already-discovered clusters
# ─────────────────────────────────────────────

def assign_new_baskets_to_clusters(
    new_embeddings: pd.DataFrame,
    reference_embeddings: pd.DataFrame,
    reference_clusters: pd.DataFrame,
    k: int = None,
) -> pd.DataFrame:
    """
    Leiden is transductive — it has no native way to place a brand-new point
    into a community that was already found. The standard workaround: find
    each new basket's k nearest neighbors (cosine) among the already-clustered
    reference baskets, and assign the majority-vote cluster among them. This
    mirrors the same majority-vote principle multiview_clustering_v5.py uses
    to assign baskets to item-level clusters — just run in embedding space
    here instead of over item membership, since these are basket embeddings.

    Parameters
    ----------
    new_embeddings : DataFrame with 'basket_id', 'gnn_embedding' — baskets to score
    reference_embeddings : DataFrame with 'basket_id', 'gnn_embedding' — already-
        embedded training baskets (e.g. basket_gnn_embeddings.parquet)
    reference_clusters : DataFrame with 'basket_id', 'need_state_cluster' — output
        of cluster_basket_embeddings() for the same reference baskets
    k : neighbors to vote over

    Returns
    -------
    DataFrame: basket_id, need_state_cluster, cluster_confidence
        (cluster_confidence = fraction of the k neighbors that agreed on the
        assigned cluster — low values flag baskets sitting between need-states)
    """
    k = k if k is not None else config.ASSIGN_NEW_BASKET_K
    ref = reference_embeddings.merge(
        reference_clusters[["basket_id", "need_state_cluster"]], on="basket_id", how="inner"
    )
    if len(ref) < ref["basket_id"].nunique():
        raise ValueError("Duplicate basket_id in reference_embeddings after merge — dedupe first.")

    ref_labels = ref["need_state_cluster"].to_numpy()
    ref_X = normalize(np.stack(ref["gnn_embedding"].values), copy=False)

    new_ids = new_embeddings["basket_id"].tolist()
    new_X = normalize(np.stack(new_embeddings["gnn_embedding"].values), copy=False)

    k_eff = min(k, len(ref))
    if HAVE_PYNNDESCENT:
        index = NNDescent(ref_X, metric="cosine", random_state=SAMPLE_SEED)
        neighbor_indices, _ = index.query(new_X, k=k_eff)
    else:
        nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
        nn.fit(ref_X)
        _, neighbor_indices = nn.kneighbors(new_X)

    assigned, confidence = [], []
    for row in neighbor_indices:
        neighbor_labels = ref_labels[row]
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        top = vals[np.argmax(counts)]
        assigned.append(int(top))
        confidence.append(float(counts.max() / len(row)))

    result = pd.DataFrame({
        "basket_id": new_ids,
        "need_state_cluster": assigned,
        "cluster_confidence": confidence,
    })
    low_conf = (result["cluster_confidence"] < 0.5).sum()
    print(f"Assigned {len(result):,} new baskets to {result['need_state_cluster'].nunique()} "
          f"existing clusters ({low_conf:,} with <50% neighbor agreement)")
    return result


# ─────────────────────────────────────────────
# GMM path — comparison / combination method
# ─────────────────────────────────────────────

def select_k_via_bic(
    basket_gnn_embeddings: pd.DataFrame,
    k_min: int = GMM_K_MIN,
    k_max: int = GMM_K_MAX,
    step: int = GMM_K_STEP,
    n_init: int = GMM_N_INIT,
    covariance_type: str = GMM_COVARIANCE,
) -> pd.DataFrame:
    """
    Standard BIC/AIC sweep for picking K in a GMM — same discipline as
    sweep_resolution() for Leiden: build once, report every K, don't
    auto-pick. This is a reasonable default, NOT your team's real
    best_k_value.py — replace once you share it.
    """
    with progress_step(f"normalising {len(basket_gnn_embeddings):,} embeddings"):
        X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values), copy=False)

    results = []
    ks = list(range(k_min, k_max + 1, step))
    for i, k in enumerate(ks, start=1):
        gmm = GaussianMixture(
            n_components=k, n_init=n_init,
            covariance_type=covariance_type, random_state=SAMPLE_SEED,
        )
        with progress_step(f"fitting GMM k={k}", i, len(ks)):
            gmm.fit(X)
        row = {
            "k": k,
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "log_likelihood": gmm.score(X),
        }
        results.append(row)
        print(f"k={k:>3}  BIC={row['bic']:,.0f}  AIC={row['aic']:,.0f}")

    return pd.DataFrame(results)


def cluster_basket_embeddings_gmm(
    basket_gnn_embeddings: pd.DataFrame,
    n_components: int,
    covariance_type: str = GMM_COVARIANCE,
    save_path: str = GMM_MODEL_PATH,
) -> pd.DataFrame:
    """
    GMM clustering over basket embeddings. Unlike Leiden, a fitted GMM can
    .predict() brand-new baskets directly (see score_new_baskets.py) — no
    k-NN majority-vote workaround needed, which is GMM's main practical
    advantage here. The fitted model is saved to `save_path` for exactly
    that reuse.

    Returns
    -------
    DataFrame: basket_id, need_state_cluster_gmm, gmm_confidence
        (gmm_confidence = the model's own posterior probability for the
        assigned component — how sure the model is about that basket)
    """
    GMM_STEPS = 3
    basket_ids = basket_gnn_embeddings["basket_id"].to_numpy()

    with progress_step(f"normalising {len(basket_ids):,} embeddings", 1, GMM_STEPS):
        # float64, not the float32 the embeddings are stored in.
        #
        # GMM's M-step forms each covariance as E[x^2] - E[x]^2. When a
        # component sits on near-identical points those two terms are nearly
        # equal, and in float32 (eps ~1.2e-7) the subtraction can cancel to a
        # small NEGATIVE number larger in magnitude than reg_covar — at which
        # point _compute_precision_cholesky refuses the covariance and the
        # whole fit aborts. That killed a 42-minute run. sklearn's own error
        # message recommends float64 for exactly this reason.
        #
        # Costs ~29GB instead of ~15GB at 57.1M x 64, plus a transient copy
        # while converting. Unremarkable on a 512GB box.
        X = normalize(
            np.stack(basket_gnn_embeddings["gnn_embedding"].values).astype(np.float64),
            copy=False,
        )

    # verbose=2 / verbose_interval=1 is sklearn's own per-EM-iteration output
    # (iteration number, log-likelihood change, time per iteration) for each of
    # the n_init restarts. This is the only progress signal available from
    # inside the fit, and at full scale each iteration is minutes long, so
    # without it the fit is a single silent call of unknown length.
    gmm = GaussianMixture(
        n_components=n_components, n_init=GMM_N_INIT,
        covariance_type=covariance_type, random_state=SAMPLE_SEED,
        init_params=GMM_INIT_PARAMS, max_iter=GMM_MAX_ITER,
        reg_covar=GMM_REG_COVAR,
        verbose=2, verbose_interval=1,
    )
    print(f"  GMM: k={n_components}, covariance={covariance_type}, "
          f"init={GMM_INIT_PARAMS}, max_iter={GMM_MAX_ITER}, "
          f"reg_covar={GMM_REG_COVAR:g}, dtype={X.dtype}, "
          f"n_init={GMM_N_INIT} restart(s) over {len(basket_ids):,} baskets")
    if GMM_INIT_PARAMS == "kmeans":
        # Announced before the silence, not after it. sklearn prints
        # "Initialization N" and then runs this with no further output, so an
        # unexplained gap of tens of minutes between that line and "Iteration 1"
        # reads exactly like a hang.
        print(f"  init='kmeans' fits a FULL k-means over all {len(basket_ids):,} "
              f"baskets before EM iteration 1, once per restart ({GMM_N_INIT}x). "
              f"Expect a long silent gap after each 'Initialization' line. "
              f"PIPELINE_GMM_INIT_PARAMS=k-means++ skips it.")
    with progress_step(f"fitting GMM (k={n_components})", 2, GMM_STEPS):
        try:
            labels = gmm.fit_predict(X)
        except ValueError as e:
            if "ill-defined empirical covariance" not in str(e):
                raise
            # sklearn's message lists four generic remedies and cannot know
            # which applies. This one does: it names the settings actually in
            # force and why this dataset provokes it.
            raise ValueError(
                f"GMM fit aborted: a component collapsed to a singular covariance.\n"
                f"  settings: k={n_components}, reg_covar={GMM_REG_COVAR:g}, "
                f"dtype={X.dtype}, covariance={covariance_type}\n"
                f"  Why this dataset provokes it: two baskets holding the same "
                f"product set produce the same graph and therefore the SAME 64-dim "
                f"embedding, so exact duplicate points exist in large numbers. A "
                f"component landing on a pile of duplicates has no variance to "
                f"estimate, and its covariance goes singular.\n"
                f"  Fix, in order: raise PIPELINE_GMM_REG_COVAR "
                f"(now {GMM_REG_COVAR:g}, try {GMM_REG_COVAR * 100:g}); then lower "
                f"PIPELINE_GMM_N_COMPONENTS (now {n_components}).\n"
                f"  Original: {e}"
            ) from e

    with progress_step("scoring posterior probabilities", 3, GMM_STEPS):
        # Chunked because only the row-wise max is wanted: a single
        # predict_proba over the full population would allocate
        # n_baskets x n_components float64 (57M x 30 is ~13.7 GB) purely to
        # collapse it to one column. Same chunking need_state_graph's
        # build_gmm_overlap already does, for the same reason.
        confidence = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), GMM_PREDICT_CHUNK):
            stop = min(start + GMM_PREDICT_CHUNK, len(X))
            confidence[start:stop] = gmm.predict_proba(X[start:stop]).max(axis=1)

    joblib.dump(gmm, save_path)
    print(f"Saved fitted GMM model to {save_path}")

    result = pd.DataFrame({
        "basket_id": basket_ids,
        "need_state_cluster_gmm": labels,
        "gmm_confidence": confidence,
    })
    print(f"GMM clusters found: {result['need_state_cluster_gmm'].nunique()}")
    print(result["need_state_cluster_gmm"].value_counts().describe())
    return result


def compare_leiden_gmm(leiden_clusters: pd.DataFrame, gmm_clusters: pd.DataFrame) -> dict:
    """
    Cross-checks the two methods via Adjusted Rand Index — a quick read on
    whether Leiden and GMM are finding roughly the same structure (ARI close
    to 1) or something meaningfully different (ARI close to 0), which is
    itself a useful diagnostic regardless of which method you end up trusting.

    Baskets Leiden could not place (UNCLUSTERED) are excluded. GMM assigns
    every basket, so keeping them would compare a real GMM partition against
    one enormous pseudo-cluster and drive the ARI toward 0 for a reason that
    has nothing to do with the two methods disagreeing.
    """
    comparable = leiden_clusters[leiden_clusters["need_state_cluster"] != UNCLUSTERED]
    n_skipped = len(leiden_clusters) - len(comparable)
    if n_skipped:
        print(f"  Excluding {n_skipped:,} UNCLUSTERED baskets from the comparison "
              f"({n_skipped / len(leiden_clusters):.1%} of the population had no "
              f"Leiden community)")
    merged = comparable.merge(gmm_clusters, on="basket_id", how="inner")
    ari = adjusted_rand_score(merged["need_state_cluster"], merged["need_state_cluster_gmm"])
    print(f"Leiden vs GMM Adjusted Rand Index: {ari:.3f}  "
          f"(1.0 = identical groupings, ~0.0 = no better than random agreement)")
    return {"n_compared": len(merged), "adjusted_rand_index": ari}


if __name__ == "__main__":
    # Two standalone modes, both so that Stage 2 decisions do not require
    # launching pipeline_main and its single-threaded Leiden:
    #
    #   python -u cluster_basket_embeddings.py --k 15 30 50   # coverage probe
    #   python -u cluster_basket_embeddings.py --build-edges  # write edge cache
    #
    # --build-edges exists because rebuilding the kNN graph and CLUSTERING it
    # are separate decisions once clustering moved to cluster_leiden_networkit.
    # Running pipeline_main to refresh the edge cache would also run leidenalg
    # over the result, single-threaded, which is the thing that does not finish.
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 tools: measure mutual-kNN coverage per k, or rebuild "
                    "the cached kNN edge list, without running Leiden."
    )
    parser.add_argument("--build-edges", action="store_true",
                        help="build and cache the kNN edge list for the CURRENT "
                             "config (k, mutual on/off), then stop. Cluster it with "
                             "cluster_leiden_networkit.py.")
    # default=None, not a list: in --build-edges mode an unpassed --k has to
    # fall through to PIPELINE_BASKET_KNN_K. With a list default it never did —
    # args.k[0] silently picked the first PROBE value (10) regardless of config,
    # so a build could be fingerprinted at a k nobody chose and pipeline_main
    # would then rebuild the whole edge list to reach the k config asked for.
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="coverage-probe mode: k values to evaluate "
                             "(default: 10 15 20 30 50). In --build-edges mode, the "
                             "first value overrides PIPELINE_BASKET_KNN_K; omit it to "
                             "use the configured value.")
    parser.add_argument("--mutual", choices=["true", "false"], default=None,
                        help="--build-edges mode: override PIPELINE_USE_MUTUAL_KNN")
    parser.add_argument("--embeddings",
                        default=os.path.join(OUTPUT_DIR, "basket_gnn_embeddings.parquet"),
                        help="basket embeddings parquet produced by Stage 1")
    parser.add_argument("--out",
                        default=os.path.join(OUTPUT_DIR, "coverage_probe.csv"),
                        help="coverage-probe mode: where to write the results table")
    args = parser.parse_args()

    with progress_step(f"loading {args.embeddings}"):
        _frame = pd.read_parquet(args.embeddings)
    print(f"  {len(_frame):,} baskets")

    if args.build_edges:
        _k = args.k[0] if args.k else BASKET_KNN_K
        _mutual = USE_MUTUAL_KNN if args.mutual is None else (args.mutual == "true")
        _k_src = "--k" if args.k else "PIPELINE_BASKET_KNN_K"
        _m_src = "--mutual" if args.mutual is not None else "PIPELINE_USE_MUTUAL_KNN"
        print(f"\nBuilding edge list: k={_k} (from {_k_src}), "
              f"mutual={_mutual} (from {_m_src})")
        if _k != BASKET_KNN_K:
            print(f"  WARNING: this builds a k={_k} graph while PIPELINE_BASKET_KNN_K="
                  f"{BASKET_KNN_K}. pipeline_main reads the config value, so it would "
                  f"see a fingerprint mismatch and rebuild the whole edge list. Set "
                  f"PIPELINE_BASKET_KNN_K={_k} in .env before running it.")
        if not _mutual:
            print(f"  One-directional: every basket keeps its {_k} neighbours whether "
                  f"or not they are returned, so coverage is 100% by construction. "
                  f"Expect roughly {len(_frame) * _k / 1e6:,.0f}M candidate pairs before "
                  f"deduplication — several times the mutual graph.")
        _edges = build_basket_knn_graph(_frame, k=_k, use_mutual=_mutual)
        print(f"\nEdge list ready: {len(_edges):,} edges.")
        print(f"Cluster it with:\n"
              f"  python -u cluster_leiden_networkit.py --resolution {LEIDEN_RESOLUTION}")
        print(f"To explore other resolutions, sweep UPWARD from 1.0:\n"
              f"  python -u cluster_leiden_networkit.py --sweep 1.0 1.5 2.0 3.0\n"
              f"  Below gamma 1.0 this graph collapses — measured on the k=10\n"
              f"  one-directional graph, gamma 0.5 put 99.4% of baskets in a single\n"
              f"  community and 0.05 put 100% of them there. 1.0-3.0 is a stable\n"
              f"  plateau where modularity varies only ~3%.")
    else:
        _table = coverage_probe(_frame, k_values=args.k)
        _table.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")
