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

import os

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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASKET_KNN_K       = 15     # neighbors per basket before mutual-kNN filtering
USE_MUTUAL_KNN     = True   # an edge only counts if BOTH baskets rank each other in top-K
LEIDEN_RESOLUTION  = 1.0    # starting point — sweep before trusting this, see sweep_resolution()
SAMPLE_SEED        = 42

# GMM defaults — placeholders, see module docstring above
GMM_K_MIN          = 5
GMM_K_MAX          = 60
GMM_K_STEP         = 5
GMM_N_INIT         = 3
GMM_COVARIANCE     = "diag"   # "diag" scales to more dimensions than "full" without blowing up

# All pipeline-produced artifacts land here, not the working directory.
OUTPUT_DIR         = "../data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GMM_MODEL_PATH     = os.path.join(OUTPUT_DIR, "gmm_basket_model.pkl")   # fitted model, reloaded by score_new_baskets.py


# ─────────────────────────────────────────────
# kNN graph over basket embeddings
# ─────────────────────────────────────────────

def build_basket_knn_graph(
    basket_gnn_embeddings: pd.DataFrame,
    k: int = BASKET_KNN_K,
    use_mutual: bool = USE_MUTUAL_KNN,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    basket_gnn_embeddings : DataFrame with columns 'basket_id', 'gnn_embedding'
        (gnn_embedding is a fixed-length float vector per row — the output of
        GNN_Train.train_and_embed()).

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
    basket_ids = basket_gnn_embeddings["basket_id"].tolist()
    n = len(basket_ids)
    X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values))

    if HAVE_PYNNDESCENT:
        print(f"  kNN method: pynndescent (approximate) — k={k}, n={n:,} baskets")
        index = NNDescent(X, n_neighbors=k + 1, metric="cosine", random_state=SAMPLE_SEED)
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
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

    # Drop each row's own self-match (its nearest "neighbor" is itself, at
    # column 0), keep the k candidate neighbor columns, and flatten into one
    # (src, dst, similarity) edge per directed pair — same neighbors the old
    # per-basket loop iterated, just as arrays instead of a Python loop body.
    neighbor_idx  = indices[:, 1:k + 1].astype(np.int64)
    neighbor_dist = distances[:, 1:k + 1]

    src = np.repeat(np.arange(n, dtype=np.int64), neighbor_idx.shape[1])
    dst = neighbor_idx.reshape(-1)
    sim = (1 - neighbor_dist).reshape(-1).astype(np.float64)

    # Defensive: drop any self-match that slipped in at a position other
    # than 0 (e.g. duplicate embeddings) — same as the old loop's `j != i`.
    not_self = dst != src
    src, dst, sim = src[not_self], dst[not_self], sim[not_self]

    # Mutual-kNN check: "does the reverse edge (j, i) also exist among the
    # edges above" — done via one integer key per directed pair and a sorted
    # search, instead of hundreds of millions of dict lookups.
    keys = src * n + dst
    order = np.argsort(keys)
    sorted_keys = keys[order]

    rev_keys   = dst * n + src
    pos        = np.clip(np.searchsorted(sorted_keys, rev_keys), 0, len(sorted_keys) - 1)
    rev_exists = sorted_keys[pos] == rev_keys
    rev_sim    = sim[order[pos]]

    basket_id_arr = np.array(basket_ids)

    if use_mutual:
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

    edges = pd.DataFrame({
        "basket_a": basket_id_arr[edge_a],
        "basket_b": basket_id_arr[edge_b],
        "similarity": edge_sim,
    }).drop_duplicates(subset=["basket_a", "basket_b"])
    edges["weight"] = minmax_scale(edges["similarity"])

    # Visible before _build_igraph/Leiden ever touch this — the kNN graph
    # itself (not just building it) is the next likely memory bottleneck at
    # full basket-population scale (potentially 1B+ edges), flagged here
    # rather than silently discovered when igraph/leidenalg choke on it.
    est_mb = edges.memory_usage(deep=True).sum() / 1e6
    print(f"Basket kNN graph ({'mutual' if use_mutual else 'one-directional'}): "
          f"{len(edges):,} edges over {n:,} baskets "
          f"(~{est_mb:,.0f} MB as a DataFrame, before igraph's own edge-list overhead)")
    return edges[["basket_a", "basket_b", "weight"]]


# ─────────────────────────────────────────────
# Leiden
# ─────────────────────────────────────────────

def _build_igraph(edges: pd.DataFrame):
    baskets = sorted(set(edges["basket_a"]) | set(edges["basket_b"]))
    idx = {b: i for i, b in enumerate(baskets)}

    g = ig.Graph()
    g.add_vertices(len(baskets))
    g.add_edges([(idx[a], idx[b]) for a, b in zip(edges["basket_a"], edges["basket_b"])])
    g.es["weight"] = edges["weight"].tolist()
    return g, baskets


def diagnose_connectivity(edges: pd.DataFrame) -> pd.Series:
    """Same check as multiview_clustering_v5.py — run before trusting a resolution sweep."""
    g, baskets = _build_igraph(edges)
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
    for r in resolutions:
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


def run_leiden_on_basket_graph(edges: pd.DataFrame, resolution: float = LEIDEN_RESOLUTION) -> pd.DataFrame:
    g, baskets = _build_igraph(edges)
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=SAMPLE_SEED,
    )
    return pd.DataFrame({"basket_id": baskets, "need_state_cluster": partition.membership})


def cluster_basket_embeddings(
    basket_gnn_embeddings: pd.DataFrame,
    resolution: float = LEIDEN_RESOLUTION,
) -> pd.DataFrame:
    """Convenience wrapper: build graph -> Leiden -> per-basket need-state cluster."""
    edges = build_basket_knn_graph(basket_gnn_embeddings)
    clusters = run_leiden_on_basket_graph(edges, resolution=resolution)

    print(f"\nNeed-state clusters found: {clusters['need_state_cluster'].nunique()}")
    print(clusters["need_state_cluster"].value_counts().describe())
    return clusters


# ─────────────────────────────────────────────
# Assigning NEW baskets to already-discovered clusters
# ─────────────────────────────────────────────

def assign_new_baskets_to_clusters(
    new_embeddings: pd.DataFrame,
    reference_embeddings: pd.DataFrame,
    reference_clusters: pd.DataFrame,
    k: int = 15,
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
    ref = reference_embeddings.merge(
        reference_clusters[["basket_id", "need_state_cluster"]], on="basket_id", how="inner"
    )
    if len(ref) < ref["basket_id"].nunique():
        raise ValueError("Duplicate basket_id in reference_embeddings after merge — dedupe first.")

    ref_labels = ref["need_state_cluster"].to_numpy()
    ref_X = normalize(np.stack(ref["gnn_embedding"].values))

    new_ids = new_embeddings["basket_id"].tolist()
    new_X = normalize(np.stack(new_embeddings["gnn_embedding"].values))

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
    X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values))

    results = []
    for k in range(k_min, k_max + 1, step):
        gmm = GaussianMixture(
            n_components=k, n_init=n_init,
            covariance_type=covariance_type, random_state=SAMPLE_SEED,
        )
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
    basket_ids = basket_gnn_embeddings["basket_id"].tolist()
    X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values))

    gmm = GaussianMixture(
        n_components=n_components, n_init=GMM_N_INIT,
        covariance_type=covariance_type, random_state=SAMPLE_SEED,
    )
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    confidence = probs.max(axis=1)

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
    """
    merged = leiden_clusters.merge(gmm_clusters, on="basket_id", how="inner")
    ari = adjusted_rand_score(merged["need_state_cluster"], merged["need_state_cluster_gmm"])
    print(f"Leiden vs GMM Adjusted Rand Index: {ari:.3f}  "
          f"(1.0 = identical groupings, ~0.0 = no better than random agreement)")
    return {"n_compared": len(merged), "adjusted_rand_index": ari}
