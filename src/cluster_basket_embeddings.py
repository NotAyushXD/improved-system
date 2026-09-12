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
GMM_MODEL_PATH     = "gmm_basket_model.pkl"   # fitted model, reloaded by score_new_baskets.py


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
    """
    basket_ids = basket_gnn_embeddings["basket_id"].tolist()
    X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values))

    if HAVE_PYNNDESCENT:
        index = NNDescent(X, n_neighbors=k + 1, metric="cosine", random_state=SAMPLE_SEED)
        indices, distances = index.neighbor_graph
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

    # per-basket top-K neighbor sets + directional similarities, needed for
    # the mutual-kNN check below
    neighbor_sets, directional_sim = [], {}
    for i in range(len(basket_ids)):
        neighbors_i = set()
        for j_pos in range(1, k + 1):
            j = int(indices[i, j_pos])
            if j != i:
                neighbors_i.add(j)
                directional_sim[(i, j)] = 1 - distances[i, j_pos]
        neighbor_sets.append(neighbors_i)

    rows = []
    if use_mutual:
        for i in range(len(basket_ids)):
            for j in neighbor_sets[i]:
                if i < j and i in neighbor_sets[j]:
                    sim_ij = directional_sim.get((i, j))
                    sim_ji = directional_sim.get((j, i))
                    avg_sim = float(np.mean([s for s in [sim_ij, sim_ji] if s is not None]))
                    if avg_sim > 0:
                        rows.append({"basket_a": basket_ids[i], "basket_b": basket_ids[j], "similarity": avg_sim})
    else:
        for i in range(len(basket_ids)):
            for j in neighbor_sets[i]:
                if directional_sim[(i, j)] > 0:
                    a, b = sorted([basket_ids[i], basket_ids[j]])
                    rows.append({"basket_a": a, "basket_b": b, "similarity": directional_sim[(i, j)]})

    edges = pd.DataFrame(rows).drop_duplicates(subset=["basket_a", "basket_b"])
    edges["weight"] = minmax_scale(edges["similarity"])

    print(f"Basket kNN graph ({'mutual' if use_mutual else 'one-directional'}): "
          f"{len(edges)} edges over {len(basket_ids)} baskets")
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
