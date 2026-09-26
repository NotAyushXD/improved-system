"""
cluster_leiden_networkit.py

Stage 2a on all cores.

leidenalg is single-threaded. On the 64-core box this pipeline runs on, a
Leiden pass over the full basket graph pinned exactly one core and did not
finish a single optimiser iteration overnight — 1.6% of the machine doing
100% of the work. NetworKit's ParallelLeiden is the same family of algorithm
with an OpenMP implementation behind it, so the other 63 cores participate.

WHY THIS IS A SEPARATE SCRIPT AND NOT A BRANCH INSIDE pipeline_main
────────────────────────────────────────────────────────────────────
The kNN edge list is already written to disk with a fingerprint manifest
(cluster_basket_embeddings._write_edge_cache), which makes it a natural
process boundary: everything upstream produced that parquet, everything
downstream consumes cluster labels. Running the expensive middle step as its
own process means it can be benchmarked against the leidenalg path on the
identical graph, re-run with different resolutions without touching Stage 1,
and killed without losing anything.

Integrate it into pipeline_main once it has actually beaten leidenalg on
YOUR graph. Until then this is the thing you run to find out.

USAGE
─────
    python -u cluster_leiden_networkit.py
    python -u cluster_leiden_networkit.py --resolution 1.0 --iterations 3
    python -u cluster_leiden_networkit.py --smoke-test 1000000   # mechanics only

RESOLUTION IS NOT BIT-COMPATIBLE WITH leidenalg
───────────────────────────────────────────────
NetworKit's `gamma` and leidenalg's RBConfigurationVertexPartition
`resolution_parameter` are both the gamma of multi-resolution modularity, so
the same value means the same thing in principle — but the two
implementations differ in tie-breaking, node ordering and refinement, and
neither is seeded identically to the other. Expect comparable cluster counts
and modularity, NOT identical labels. Compare with modularity and ARI, not by
diffing assignments.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import networkit as nk

import config
from cluster_basket_embeddings import (
    UNCLUSTERED,
    BASKET_EDGES_PATH,
    LEIDEN_RESOLUTION,
    LEIDEN_N_ITERATIONS,
    MIN_GRAPH_COVERAGE,
    OUTPUT_DIR,
    _atomic_replace,
    _cache_manifest_path,
    _check_graph_coverage,
    _read_manifest,
    fmt_duration,
    progress_step,
)

def labels_path_for(edges_path: str, resolution: float) -> str:
    """
    One label file per (graph, resolution), named after the graph it came from.

    `basket_knn_edges_k10_onedir.parquet` at gamma 1.5 gives
    `basket_need_state_clusters_k10_onedir_r1p5.parquet`.

    Same reasoning as _edge_cache_path: the manifest already refuses a
    mismatch, but a single fixed filename means a run at a different
    resolution overwrites the labels before anything checks. At 36 minutes a
    run, and with a resolution sweep being the normal way to choose one, that
    is worth avoiding structurally rather than remembering to avoid.

    The decimal point becomes 'p' deliberately: os.path.splitext on
    "..._r1.5.parquet" splits at the FIRST dot from the right of the final
    component — giving ".5.parquet" as the extension — which would put the
    manifest sidecar somewhere unrelated to the file it describes.
    """
    stem = os.path.splitext(os.path.basename(edges_path))[0]
    stem = stem.replace("basket_knn_edges", "basket_need_state_clusters")
    return os.path.join(
        OUTPUT_DIR, f"{stem}_r{str(float(resolution)).replace('.', 'p')}.parquet"
    )


NETWORKIT_CLUSTERS_PATH = labels_path_for(BASKET_EDGES_PATH, LEIDEN_RESOLUTION)


# ─────────────────────────────────────────────
# Label cache — so a downstream rerun does not re-cluster
# ─────────────────────────────────────────────
#
# Clustering is now 40 minutes rather than never-finishing, but that is still
# 40 minutes to repeat for nothing when pipeline_main only wants the labels so
# it can run the GMM and the need-state graphs. Fingerprinted rather than
# "file exists", for the same reason the edge cache is: ParallelLeiden
# randomises, so silently reusing labels grown from a different graph or a
# different gamma would produce need-state graphs describing a partition
# nobody can reproduce.


def label_manifest(edge_manifest, resolution: float, iterations: int) -> dict:
    """
    Everything that changes which labels come out.

    The edge manifest is embedded whole rather than re-derived: it already
    fingerprints k, mutual-kNN, seed, basket count, backend and a digest of
    the embeddings, and it is sitting on disk next to the edge list. Recomputing
    an embedding digest here would mean loading the 15GB embeddings frame to
    re-answer a question already answered.
    """
    return {
        "edges": edge_manifest,
        "resolution": float(resolution),
        "iterations": int(iterations),
        "cluster_backend": "networkit-parallelleiden",
    }


def write_labels(labels: pd.DataFrame, manifest: dict, path: str = NETWORKIT_CLUSTERS_PATH):
    """Parquet first, manifest second, each replaced atomically — as _write_edge_cache."""
    tmp = path + ".tmp"
    labels.to_parquet(tmp, index=False)
    _atomic_replace(tmp, path)

    manifest_path = _cache_manifest_path(path)
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    _atomic_replace(tmp_manifest, manifest_path)


def load_cached_labels(edge_manifest, resolution: float, iterations: int,
                       path: str = NETWORKIT_CLUSTERS_PATH):
    """
    Saved labels if they were grown from exactly this graph and gamma, else None.

    A None edge_manifest (no edge cache written) can never match a stored one,
    so unverifiable labels are refused rather than trusted — the safe direction.
    """
    if not os.path.exists(path):
        return None
    if _read_manifest(_cache_manifest_path(path)) != label_manifest(
        edge_manifest, resolution, iterations
    ):
        return None
    return pd.read_parquet(path)


def build_networkit_graph(edges: pd.DataFrame):
    """
    Edge list -> NetworKit graph, plus the basket ids behind each vertex id.

    GraphFromCoo takes the endpoint arrays whole. The one-addEdge-at-a-time
    alternative is a Python loop over ~96.5M edges, which would cost more than
    the parallel Leiden it is feeding — the whole point of coming here.

    Vertices are factorized exactly the way cluster_basket_embeddings does it
    (both endpoint columns concatenated, factorized once, sort=True), so a
    vertex means the same basket in both implementations and their outputs are
    directly comparable.
    """
    n_edges = len(edges)

    with progress_step(f"indexing vertices across {n_edges:,} edges", 1, 2):
        codes, baskets = pd.factorize(
            pd.concat([edges["basket_a"], edges["basket_b"]], ignore_index=True),
            sort=True,
        )
        if len(codes) and codes.min() < 0:
            raise ValueError(
                f"{int((codes < 0).sum()):,} null basket id(s) in the edge list — "
                f"these cannot be graph vertices. Fix them upstream rather than "
                f"silently dropping or remapping the affected edges."
            )
        rows = codes[:n_edges]
        cols = codes[n_edges:]
        weights = edges["weight"].to_numpy(dtype=np.float64)

    with progress_step(f"building NetworKit graph ({len(baskets):,} vertices, "
                       f"{n_edges:,} edges)", 2, 2):
        graph = nk.GraphFromCoo(
            (weights, (rows, cols)),
            n=len(baskets),
            weighted=True,
            directed=False,
        )

    return graph, baskets


def run_parallel_leiden(graph, resolution: float, iterations: int, want_membership=True):
    """
    ParallelLeiden over the whole graph, timed, with modularity reported.

    want_membership=False skips materialising the partition vector, which at
    28.7M vertices is a Python list of 28.7M ints — pure waste during a
    resolution sweep, where only the community count and modularity matter.
    """
    print(f"  ParallelLeiden: gamma={resolution}, up to {iterations} iteration(s), "
          f"{nk.getMaxNumberOfThreads()} threads over {graph.numberOfNodes():,} "
          f"vertices / {graph.numberOfEdges():,} edges")

    started = time.perf_counter()
    with progress_step(f"ParallelLeiden (gamma={resolution})"):
        algorithm = nk.community.ParallelLeiden(
            graph, randomize=True, iterations=iterations, gamma=resolution,
        )
        algorithm.run()
        partition = algorithm.getPartition()
    elapsed = time.perf_counter() - started

    n_communities = partition.numberOfSubsets()
    modularity = nk.community.Modularity().getQuality(partition, graph)

    # Community SIZES, from subsetSizes() rather than the membership vector:
    # one entry per community (tens of thousands) instead of one per vertex
    # (tens of millions), so this is affordable even mid-sweep.
    #
    # The headline count alone cannot tell a real partition from a collapsed
    # one. Dropping the mutual-kNN filter lets generic "hub" baskets bridge
    # unrelated need-states, and the way that failure shows up is not a bad
    # modularity score — it is one community swallowing most of the
    # population while a long tail of singletons makes the count still look
    # healthy. largest_share is the number that exposes it.
    sizes = np.sort(np.asarray(partition.subsetSizes(), dtype=np.int64))[::-1]
    largest_share = sizes[0] / sizes.sum() if len(sizes) else 0.0

    membership = (np.asarray(partition.getVector(), dtype=np.int64)
                  if want_membership else None)

    print(f"  -> {n_communities:,} communities, "
          f"modularity={modularity:.4f}, in {fmt_duration(elapsed)}")
    print(f"     sizes: largest={sizes[0]:,} ({largest_share:.1%} of clustered baskets), "
          f"median={int(np.median(sizes)):,}, "
          f"top5={[int(s) for s in sizes[:5]]}")
    if largest_share > 0.5:
        print(f"     WARNING: one community holds {largest_share:.1%} of all clustered "
              f"baskets. That is hub collapse, not a need-state — unrelated groups have "
              f"been bridged into one. Raise the resolution, or cap in-degree so no "
              f"single basket can absorb thousands of one-directional edges.")
    return membership, n_communities, modularity, elapsed, largest_share


def main():
    parser = argparse.ArgumentParser(
        description="Run Leiden over the cached basket kNN graph using all cores."
    )
    parser.add_argument("--edges", default=BASKET_EDGES_PATH,
                        help="cached kNN edge list parquet")
    parser.add_argument("--embeddings",
                        default=os.path.join(OUTPUT_DIR, "basket_gnn_embeddings.parquet"),
                        help="basket embeddings, read only for its basket_id column so "
                             "coverage can be checked and uncovered baskets labelled")
    # default=None so the name follows the --edges and --resolution ACTUALLY
    # given. Binding it to config's resolution at import meant
    # `--resolution 2.0` wrote its labels over the file belonging to whatever
    # resolution .env happened to name.
    parser.add_argument("--out", default=None,
                        help="default: named after the graph and resolution in use, "
                             "e.g. basket_need_state_clusters_k10_onedir_r1p5.parquet")
    parser.add_argument("--resolution", type=float, default=LEIDEN_RESOLUTION)
    parser.add_argument("--sweep", type=float, nargs="+", default=None, metavar="GAMMA",
                        help="try several resolutions against one graph load and print a "
                             "comparison table. Writes no cluster labels — pick a value "
                             "from the table, then rerun with --resolution. Exists because "
                             "gamma=1.0 produced 94,845 communities on this graph, which "
                             "is a micro-clustering rather than a set of need-states.")
    parser.add_argument("--iterations", type=int, default=LEIDEN_N_ITERATIONS)
    parser.add_argument("--threads", type=int, default=None,
                        help="OpenMP threads (default: all available)")
    parser.add_argument("--min-coverage", type=float, default=MIN_GRAPH_COVERAGE)
    parser.add_argument("--smoke-test", type=int, default=None, metavar="N",
                        help="use only the first N edges. Checks that the mechanics "
                             "work end to end; the resulting clusters are NOT a "
                             "result, because truncating an edge list is not a "
                             "meaningful subgraph.")
    args = parser.parse_args()
    if args.out is None:
        args.out = labels_path_for(args.edges, args.resolution)

    if args.threads is not None:
        nk.setNumberOfThreads(args.threads)
    print(f"NetworKit {nk.__version__}, {nk.getMaxNumberOfThreads()} threads")

    with progress_step(f"loading {args.edges}"):
        edges = pd.read_parquet(args.edges)
    print(f"  {len(edges):,} edges")

    if args.smoke_test:
        edges = edges.head(args.smoke_test)
        print(f"  SMOKE TEST: truncated to {len(edges):,} edges. The clusters this "
              f"produces are not a result — rerun without --smoke-test for one.")

    graph, baskets = build_networkit_graph(edges)
    del edges

    if args.sweep:
        sweep_rows = []
        for gamma in args.sweep:
            _, n_communities, modularity, elapsed, largest_share = run_parallel_leiden(
                graph, gamma, args.iterations, want_membership=False,
            )
            sweep_rows.append({
                "resolution": gamma,
                "n_communities": n_communities,
                "modularity": modularity,
                "largest_share": round(largest_share, 4),
                "seconds": round(elapsed, 1),
            })
        table = pd.DataFrame(sweep_rows)
        sweep_path = os.path.join(OUTPUT_DIR, "networkit_resolution_sweep.csv")
        table.to_csv(sweep_path, index=False)
        print()
        print("Resolution sweep (no labels written — rerun with --resolution <value>):")
        print(table.to_string(index=False))
        print(f"\nSaved {sweep_path}")
        print("\nLower gamma = fewer, larger communities. Modularity alone does not "
              "pick the answer: it tends to favour the fine-grained end, so read it "
              "alongside the community count you can actually act on.")
        return

    universe = None
    if args.embeddings and os.path.exists(args.embeddings) and not args.smoke_test:
        with progress_step(f"reading basket ids from {args.embeddings}"):
            all_ids = pd.read_parquet(args.embeddings, columns=["basket_id"])["basket_id"]
        universe = _check_graph_coverage(baskets, all_ids.to_numpy(), args.min_coverage)
        del all_ids
    elif not args.smoke_test:
        print(f"  NOTE: {args.embeddings} not found — skipping the coverage check. "
              f"Only baskets present in the edge list will appear in the output.")

    membership, n_communities, modularity, elapsed, largest_share = run_parallel_leiden(
        graph, args.resolution, args.iterations,
    )
    del graph

    clustered = pd.DataFrame({"basket_id": baskets, "need_state_cluster": membership})

    if universe is not None:
        result = pd.DataFrame({"basket_id": universe}).merge(
            clustered, on="basket_id", how="left",
        )
        n_missing = int(result["need_state_cluster"].isna().sum())
        result["need_state_cluster"] = (
            result["need_state_cluster"].fillna(UNCLUSTERED).astype(np.int64)
        )
        if n_missing:
            print(f"  {n_missing:,} baskets had no vertex in the graph — "
                  f"need_state_cluster={UNCLUSTERED}")
    else:
        result = clustered

    edge_manifest = _read_manifest(_cache_manifest_path(args.edges))
    if edge_manifest is None:
        print(f"  NOTE: no manifest beside {args.edges}, so these labels cannot be "
              f"fingerprinted. pipeline_main will refuse to reuse them and will ask "
              f"you to re-cluster.")
    write_labels(result, label_manifest(edge_manifest, args.resolution, args.iterations),
                 path=args.out)
    print(f"\nSaved {args.out} ({len(result):,} rows)")
    print(f"  communities: {result.loc[result['need_state_cluster'] != UNCLUSTERED, 'need_state_cluster'].nunique():,}")
    print(f"  modularity : {modularity:.4f}")
    print(f"  Leiden wall: {fmt_duration(elapsed)}")
    print(f"\nCompare against the leidenalg path before switching the pipeline over: "
          f"same graph, so modularity is directly comparable and a large gap in "
          f"either direction is worth understanding rather than accepting.")


if __name__ == "__main__":
    main()
