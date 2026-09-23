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
    _check_graph_coverage,
    fmt_duration,
    progress_step,
)

NETWORKIT_CLUSTERS_PATH = os.path.join(OUTPUT_DIR, "basket_need_state_clusters_networkit.parquet")


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


def run_parallel_leiden(graph, resolution: float, iterations: int):
    """ParallelLeiden over the whole graph, timed, with modularity reported."""
    print(f"  ParallelLeiden: gamma={resolution}, up to {iterations} iteration(s), "
          f"{nk.getMaxNumberOfThreads()} threads over {graph.numberOfNodes():,} "
          f"vertices / {graph.numberOfEdges():,} edges")

    started = time.perf_counter()
    with progress_step("ParallelLeiden"):
        algorithm = nk.community.ParallelLeiden(
            graph, randomize=True, iterations=iterations, gamma=resolution,
        )
        algorithm.run()
        partition = algorithm.getPartition()
    elapsed = time.perf_counter() - started

    membership = np.asarray(partition.getVector(), dtype=np.int64)
    modularity = nk.community.Modularity().getQuality(partition, graph)

    print(f"  -> {partition.numberOfSubsets():,} communities, "
          f"modularity={modularity:.4f}, in {fmt_duration(elapsed)}")
    return membership, modularity, elapsed


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
    parser.add_argument("--out", default=NETWORKIT_CLUSTERS_PATH)
    parser.add_argument("--resolution", type=float, default=LEIDEN_RESOLUTION)
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

    universe = None
    if args.embeddings and os.path.exists(args.embeddings) and not args.smoke_test:
        with progress_step(f"reading basket ids from {args.embeddings}"):
            all_ids = pd.read_parquet(args.embeddings, columns=["basket_id"])["basket_id"]
        universe = _check_graph_coverage(baskets, all_ids.to_numpy(), args.min_coverage)
        del all_ids
    elif not args.smoke_test:
        print(f"  NOTE: {args.embeddings} not found — skipping the coverage check. "
              f"Only baskets present in the edge list will appear in the output.")

    membership, modularity, elapsed = run_parallel_leiden(
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

    result.to_parquet(args.out, index=False)
    print(f"\nSaved {args.out} ({len(result):,} rows)")
    print(f"  communities: {result.loc[result['need_state_cluster'] != UNCLUSTERED, 'need_state_cluster'].nunique():,}")
    print(f"  modularity : {modularity:.4f}")
    print(f"  Leiden wall: {fmt_duration(elapsed)}")
    print(f"\nCompare against the leidenalg path before switching the pipeline over: "
          f"same graph, so modularity is directly comparable and a large gap in "
          f"either direction is worth understanding rather than accepting.")


if __name__ == "__main__":
    main()
