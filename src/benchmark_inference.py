"""
benchmark_inference.py

Measures where per-basket inference time ACTUALLY goes, on your real
co-purchase matrix and your real baskets — and proves that the optimised
per-basket submatrix lookup returns exactly the same numbers as the old one
before you commit days of compute to it.

Run this BEFORE resuming a long inference job:

    cd src
    python benchmark_inference.py                  # 500 baskets, default
    python benchmark_inference.py --n-baskets 2000 # more samples, tighter estimate

It needs the artifacts a previous pipeline_main.py run already wrote:
    ../data/output/copurchase_sparse.npz
    ../data/output/product_id_to_index.pkl
    ../data/output/product_embeddings.parquet
and the baskets table in DuckDB (dataset tag configurable, default "train").

WHY THIS EXISTS
───────────────
Inference was measured at ~17.4 minutes per 50,000-basket chunk — ~20.9ms per
basket, which over a 57M-basket population is ~9 days. That is far too slow
for a graph with a few dozen nodes whose forward pass costs microseconds, so
the time has to be going somewhere other than the model. This script finds
out where, rather than guessing.

WHAT IT REPORTS
───────────────
  1. Correctness  — optimised submatrix vs. the original csr[rows][:,cols],
                    element-for-element, on YOUR matrix.
  2. Breakdown    — median/mean microseconds per basket for each stage:
                    submatrix extraction, graph construction, batched encode.
  3. Projection   — estimated wall clock for the baskets you have left.
"""

import argparse
import os
import pickle
import statistics
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import config
import duckdb_manager
import basket_store
import parquet_loader
from GraphBuilder import prepare_globals, _basket_dense_cp_submatrix, build_one_graph
from GNN_Train import BasketGNN


def _old_submatrix(products, csr, pid2idx):
    """The original implementation, kept here purely as a correctness and
    speed reference. Materialises whole rows before narrowing to columns."""
    gidx = np.array(sorted({pid2idx[p] for p in products if p in pid2idx}), dtype=np.int64)
    if len(gidx) == 0:
        return {}, np.zeros((0, 0), dtype=np.float32)
    lmap = {int(g): l for l, g in enumerate(gidx)}
    return lmap, np.asarray(csr[gidx][:, gidx].todense(), dtype=np.float32)


def _fmt(us):
    return f"{us/1000:.2f} ms" if us >= 1000 else f"{us:.0f} us"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-baskets", type=int, default=500,
                    help="How many real baskets to time (default 500).")
    ap.add_argument("--dataset-tag", default=config.TRAIN_DATASET_TAG)
    ap.add_argument("--remaining-baskets", type=int, default=None,
                    help="Baskets still to embed, for the wall-clock projection.")
    ap.add_argument("--skip-old", action="store_true",
                    help="Skip the slow reference implementation (use if the old "
                         "path is so slow it dominates this benchmark too).")
    args = ap.parse_args()

    out = config.OUTPUT_DIR
    print("Loading artifacts from a previous pipeline run...")
    with open(os.path.join(out, "product_id_to_index.pkl"), "rb") as f:
        pid2idx = pickle.load(f)
    with open(os.path.join(out, "product_units_avg.pkl"), "rb") as f:
        units_avg = pickle.load(f)
    csr_raw = sp.load_npz(os.path.join(out, "copurchase_sparse.npz"))
    prod_df = parquet_loader.load_product_embeddings(
        os.path.join(out, "product_embeddings.parquet"))
    product_embedding = dict(zip(prod_df["tpnb"], prod_df["embedding"]))

    csr_info = csr_raw.tocsr()
    nnz = csr_info.nnz
    n_prod = csr_info.shape[0]
    print(f"  co-purchase matrix: {n_prod:,} x {n_prod:,}, nnz={nnz:,} "
          f"(mean {nnz/max(n_prod,1):,.0f} nonzeros per product row)")
    print(f"  ^ that mean row length is what the OLD implementation copied per "
          f"basket product.\n")

    G = prepare_globals(
        product_embedding=product_embedding,
        product_id_to_index=pid2idx,
        copurchase_sparse=csr_raw,
        product_units_avg=units_avg,
    )
    csr = G["csr"]

    con = duckdb_manager.get_connection()
    df = basket_store.get_basket_range(con, args.dataset_tag, 1, args.n_baskets)
    if len(df) == 0:
        raise SystemExit(f"No baskets found in baskets_{args.dataset_tag}. "
                         f"Has pipeline_main.py Stage 0 run?")
    sizes = df["products"].map(len)
    print(f"Timing {len(df):,} real baskets "
          f"(products per basket: median {int(sizes.median())}, "
          f"p90 {int(sizes.quantile(0.9))}, max {int(sizes.max())})\n")

    # ── 1. CORRECTNESS ──────────────────────────────────────────────────
    if args.skip_old:
        print("1. CORRECTNESS: skipped (--skip-old)\n")
    else:
        print("1. CORRECTNESS: optimised submatrix vs. original csr[rows][:,cols]")
        n_check = min(100, len(df))
        bad = 0
        for i in range(n_check):
            prods = df["products"].iloc[i]
            om, od = _old_submatrix(prods, csr, pid2idx)
            nm, nd = _basket_dense_cp_submatrix(prods, csr, pid2idx)
            if om != nm or od.shape != nd.shape or not np.array_equal(od, nd):
                bad += 1
                if bad == 1:
                    print(f"   MISMATCH on basket {i} ({df['basket_id'].iloc[i]})")
        if bad:
            print(f"   *** {bad}/{n_check} baskets DIFFER — DO NOT USE THE OPTIMISED "
                  f"PATH. Results would be inconsistent with already-embedded chunks.")
            raise SystemExit(1)
        print(f"   {n_check} baskets: element-for-element identical — safe to resume "
              f"a partially completed run.\n")

    # ── 2. BREAKDOWN ────────────────────────────────────────────────────
    print("2. BREAKDOWN: per-basket cost by stage")
    t_new, t_old, t_graph = [], [], []

    for i in range(len(df)):
        prods = df["products"].iloc[i]
        units = df["units"].iloc[i]

        t0 = time.perf_counter()
        lmap, dense = _basket_dense_cp_submatrix(prods, csr, pid2idx)
        t1 = time.perf_counter()
        build_one_graph(
            products=prods, units=units, basket_id=df["basket_id"].iloc[i],
            emb_matrix=G["emb_matrix"], emb_dim=G["emb_dim"],
            subcluster_arr=G["subcluster_arr"],
            distinctiveness_arr=G["distinctiveness_arr"],
            dense_cp=dense, local_idx=lmap, product_id_to_index=pid2idx,
        )
        t2 = time.perf_counter()
        t_new.append((t1 - t0) * 1e6)
        t_graph.append((t2 - t1) * 1e6)

        if not args.skip_old and i < min(100, len(df)):
            t3 = time.perf_counter()
            _old_submatrix(prods, csr, pid2idx)
            t_old.append((time.perf_counter() - t3) * 1e6)

    med_new = statistics.median(t_new)
    med_graph = statistics.median(t_graph)

    print(f"   submatrix (OPTIMISED) : median {_fmt(med_new):>10}   "
          f"mean {_fmt(statistics.mean(t_new))}")
    if t_old:
        med_old = statistics.median(t_old)
        print(f"   submatrix (ORIGINAL)  : median {_fmt(med_old):>10}   "
              f"mean {_fmt(statistics.mean(t_old))}")
        print(f"   ---> speedup on this stage: {med_old/max(med_new,1e-9):,.0f}x")
    print(f"   build_one_graph       : median {_fmt(med_graph):>10}   "
          f"mean {_fmt(statistics.mean(t_graph))}")

    # Batched encode, the way inference actually runs it.
    print("\n   timing batched model.encode() ...")
    from torch_geometric.loader import DataLoader as PyGDataLoader
    graphs = []
    for i in range(len(df)):
        prods = df["products"].iloc[i]
        lmap, dense = _basket_dense_cp_submatrix(prods, csr, pid2idx)
        graphs.append(build_one_graph(
            products=prods, units=df["units"].iloc[i], basket_id=df["basket_id"].iloc[i],
            emb_matrix=G["emb_matrix"], emb_dim=G["emb_dim"],
            subcluster_arr=G["subcluster_arr"],
            distinctiveness_arr=G["distinctiveness_arr"],
            dense_cp=dense, local_idx=lmap, product_id_to_index=pid2idx))

    model = BasketGNN(in_dim=G["in_dim"], edge_dim=config.EDGE_DIM,
                      hidden_dim=config.HIDDEN_DIM, out_dim=config.OUT_DIM).eval()
    loader = PyGDataLoader(graphs, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            model.encode(batch)
    enc_us = (time.perf_counter() - t0) * 1e6 / len(df)
    print(f"   encode (batched)      : {_fmt(enc_us):>10} per basket "
          f"(torch threads={torch.get_num_threads()})")

    total_new = med_new + med_graph + enc_us
    print(f"\n   TOTAL (optimised)     : {_fmt(total_new)} per basket")
    if t_old:
        total_old = statistics.median(t_old) + med_graph + enc_us
        print(f"   TOTAL (original)      : {_fmt(total_old)} per basket")

    # ── 3. PROJECTION ───────────────────────────────────────────────────
    print("\n3. PROJECTION")
    remaining = args.remaining_baskets
    if remaining is None:
        try:
            remaining = basket_store.count_baskets(con, args.dataset_tag)
            print(f"   (using full basket count; pass --remaining-baskets for "
                  f"a partially completed run)")
        except Exception:
            remaining = 0
    if remaining:
        hrs_new = remaining * total_new / 1e6 / 3600
        print(f"   {remaining:,} baskets remaining")
        print(f"   optimised, 1 process : {hrs_new:,.1f} hours ({hrs_new/24:,.1f} days)")
        if t_old:
            hrs_old = remaining * total_old / 1e6 / 3600
            print(f"   original,  1 process : {hrs_old:,.1f} hours ({hrs_old/24:,.1f} days)")
        cores = os.cpu_count() or 1
        for w in sorted({2, 4, max(1, cores // 2), cores}):
            if w > 1:
                print(f"   optimised, {w:>2} procs   : {hrs_new/w:,.1f} hours "
                      f"(perfect scaling; see the memory caveat in the notes)")
    print("\nDone.")


if __name__ == "__main__":
    main()
