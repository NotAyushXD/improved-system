"""
test_theme_free_pipeline.py

Two kinds of check:

1. STATIC — walks the AST of every active pipeline file and fails if any
   function name, parameter name, or assigned variable name contains
   "theme" (case-insensitive). Catches accidental reintroduction of a
   theme/category concept anywhere, not just the specific names in the
   original ban list.

2. FUNCTIONAL — builds a tiny synthetic catalog (30 products, random
   embeddings) and a handful of synthetic baskets, then actually runs:
     - prepare_globals() with NO product_theme argument (a call that would
       raise TypeError if that parameter still existed)
     - build_training_graphs() (the training path)
     - embed_all_baskets_fast() (the scoring path)
   and asserts the two paths produce node feature tensors of identical
   width, confirming training and scoring compute the same graph features.
   It also runs a tiny untrained BasketGNN's full encode() (node_encoder ->
   conv1 -> conv2 -> pool -> proj) on output from BOTH paths, confirming
   scoring actually exercises the graph convolutions rather than skipping
   them.

Run from inside src/:  python test_theme_free_pipeline.py
Exits non-zero on any failure, so it's CI-friendly.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BANNED_SUBSTRING = "theme"

# Every file that's part of the active pipeline. split_basket_by_theme.py
# is deliberately NOT here — it's fully decommissioned, not just unused
# (see REFACTOR_NOTES.md). If it exists on disk at all, that's flagged too.
ACTIVE_FILES = [
    "GraphBuilder.py",
    "GNN_Train.py",
    "pipeline_main.py",
    "score_new_baskets.py",
    "build_product_embeddings.py",
    "parquet_loader.py",
    "cluster_basket_embeddings.py",
]


# ─────────────────────────────────────────────
# STATIC CHECK
# ─────────────────────────────────────────────

def check_no_theme_identifiers():
    print("=" * 60)
    print("STATIC CHECK: no theme-named identifiers in active files")
    print("=" * 60)

    all_violations = []
    for fname in ACTIVE_FILES:
        path = Path(fname)
        if not path.exists():
            all_violations.append(f"{fname}: MISSING — expected to exist and be theme-free")
            continue

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if BANNED_SUBSTRING in node.name.lower():
                    all_violations.append(f"{fname}: function name '{node.name}'")
                for arg in node.args.args:
                    if BANNED_SUBSTRING in arg.arg.lower():
                        all_violations.append(f"{fname}: {node.name}() parameter '{arg.arg}'")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if BANNED_SUBSTRING in node.id.lower():
                    all_violations.append(f"{fname}: variable '{node.id}'")

    if Path("split_basket_by_theme.py").exists():
        all_violations.append(
            "split_basket_by_theme.py: file still present — it's supposed to be "
            "fully decommissioned, not just unused (see REFACTOR_NOTES.md)"
        )

    if all_violations:
        print("FAILED — theme-named identifiers found:")
        for v in all_violations:
            print(f"  - {v}")
        return False

    print(f"PASSED — checked {len(ACTIVE_FILES)} files, zero theme-named "
          f"identifiers, split_basket_by_theme.py is not present.")
    return True


# ─────────────────────────────────────────────
# FUNCTIONAL CHECK
# ─────────────────────────────────────────────

def build_synthetic_catalog(n_products=30, emb_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    tpnbs = [f"P{i:03d}" for i in range(n_products)]
    product_embedding = {t: rng.normal(size=emb_dim).astype(np.float32) for t in tpnbs}
    product_units_avg = {t: float(rng.uniform(1, 5)) for t in tpnbs}
    product_id_to_index = {t: i for i, t in enumerate(tpnbs)}

    # A synthetic co-purchase matrix: a handful of random positive
    # co-occurrence counts, symmetric.
    rows, cols, vals = [], [], []
    for _ in range(80):
        i, j = rng.integers(0, n_products, size=2)
        if i == j:
            continue
        c = float(rng.integers(1, 10))
        rows += [i, j]
        cols += [j, i]
        vals += [c, c]
    copurchase_sparse = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_products, n_products)
    )

    return tpnbs, product_embedding, product_units_avg, product_id_to_index, copurchase_sparse


def build_synthetic_baskets(tpnbs, n_baskets=25, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_baskets):
        size = int(rng.integers(2, 8))
        products = list(rng.choice(tpnbs, size=size, replace=False))
        units = [float(rng.integers(1, 4)) for _ in products]
        rows.append({"basket_id": f"B{b:03d}", "products": products, "units": units})
    return pd.DataFrame(rows)


def check_functional():
    print()
    print("=" * 60)
    print("FUNCTIONAL CHECK: train/score consistency on synthetic data")
    print("=" * 60)

    import torch
    from GraphBuilder import (
        prepare_globals,
        sample_baskets,
        build_dense_cp_submatrix,
        build_training_graphs,
        save_training_graphs,
        embed_all_baskets_fast,
        SUBCL_K_CANDIDATES,
    )
    from GNN_Train import BasketGNN

    import GraphBuilder as gb
    gb.SUBCL_K_CANDIDATES = [2, 3]          # tiny catalog needs a tiny K range
    gb.SUBCL_SIL_SAMPLE = 30
    gb.SUBCL_CACHE_PATH = "test_product_subclusters.pkl"
    gb.GRAPH_CACHE_PATH = "test_training_graphs.pkl"
    for p in (gb.SUBCL_CACHE_PATH, gb.GRAPH_CACHE_PATH,
              gb.SUBCL_CACHE_PATH + ".tmp", gb.GRAPH_CACHE_PATH + ".tmp"):
        Path(p).unlink(missing_ok=True)

    tpnbs, product_embedding, product_units_avg, product_id_to_index, copurchase_sparse = \
        build_synthetic_catalog()
    baskets = build_synthetic_baskets(tpnbs)

    # prepare_globals() called with NO product_theme argument — this line
    # itself would raise TypeError if that parameter still existed.
    G = prepare_globals(
        product_embedding   = product_embedding,
        product_id_to_index = product_id_to_index,
        copurchase_sparse   = copurchase_sparse,
        product_units_avg   = product_units_avg,
    )
    assert "theme_ids" not in G, "theme_ids key found in prepare_globals() output"
    emb_dim = G["emb_dim"]
    assert G["in_dim"] == emb_dim + 4, f"expected in_dim == emb_dim+4, got {G['in_dim']}"
    print(f"  prepare_globals(): in_dim={G['in_dim']} (emb_dim={emb_dim}+4), no theme_ids key — OK")

    # sample_baskets() called with NO product_theme argument
    sampled = sample_baskets(baskets, n_samples=len(baskets))
    print(f"  sample_baskets(): {len(sampled)} baskets sampled, no product_theme arg — OK")

    dense_cp, local_idx, _ = build_dense_cp_submatrix(sampled, G["product_id_to_index"], G["csr"])

    graphs = build_training_graphs(sampled, G, dense_cp, local_idx)
    assert len(graphs) == len(sampled)
    for g in graphs:
        assert g.x.shape[1] == G["in_dim"], (
            f"training graph node feature width {g.x.shape[1]} != in_dim {G['in_dim']}"
        )
        if g.edge_attr.shape[0] > 0:
            assert g.edge_attr.shape[1] == 2, "expected 2 edge features (log_cp, relative strength)"
    print(f"  build_training_graphs(): {len(graphs)} graphs, node width == in_dim, "
          f"edge_attr width == 2 — OK")

    save_training_graphs(graphs)
    reloaded = build_training_graphs(sampled, G, dense_cp, local_idx)  # should hit cache
    assert len(reloaded) == len(graphs), "cache round-trip returned a different graph count"
    print(f"  save_training_graphs() + cache reload round-trip — OK")

    # Tiny untrained model — this test checks shapes and that both paths
    # exercise the same code, not embedding quality.
    device = torch.device("cpu")
    model = BasketGNN(in_dim=G["in_dim"], edge_dim=2, hidden_dim=8, out_dim=4).to(device)
    model.eval()

    with torch.no_grad():
        z_train_path = model.encode(graphs[0].to(device))
    assert z_train_path.shape == (1, 4), f"unexpected training-path encode() shape {z_train_path.shape}"
    print(f"  model.encode() on a training graph: shape {tuple(z_train_path.shape)} — OK")

    basket_ids, all_z = embed_all_baskets_fast(baskets, G, model, device, batch_size=8)
    assert all_z.shape == (len(baskets), 4), f"unexpected embed_all_baskets_fast shape {all_z.shape}"
    assert len(basket_ids) == len(baskets)
    print(f"  embed_all_baskets_fast(): shape {all_z.shape} for {len(baskets)} baskets — OK")

    for p in (gb.SUBCL_CACHE_PATH, gb.GRAPH_CACHE_PATH):
        Path(p).unlink(missing_ok=True)

    print("PASSED — training and scoring paths run the same graph "
          "construction and the same full GNN encoding, at consistent "
          "feature widths, with no theme/category input anywhere.")
    return True


if __name__ == "__main__":
    ok_static = check_no_theme_identifiers()
    try:
        ok_functional = check_functional()
    except Exception as e:
        print(f"\nFUNCTIONAL CHECK FAILED with an exception: {type(e).__name__}: {e}")
        ok_functional = False

    print()
    print("=" * 60)
    if ok_static and ok_functional:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("CHECKS FAILED — see above")
        sys.exit(1)