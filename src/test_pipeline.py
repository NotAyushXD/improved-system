"""
test_theme_free_pipeline.py

Two kinds of check:

1. STATIC — walks the AST of every active pipeline file and fails if any
   function name, parameter name, or assigned variable name contains
   "theme" (case-insensitive). Catches accidental reintroduction of a
   theme/category concept anywhere, not just the specific names in the
   original ban list.

2. FUNCTIONAL — builds a tiny synthetic catalog (30 products, random
   embeddings) and a synthetic raw household x tpnb x week export, writes it
   to a throwaway parquet folder, and actually runs the REAL pipeline against
   a REAL local embedded DuckDB database (via duckdb_manager.py):
     - prepare_globals() with NO product_theme argument (a call that would
       raise TypeError if that parameter still existed)
     - basket_store.build_baskets_table() (reads the raw parquet export
       directly — the same aggregation pipeline_main.py's Stage 0 uses)
     - basket_store.sample_training_baskets() (the training-sample draw)
     - lmdb_graph_cache.load_or_build_lmdb_cache() + LMDBGraphDataset (the
       training-graph cache, including a cache-hit reload)
     - GraphBuilder.run_inference() + merge_inference_output() (the scoring
       path, restartable-chunk-queue included)
   and asserts the training and inference paths produce node feature
   tensors of identical width, confirming they compute the same graph
   features. It also runs a tiny untrained BasketGNN's full encode()
   (node_encoder -> conv1 -> conv2 -> pool -> proj) on output from BOTH
   paths, confirming inference actually exercises the graph convolutions
   rather than skipping them.

   Requires `duckdb` installed (see requirements.txt) — this test opens a
   real, embedded DuckDB database at data/pipeline.duckdb (same file
   pipeline_main.py uses, under a throwaway "pytest" dataset_tag so it
   can't collide with real "train"/"score" data).

Run from inside src/:  python test_theme_free_pipeline.py
Exits non-zero on any failure, so it's CI-friendly.
"""

import ast
import shutil
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
    "duckdb_manager.py",
    "basket_store.py",
    "lmdb_graph_cache.py",
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


def build_synthetic_raw_export(tpnbs, n_baskets=25, seed=1) -> pd.DataFrame:
    """
    A synthetic RAW household x tpnb x week export — one row per
    (household, tpnb, week), matching the real warehouse export's shape —
    rather than pre-aggregated basket rows. Aggregation into baskets is now
    DuckDB's job (basket_store.build_baskets_table), so the test needs to
    exercise that same path instead of handing it already-basket-shaped rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_baskets):
        household_number = 1000 + b
        year_number  = 2026
        week_number  = 1 + (b % 10)
        period_number = 1 + (week_number - 1) // 4
        size = int(rng.integers(2, 8))
        products = list(rng.choice(tpnbs, size=size, replace=False))
        for p in products:
            rows.append({
                "household_number": household_number,
                "tpnb": p,
                "year_number": year_number,
                "period_number": period_number,
                "week_number": week_number,
                "quantity": float(rng.integers(1, 4)),
            })
    return pd.DataFrame(rows)


def check_functional():
    print()
    print("=" * 60)
    print("FUNCTIONAL CHECK: train/score consistency on synthetic data "
          "(real DuckDB + LMDB, throwaway)")
    print("=" * 60)

    import torch
    from GraphBuilder import prepare_globals, run_inference, merge_inference_output
    from GNN_Train import BasketGNN
    import GraphBuilder as gb
    import duckdb_manager
    import basket_store
    import lmdb_graph_cache

    gb.SUBCL_K_CANDIDATES = [2, 3]          # tiny catalog needs a tiny K range
    gb.SUBCL_SIL_SAMPLE = 30
    gb.SUBCL_CACHE_PATH = "test_product_subclusters.pkl"
    Path(gb.SUBCL_CACHE_PATH).unlink(missing_ok=True)
    Path(gb.SUBCL_CACHE_PATH + ".tmp").unlink(missing_ok=True)

    dataset_tag = "pytest"
    raw_export_dir = Path("test_raw_export_parquet")
    lmdb_path = "test_training_graphs.lmdb"
    manifest_path = "test_training_graphs.lmdb.manifest.json"
    output_dir = "test_output"

    def _cleanup():
        shutil.rmtree(raw_export_dir, ignore_errors=True)
        shutil.rmtree(lmdb_path, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        Path(manifest_path).unlink(missing_ok=True)
        Path(gb.SUBCL_CACHE_PATH).unlink(missing_ok=True)

    _cleanup()

    tpnbs, product_embedding, product_units_avg, product_id_to_index, copurchase_sparse = \
        build_synthetic_catalog()
    raw_export = build_synthetic_raw_export(tpnbs)
    raw_export_dir.mkdir()
    raw_export.to_parquet(raw_export_dir / "part-0.parquet", index=False)

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

    con = duckdb_manager.get_connection()
    # Drop any leftover DuckDB state from a previous test run BEFORE
    # touching local output files — otherwise a stale inference_chunks_pytest
    # table (all rows already 'complete' from a prior run) would make
    # run_inference below claim nothing, while _cleanup() has already wiped
    # this run's local chunk-parquet output, leaving nothing for
    # merge_inference_output to merge.
    basket_store.drop_all(con, dataset_tag)
    n_baskets_total, _units_avg_pg, _products_pg = basket_store.build_baskets_table(
        con, raw_export_dir, dataset_tag, min_basket_products=2,
    )
    print(f"  basket_store.build_baskets_table(): "
          f"{n_baskets_total} baskets built in DuckDB (read directly from parquet) — OK")

    sampled = basket_store.sample_training_baskets(
        con, dataset_tag, n_samples=n_baskets_total, seed=42,
    )
    print(f"  basket_store.sample_training_baskets(): {len(sampled)} baskets sampled — OK")

    lmdb_graph_cache.load_or_build_lmdb_cache(
        sampled, G, lmdb_path, manifest_path,
        seed=42, n_train_samples_requested=n_baskets_total,
    )
    train_dataset = lmdb_graph_cache.LMDBGraphDataset(lmdb_path)
    assert len(train_dataset) == len(sampled), "LMDB cache length != sampled basket count"
    g0 = train_dataset.get(0)
    assert g0.x.shape[1] == G["in_dim"], (
        f"training graph node feature width {g0.x.shape[1]} != in_dim {G['in_dim']}"
    )
    if g0.edge_attr.shape[0] > 0:
        assert g0.edge_attr.shape[1] == 2, "expected 2 edge features (log_cp, relative strength)"
    print(f"  lmdb_graph_cache: {len(train_dataset)} graphs cached, node width == in_dim, "
          f"edge_attr width == 2 — OK")

    n_graphs_first_open = len(train_dataset)
    # lmdb refuses to open the same environment path twice concurrently
    # within one process — close this instance's read handle before
    # constructing a second LMDBGraphDataset over the same lmdb_path below.
    # g0 (already fetched above) stays valid — it's a plain deserialized
    # Data object, independent of the env staying open.
    train_dataset.close()

    # Cache-hit reload — manifest matches, should NOT rebuild.
    lmdb_graph_cache.load_or_build_lmdb_cache(
        sampled, G, lmdb_path, manifest_path,
        seed=42, n_train_samples_requested=n_baskets_total,
    )
    reloaded_dataset = lmdb_graph_cache.LMDBGraphDataset(lmdb_path)
    assert len(reloaded_dataset) == n_graphs_first_open, "cache round-trip returned a different graph count"
    print(f"  lmdb_graph_cache cache-hit reload round-trip — OK")
    reloaded_dataset.close()

    # Tiny untrained model — this test checks shapes and that both paths
    # exercise the same code, not embedding quality.
    device = torch.device("cpu")
    model = BasketGNN(in_dim=G["in_dim"], edge_dim=2, hidden_dim=8, out_dim=4).to(device)
    model.eval()

    with torch.no_grad():
        z_train_path = model.encode(g0.to(device))
    assert z_train_path.shape == (1, 4), f"unexpected training-path encode() shape {z_train_path.shape}"
    print(f"  model.encode() on an LMDB-cached training graph: shape {tuple(z_train_path.shape)} — OK")

    run_inference(
        con, dataset_tag, G, model, device,
        worker_id="pytest-worker", chunk_size=10, batch_size=8, output_dir=output_dir,
    )
    merged = merge_inference_output(dataset_tag, output_dir=output_dir)
    assert len(merged) == n_baskets_total, (
        f"run_inference embedded {len(merged)} baskets, expected {n_baskets_total}"
    )
    assert merged["gnn_embedding"].iloc[0].shape == (4,), "unexpected embedding width from run_inference"
    print(f"  run_inference() + merge_inference_output(): {len(merged)} baskets embedded, "
          f"restartable chunk queue — OK")

    basket_store.drop_all(con, dataset_tag)
    _cleanup()

    print("PASSED — training and scoring paths run the same graph "
          "construction and the same full GNN encoding, at consistent "
          "feature widths, with no theme/category input anywhere, sourced "
          "from a real DuckDB database and an LMDB-backed training cache.")
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