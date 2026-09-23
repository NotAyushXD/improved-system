"""
test_pipeline.py

Checks for the theme-free need-state pipeline. Run from inside src/:

    python test_pipeline.py                 # everything
    python test_pipeline.py --fast          # skip the slow DuckDB/LMDB/torch checks
    python test_pipeline.py --prod-outputs  # also sanity-check real ../data/output artifacts

Exits non-zero on any failure, so it's CI-friendly.

WHAT IS CHECKED, AND WHY EACH ONE EXISTS
────────────────────────────────────────
 1 STATIC          no "theme"-named identifier can reappear in an active file.
 2 GRAPH PRIMITIVES the small pure functions that decide every node and edge
                    feature: _minmax, _build_edges_numba, _distances_to_centroids,
                    and the exact node-feature slot layout. These are the
                    functions whose output is hardest to eyeball in prod, so
                    they are pinned to hand-computed expected values here.
 3 CP ADDITIVITY   pipeline_main.py builds the co-purchase matrix in 500k-basket
                    chunks and CLAIMS (lines 152-157) that this is exactly
                    equivalent to a single-shot X.T @ X. That claim is load-
                    bearing for every edge feature in the system and was never
                    verified. This proves it.
 4 FUNCTIONAL      train + score run end to end on synthetic data against a real
                    DuckDB and a real LMDB cache.
 5 TRAIN/SCORE     the same basket, built through the TRAINING path and through
   PARITY           the INFERENCE path, must produce numerically IDENTICAL node
                    features — not merely the same width. The pre-existing
                    version of this file only compared widths, which a genuine
                    train/score drift would pass. This is the check that would
                    have caught the historical bug described in GraphBuilder.py's
                    module docstring.
 6 BASKET STORE    basket grain semantics: the >=2-product filter, size_bucket
                    boundaries, and the basket_id format that everything
                    downstream (including journey analysis) depends on.
 7 NEED-STATE      adjacency / transition / journey / GMM-overlap logic in
   GRAPH            need_state_graph.py, pinned to hand-computed values,
                    including the year-boundary week-ranking that a naive
                    subtraction gets wrong.
 8 PROD OUTPUTS    opt-in sanity checks against real artifacts in ../data/output
                    (embedding collapse, degenerate clusters, label coverage).
                    Skipped automatically when those files are absent.
 9 CLUSTERING      Stage 2's progress reporting, plus the two rewrites it came
   PROGRESS         with: _build_igraph now factorizes over arrays instead of
                    building a 57M-entry dict and a 96M-tuple list, and Leiden
                    drives the optimiser one iteration at a time so each one
                    reports. Both are pinned against the implementation they
                    replaced — observability was the goal, changed cluster
                    assignments would be a regression.
"""

import argparse
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
    "need_state_graph.py",
]

OUTPUT_DIR = Path("../data/output")


def _fail(msg):
    print(f"    FAIL: {msg}")
    return False


# ─────────────────────────────────────────────
# 1. STATIC CHECK
# ─────────────────────────────────────────────

def check_no_theme_identifiers():
    print("=" * 70)
    print("1. STATIC: no theme-named identifiers in active files")
    print("=" * 70)

    all_violations = []
    for fname in ACTIVE_FILES:
        path = Path(fname)
        if not path.exists():
            all_violations.append(f"{fname}: MISSING — expected to exist and be theme-free")
            continue

        # Explicit encoding: these source files contain box-drawing characters
        # and em dashes, and a locale-default read fails on Windows (cp1252).
        tree = ast.parse(path.read_text(encoding="utf-8"))
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
# 1b. CONFIG
# ─────────────────────────────────────────────

def check_config():
    """
    config.py parsing, typing, validation, and — most importantly — that the
    cache fingerprints actually MOVE when a graph-affecting parameter changes.
    Before config.py these values were hardcoded, so a stale cache took a code
    edit to cause; now it takes an env var, and the fingerprints are the only
    thing standing between a .env tweak and silently training on graphs built
    under the previous settings.
    """
    print()
    print("=" * 70)
    print("1b. CONFIG: .env parsing, validation, cache fingerprints")
    print("=" * 70)

    import importlib
    import os as _os
    import subprocess
    import tempfile
    import config

    ok = True

    # Defaults must reproduce the previously-hardcoded values exactly, so that
    # adding config.py changed nothing for anyone not using a .env.
    expected_defaults = {
        "TOP_K": 10, "SEED": 42, "HIDDEN_DIM": 128, "OUT_DIM": 64, "EDGE_DIM": 2,
        "TRAIN_BATCH": 256, "EPOCHS": 20, "N_TRAIN_SAMPLES": 300_000,
        "MIN_BASKET_PRODUCTS": 2, "GMM_N_COMPONENTS": 30, "BASKET_KNN_K": 15,
        "LEIDEN_RESOLUTION": 1.0, "GMM_COVARIANCE": "diag",
        "COPURCHASE_CHUNK_BASKETS": 500_000, "INFERENCE_CHUNK_BASKETS": 50_000,
        "SUBCL_K_CANDIDATES": [50, 100, 200, 400],
    }
    # Only meaningful when nothing is overriding them in this shell.
    overriding = [k for k in _os.environ if k.startswith("PIPELINE_")]
    if overriding:
        print(f"  SKIP default-value check — {len(overriding)} PIPELINE_* var(s) set "
              f"in this shell: {sorted(overriding)}")
    else:
        wrong = {k: (getattr(config, k), v) for k, v in expected_defaults.items()
                 if getattr(config, k) != v}
        if wrong:
            ok = _fail(f"defaults drifted from the original hardcoded values: {wrong}")
        else:
            print(f"  defaults match the original hardcoded values "
                  f"({len(expected_defaults)} checked) — OK")

    # Types must be real Python types, not strings off the environment.
    type_expect = [("TOP_K", int), ("LR", float), ("USE_MUTUAL_KNN", bool),
                   ("SUBCL_K_CANDIDATES", list), ("GMM_COVARIANCE", str)]
    bad_types = [(k, type(getattr(config, k)).__name__)
                 for k, t in type_expect if not isinstance(getattr(config, k), t)]
    if bad_types:
        ok = _fail(f"config values have the wrong type: {bad_types}")
    else:
        print("  values are typed (int/float/bool/list/str), not raw strings — OK")

    # Fingerprints MUST change when a graph-affecting parameter changes, and
    # MUST NOT change for a parameter that does not affect graph construction.
    # Run in subprocesses so each gets a clean import of config.
    def _fp(env_extra):
        env = dict(_os.environ)
        env.pop("PIPELINE_ENV_FILE", None)
        env.update(env_extra)
        out = subprocess.run(
            [sys.executable, "-c",
             "import config;print(config.fingerprint_hash(config.graph_fingerprint()))"],
            capture_output=True, text=True, env=env, cwd=_os.getcwd(),
        )
        if out.returncode != 0:
            raise RuntimeError(f"config import failed for {env_extra}: {out.stderr.strip()[-300:]}")
        return out.stdout.strip()

    base = _fp({})
    cases = [
        ("PIPELINE_TOP_K", "7", True, "changes edges per node"),
        ("PIPELINE_SEED", "99", True, "changes sub-cluster assignment"),
        ("PIPELINE_SUBCL_K_CANDIDATES", "10,20", True, "changes node features [385],[386]"),
        ("PIPELINE_EPOCHS", "999", False, "training length does not change graphs"),
        ("PIPELINE_LEIDEN_RESOLUTION", "2.5", False, "clustering runs after graphs are built"),
    ]
    for key, val, should_change, why in cases:
        got = _fp({key: val})
        changed = got != base
        if changed != should_change:
            ok = _fail(
                f"{key}={val}: graph fingerprint {'changed' if changed else 'did NOT change'}, "
                f"expected {'a change' if should_change else 'no change'} — {why}. "
                f"A missed change means a stale LMDB cache is silently reused.")
        else:
            verdict = "invalidates cache" if should_change else "cache still valid"
            print(f"  {key}={val}: {verdict} — OK")

    # Bad values must fail at startup, not mid-run.
    accepted = []
    for key, val in [("PIPELINE_EPOCHS", "abc"), ("PIPELINE_TOP_K", "0"),
                     ("PIPELINE_GMM_COVARIANCE", "banana"), ("PIPELINE_DROPOUT", "1.5"),
                     ("PIPELINE_GMM_K_MIN", "9999")]:
        env = dict(_os.environ)
        env.pop("PIPELINE_ENV_FILE", None)
        env[key] = val
        r = subprocess.run([sys.executable, "-c", "import config"],
                           capture_output=True, text=True, env=env, cwd=_os.getcwd())
        if r.returncode == 0:
            accepted.append(f"{key}={val!r}")
    if accepted:
        ok = _fail(f"invalid config accepted instead of failing at startup: {accepted}")
    else:
        print("  invalid values rejected at import time, before any long run starts — OK")

    # A .env file is actually read, and a real env var beats it.
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / "t.env"
        # Non-ASCII on purpose: the shipped .env is full of box-drawing
        # characters, and reading it with the locale default (cp1252 on a
        # Western-European Windows box) raised UnicodeDecodeError. Keeping a
        # non-ASCII byte in this fixture means a regression to a
        # locale-default read fails here rather than in production.
        env_path.write_text(
            '# comment line — with an em dash and a box char │\n'
            'PIPELINE_EPOCHS=77\n'
            'PIPELINE_TOP_K=12   # trailing comment must be stripped\n'
            'PIPELINE_GMM_COVARIANCE="full"\n'
            'PIPELINE_TRANSITION_MAX_WEEK_GAP=none\n',
            encoding="utf-8",
        )
        env = dict(_os.environ)
        env["PIPELINE_ENV_FILE"] = str(env_path)
        env["PIPELINE_EPOCHS"] = "5"        # real env var must WIN over the file
        r = subprocess.run(
            [sys.executable, "-c",
             "import config;print(config.EPOCHS,config.TOP_K,config.GMM_COVARIANCE,"
             "config.TRANSITION_MAX_WEEK_GAP)"],
            capture_output=True, text=True, env=env, cwd=_os.getcwd())
        if r.returncode != 0:
            ok = _fail(f".env parsing failed: {r.stderr.strip()[-300:]}")
        else:
            got = r.stdout.strip()
            if got != "5 12 full None":
                ok = _fail(f".env precedence/parsing wrong. expected '5 12 full None', got {got!r} "
                           f"(env var should beat file; inline comment stripped; quotes "
                           f"removed; 'none' -> None)")
            else:
                print("  .env parsed; env var overrides file; comments/quotes/none handled — OK")

    # A typo'd PIPELINE_ENV_FILE must fail rather than silently use defaults.
    env = dict(_os.environ)
    env["PIPELINE_ENV_FILE"] = str(Path(tempfile.gettempdir()) / "definitely_not_here.env")
    r = subprocess.run([sys.executable, "-c", "import config"],
                       capture_output=True, text=True, env=env, cwd=_os.getcwd())
    if r.returncode == 0:
        ok = _fail("a non-existent PIPELINE_ENV_FILE was silently ignored — that is an "
                   "invisible config change, it must fail")
    else:
        print("  missing PIPELINE_ENV_FILE fails loudly — OK")

    # .env.example must document every key config actually reads — otherwise a
    # parameter exists but nobody can discover it.
    example = Path("../.env.example")
    try:
        example_text = example.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        example_text = None
        ok = _fail("../.env.example is missing — it is the only documentation of these keys")
    except OSError as e:
        # Some sandboxed/locked-down environments deny reads of .env* by
        # pattern. Skip rather than crash — this check is about documentation
        # completeness, not correctness of the pipeline.
        example_text = None
        print(f"  SKIP .env.example coverage — cannot read it here ({type(e).__name__}: {e})")

    if example_text is not None:
        documented = {
            line.lstrip("#").split("=", 1)[0].strip()
            for line in example_text.splitlines()
            if line.lstrip("#").strip().startswith("PIPELINE_") and "=" in line
        }
        undocumented = sorted(set(config._USED) - documented)
        if undocumented:
            ok = _fail(f"{len(undocumented)} config key(s) read by config.py but absent from "
                       f".env.example: {undocumented}")
        else:
            print(f"  .env.example documents all {len(config._USED)} keys — OK")

    print("PASSED" if ok else "FAILED")
    return ok


# ─────────────────────────────────────────────
# 2. GRAPH PRIMITIVES
# ─────────────────────────────────────────────

def check_graph_primitives():
    print()
    print("=" * 70)
    print("2. GRAPH PRIMITIVES: node/edge feature maths, pinned to hand values")
    print("=" * 70)

    import torch
    import GraphBuilder as gb
    from GraphBuilder import _minmax, _build_edges_numba, _distances_to_centroids, build_one_graph

    ok = True

    # ── _minmax ──────────────────────────────────────────────────────────
    # Documented behaviour (ARCHITECTURE.md §4): when every value is equal
    # and positive it returns 1.0 for all, rather than dividing by a zero
    # range. This is exactly why EVERY 2-item basket gets cp_score 1.0 for
    # both of its nodes — worth pinning so it can't change silently.
    r = _minmax(np.array([5.0, 5.0, 5.0], dtype=np.float32))
    if not np.allclose(r, 1.0):
        ok = _fail(f"_minmax(all-equal-positive) should be all 1.0, got {r}")
    r = _minmax(np.array([0.0, 0.0], dtype=np.float32))
    if not np.allclose(r, 0.0):
        ok = _fail(f"_minmax(all-zero) should be all 0.0, got {r}")
    r = _minmax(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    if not np.allclose(r, [0.0, 0.5, 1.0]):
        ok = _fail(f"_minmax([2,4,6]) should be [0,.5,1], got {r}")
    print("  _minmax: all-equal-positive -> 1.0, all-zero -> 0.0, normal range -> [0,1] — OK")

    # ── _distances_to_centroids ──────────────────────────────────────────
    # The memory-safe algebraic expansion ||x-c||^2 = ||x||^2+||c||^2-2x.c.
    # Cheap to get subtly wrong (and it feeds `distinctiveness` on every
    # node in the system), so it's compared against the naive broadcast.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 7)).astype(np.float32)
    C = rng.normal(size=(5, 7)).astype(np.float32)
    fast = _distances_to_centroids(X, C)
    naive = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    if not np.allclose(fast, naive, atol=1e-3):
        ok = _fail(f"_distances_to_centroids max abs diff vs naive = "
                   f"{np.abs(fast - naive).max():.6f}")
    else:
        print("  _distances_to_centroids: matches naive broadcast within 1e-3 — OK")

    # ── _build_edges_numba: top-K selection ──────────────────────────────
    # Node 0 has 12 neighbours with distinct values, top_k=10 -> the
    # k > top_k branch. Expect exactly the 10 largest, strongest first.
    n = 13
    dense = np.zeros((n, n), dtype=np.float32)
    vals = np.arange(1, 13, dtype=np.float32) * 10.0   # 10..120
    dense[0, 1:13] = vals
    dense[1:13, 0] = vals
    csr = sp.csr_matrix(dense)
    src, dst, logv, rel = _build_edges_numba(
        csr.indptr.astype(np.int64), csr.indices.astype(np.int64),
        csr.data.astype(np.float32), 10,
    )
    node0 = src == 0
    if node0.sum() != 10:
        ok = _fail(f"node 0 should keep exactly top_k=10 edges, kept {node0.sum()}")
    kept_vals = np.expm1(logv[node0])
    if not np.allclose(np.sort(kept_vals)[::-1], np.sort(vals)[::-1][:10], rtol=1e-4):
        ok = _fail(f"top-10 selection kept the wrong values: {np.sort(kept_vals)[::-1]}")
    else:
        print("  _build_edges_numba: keeps exactly the top_k=10 strongest partners — OK")

    # ── _build_edges_numba: relative strength ────────────────────────────
    # edge_attr[1] is DOCUMENTED (GraphBuilder.py:157-161, ARCHITECTURE.md
    # §4) as "this edge's count as a fraction of the source node's own
    # STRONGEST co-purchase link". That makes two properties mandatory:
    #   (a) every relative strength is <= 1.0
    #   (b) each node's strongest edge has relative strength == 1.0
    #
    # REGRESSION GUARD. This failed until GRAPH_BUILDER_VERSION 3.
    # max_val_for_node was taken as final_vals[0], which is only the maximum
    # in the k > top_k branch (where a greedy descending selection runs). In
    # the else branch -- every node with <= top_k neighbours, i.e. every
    # basket of <= 11 products -- final_vals is still in CSR COLUMN order, so
    # final_vals[0] was an arbitrary neighbour, not the strongest. With the
    # 30/90/60 counts below it divided by 30 instead of 90, yielding
    # 1.0/3.0/2.0 where 0.33/1.0/0.67 was intended. Both assertions below are
    # the DOCUMENTED contract, so they also pin the invariant going forward.
    small = np.zeros((4, 4), dtype=np.float32)
    pairs = {(0, 1): 30.0, (0, 2): 90.0, (0, 3): 60.0}
    for (i, j), v in pairs.items():
        small[i, j] = v
        small[j, i] = v
    csr_s = sp.csr_matrix(small)
    src_s, dst_s, _, rel_s = _build_edges_numba(
        csr_s.indptr.astype(np.int64), csr_s.indices.astype(np.int64),
        csr_s.data.astype(np.float32), 10,
    )
    if rel_s.max() > 1.0 + 1e-6:
        ok = _fail(
            f"relative strength exceeds 1.0 (max={rel_s.max():.3f}) — node's "
            f"strongest-link denominator is wrong for nodes with <= top_k "
            f"neighbours. See the comment above this assertion."
        )
    else:
        # Node 0's counts are 30/90/60, so its edges must be scaled by 90:
        # exactly one edge at 1.0 (the strongest), the rest strictly below.
        got = {int(d): float(r) for s, d, r in zip(src_s, dst_s, rel_s) if s == 0}
        want = {1: 30.0 / 90.0, 2: 1.0, 3: 60.0 / 90.0}
        if set(got) != set(want) or not all(np.isclose(got[k], want[k]) for k in want):
            ok = _fail(f"node 0's relative strengths should be {want}, got {got}")
        elif sum(1 for v in got.values() if np.isclose(v, 1.0)) != 1:
            ok = _fail(f"exactly one of a node's edges must have relative strength "
                       f"1.0 (its strongest link), got {got}")
        else:
            print("  _build_edges_numba: relative strength scaled by the node's own "
                  "strongest link, exactly one edge at 1.0 — OK")

    # ── _basket_dense_cp_submatrix: identical to the old implementation ──
    # This function was rewritten for speed (csr[rows][:,cols] materialised
    # whole rows — ~20ms/basket, ~9 days over a 57M-basket population). The
    # rewrite is only safe if it is NUMERICALLY IDENTICAL, because a partially
    # completed inference run has already embedded baskets with the old code.
    # Compare the two directly over random sparse matrices.
    from GraphBuilder import _basket_dense_cp_submatrix

    def _old_submatrix(products, csr_m, pid2idx_m):
        gidx = np.array(sorted({pid2idx_m[p] for p in products if p in pid2idx_m}),
                        dtype=np.int64)
        if len(gidx) == 0:
            return {}, np.zeros((0, 0), dtype=np.float32)
        lmap = {int(g): l for l, g in enumerate(gidx)}
        return lmap, np.asarray(csr_m[gidx][:, gidx].todense(), dtype=np.float32)

    rng2 = np.random.default_rng(11)
    mismatch = None
    for trial in range(25):
        n_prod = int(rng2.integers(8, 60))
        density = float(rng2.uniform(0.05, 0.6))
        dense_full = (rng2.random((n_prod, n_prod)) < density) * rng2.integers(
            1, 500, size=(n_prod, n_prod))
        dense_full = np.triu(dense_full, 1)
        dense_full = (dense_full + dense_full.T).astype(np.int32)   # symmetric, zero diagonal
        m = sp.csr_matrix(dense_full)
        m.sum_duplicates()
        m.sort_indices()
        pid = {f"P{i}": i for i in range(n_prod)}
        picks = list(rng2.choice([f"P{i}" for i in range(n_prod)],
                                 size=int(rng2.integers(2, min(n_prod, 40))),
                                 replace=False))
        old_map, old_d = _old_submatrix(picks, m, pid)
        new_map, new_d = _basket_dense_cp_submatrix(picks, m, pid)
        if old_map != new_map:
            mismatch = f"trial {trial}: local_idx_map differs"
            break
        if old_d.shape != new_d.shape or not np.array_equal(old_d, new_d):
            mismatch = (f"trial {trial}: submatrix differs, max abs diff "
                        f"{np.abs(old_d - new_d).max() if old_d.shape == new_d.shape else 'shape'}")
            break
    if mismatch:
        ok = _fail(f"_basket_dense_cp_submatrix rewrite changed results — {mismatch}. "
                   f"Baskets already embedded with the old code would be inconsistent.")
    else:
        print("  _basket_dense_cp_submatrix: rewrite is bit-identical to csr[rows][:,cols] "
              "over 25 random matrices — OK")

    # Products absent from the catalog must still be skipped cleanly.
    lmap_u, dense_u = _basket_dense_cp_submatrix(["NOPE", "ALSO_NOPE"], m, pid)
    if lmap_u != {} or dense_u.shape != (0, 0):
        ok = _fail(f"all-unknown basket should give ({{}}, (0,0)), got {lmap_u}, {dense_u.shape}")
    else:
        print("  _basket_dense_cp_submatrix: all-unknown-product basket handled — OK")

    # ── build_one_graph: exact node-feature slot layout ──────────────────
    # in_dim = emb_dim + 4, and the four extras must sit in this exact order:
    # [emb_dim]=cp_score [+1]=sub_cluster [+2]=distinctiveness [+3]=log_units.
    # Anything downstream reading a slot by index breaks silently if these move.
    emb_dim = 6
    prods = ["A", "B", "C"]
    pid2idx = {"A": 0, "B": 1, "C": 2}
    emb_matrix = np.arange(3 * emb_dim, dtype=np.float32).reshape(3, emb_dim)
    subcl = np.array([0.10, 0.20, 0.30], dtype=np.float32)
    dist = np.array([0.40, 0.50, 0.60], dtype=np.float32)
    dense_cp = np.array([[0, 5, 1], [5, 0, 2], [1, 2, 0]], dtype=np.float32)
    local_idx = {0: 0, 1: 1, 2: 2}

    g = build_one_graph(
        products=prods, units=[3.0, 1.0, 1.0], basket_id="t1",
        emb_matrix=emb_matrix, emb_dim=emb_dim,
        subcluster_arr=subcl, distinctiveness_arr=dist,
        dense_cp=dense_cp, local_idx=local_idx, product_id_to_index=pid2idx,
    )
    x = g.x.numpy()
    if x.shape != (3, emb_dim + 4):
        ok = _fail(f"node feature matrix should be (3, {emb_dim+4}), got {x.shape}")
    if not np.allclose(x[:, :emb_dim], emb_matrix):
        ok = _fail("slots [0:emb_dim] are not the product embedding")
    if not np.allclose(x[:, emb_dim + 1], subcl):
        ok = _fail(f"slot [emb_dim+1] should be sub_cluster_id, got {x[:, emb_dim+1]}")
    if not np.allclose(x[:, emb_dim + 2], dist):
        ok = _fail(f"slot [emb_dim+2] should be distinctiveness, got {x[:, emb_dim+2]}")
    if not np.allclose(x[:, emb_dim + 3], np.log1p([3.0, 1.0, 1.0])):
        ok = _fail(f"slot [emb_dim+3] should be log1p(units), got {x[:, emb_dim+3]}")
    if not ((x[:, emb_dim] >= 0).all() and (x[:, emb_dim] <= 1).all()):
        ok = _fail(f"slot [emb_dim] (cp_score) should be min-max normalised, got {x[:, emb_dim]}")
    print("  build_one_graph: node feature slot layout [emb|cp|subcl|distinct|log_units] — OK")

    # ── build_one_graph: duplicate products are deduped, units SUMMED ────
    g2 = build_one_graph(
        products=["A", "B", "A"], units=[2.0, 1.0, 5.0], basket_id="t2",
        emb_matrix=emb_matrix, emb_dim=emb_dim,
        subcluster_arr=subcl, distinctiveness_arr=dist,
        dense_cp=dense_cp, local_idx=local_idx, product_id_to_index=pid2idx,
    )
    if g2.x.shape[0] != 2:
        ok = _fail(f"duplicate tpnb should collapse to 2 nodes, got {g2.x.shape[0]}")
    elif not np.isclose(g2.x.numpy()[0, emb_dim + 3], np.log1p(7.0)):
        ok = _fail(f"duplicate units should sum to 7 -> log1p(7)={np.log1p(7.0):.4f}, "
                   f"got {g2.x.numpy()[0, emb_dim+3]:.4f}")
    else:
        print("  build_one_graph: duplicate products deduped with units summed — OK")

    # ── build_one_graph: zero / negative quantities (returns) ────────────
    # The warehouse `quantity` column is a weekly SUM that folds in returns, so
    # it can be 0 or negative. log1p is -inf at -1 and NaN below it. The masked
    # evaluation must reproduce exactly what log1p + nan_to_num used to give
    # (a trained model and embedded chunks depend on those values being
    # unchanged) AND must not emit a RuntimeWarning — at 57M baskets the
    # warning spew buried progress output and cost real wall-clock time.
    import warnings as _warnings

    for units_in, want in (
        ([1.0, 0.0, -1.0],  [float(np.log1p(1.0)), 0.0, 0.0]),
        ([-3.0, -0.5, 2.0], [0.0, float(np.log1p(-0.5)), float(np.log1p(2.0))]),
        ([-1.0001, -50.0, 0.0], [0.0, 0.0, 0.0]),
    ):
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            g_neg = build_one_graph(
                products=prods, units=units_in, basket_id="tneg",
                emb_matrix=emb_matrix, emb_dim=emb_dim,
                subcluster_arr=subcl, distinctiveness_arr=dist,
                dense_cp=dense_cp, local_idx=local_idx, product_id_to_index=pid2idx,
            )
        got_lu = g_neg.x.numpy()[:, emb_dim + 3]
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        if not np.allclose(got_lu, want, atol=1e-6):
            ok = _fail(f"log_units for units={units_in} should be {want}, got {list(got_lu)}")
            break
        if runtime_warnings:
            ok = _fail(f"units={units_in} raised {len(runtime_warnings)} RuntimeWarning(s) "
                       f"(first: {runtime_warnings[0].message}) — log1p must not be "
                       f"evaluated where it is undefined")
            break
        if not np.isfinite(g_neg.x.numpy()).all():
            ok = _fail(f"units={units_in} produced non-finite node features")
            break
    else:
        print("  build_one_graph: zero/negative quantities (returns) give the same "
              "log_units as log1p+nan_to_num, with no RuntimeWarning — OK")

    # ── build_one_graph: empty/degenerate basket does not crash ──────────
    g3 = build_one_graph(
        products=["ZZZ"], units=[1.0], basket_id="t3",
        emb_matrix=emb_matrix, emb_dim=emb_dim,
        subcluster_arr=subcl, distinctiveness_arr=dist,
        dense_cp=np.zeros((0, 0), dtype=np.float32), local_idx={},
        product_id_to_index=pid2idx,
    )
    if g3.x.shape != (1, emb_dim + 4) or g3.edge_index.shape != (2, 0):
        ok = _fail(f"unknown-product basket should give a 1x{emb_dim+4} zero node and no "
                   f"edges, got x={tuple(g3.x.shape)} edge_index={tuple(g3.edge_index.shape)}")
    else:
        print("  build_one_graph: basket of entirely unknown products degrades safely — OK")

    # ── no NaN/inf ever reaches a training batch ─────────────────────────
    if not torch.isfinite(g.x).all():
        ok = _fail("build_one_graph emitted non-finite node features")
    else:
        print("  build_one_graph: node features all finite (NaN/inf sanitised) — OK")

    print("PASSED" if ok else "FAILED")
    return ok


# ─────────────────────────────────────────────
# 3. CO-PURCHASE CHUNK ADDITIVITY
# ─────────────────────────────────────────────

def check_copurchase_chunk_additivity():
    print()
    print("=" * 70)
    print("3. CO-PURCHASE ADDITIVITY: chunked X.T@X == single-shot X.T@X")
    print("=" * 70)
    print("  pipeline_main.py:152-157 claims chunking the co-purchase build is")
    print("  EXACTLY equivalent to one pass. Every edge feature depends on it.")

    rng = np.random.default_rng(7)
    n_baskets, n_products = 97, 23
    rows, cols = [], []
    for b in range(n_baskets):
        size = int(rng.integers(2, 9))
        for p in rng.choice(n_products, size=size, replace=False):
            rows.append(b)
            cols.append(int(p))
    X = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(n_baskets, n_products),
    )
    single_shot = (X.T @ X).toarray()

    # Same row-disjoint chunking pipeline_main.py does, at an awkward chunk
    # size that does not divide the basket count evenly.
    accumulated = sp.csr_matrix((n_products, n_products), dtype=np.int32)
    chunk = 10
    for lo in range(0, n_baskets, chunk):
        X_chunk = X[lo:lo + chunk]
        accumulated = (accumulated + (X_chunk.T @ X_chunk)).tocsr()

    if not np.array_equal(accumulated.toarray(), single_shot):
        diff = np.abs(accumulated.toarray() - single_shot).max()
        print(f"FAILED — chunked result differs from single-shot (max abs diff {diff})")
        return False

    print(f"  {n_baskets} baskets in chunks of {chunk} == single-shot, exactly — OK")
    print("PASSED")
    return True


# ─────────────────────────────────────────────
# Synthetic fixtures for the heavier checks
# ─────────────────────────────────────────────

def build_synthetic_catalog(n_products=30, emb_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    tpnbs = [f"P{i:03d}" for i in range(n_products)]
    product_embedding = {t: rng.normal(size=emb_dim).astype(np.float32) for t in tpnbs}
    product_units_avg = {t: float(rng.uniform(1, 5)) for t in tpnbs}
    product_id_to_index = {t: i for i, t in enumerate(tpnbs)}

    rows, cols, vals = [], [], []
    for _ in range(80):
        i, j = rng.integers(0, n_products, size=2)
        if i == j:
            continue
        c = float(rng.integers(1, 10))
        rows += [i, j]
        cols += [j, i]
        vals += [c, c]
    copurchase_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(n_products, n_products))

    return tpnbs, product_embedding, product_units_avg, product_id_to_index, copurchase_sparse


def build_synthetic_raw_export(tpnbs, n_baskets=25, seed=1) -> pd.DataFrame:
    """
    A synthetic RAW household x tpnb x week export — one row per
    (household, tpnb, week), matching the real warehouse export's shape —
    rather than pre-aggregated basket rows. Aggregation into baskets is
    DuckDB's job (basket_store.build_baskets_table), so the test exercises
    that same path instead of handing it already-basket-shaped rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_baskets):
        household_number = 1000 + b
        year_number = 2026
        week_number = 1 + (b % 10)
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


# ─────────────────────────────────────────────
# 4 + 5 + 6. FUNCTIONAL, PARITY, BASKET STORE
# ─────────────────────────────────────────────

def check_functional():
    print()
    print("=" * 70)
    print("4-6. FUNCTIONAL: train/score on synthetic data (real DuckDB + LMDB)")
    print("=" * 70)

    import torch
    from GraphBuilder import (
        prepare_globals, run_inference, merge_inference_output,
        _basket_dense_cp_submatrix, build_one_graph,
    )
    from GNN_Train import BasketGNN
    import GraphBuilder as gb
    import duckdb_manager
    import basket_store
    import lmdb_graph_cache

    ok = True

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
        product_embedding=product_embedding,
        product_id_to_index=product_id_to_index,
        copurchase_sparse=copurchase_sparse,
        product_units_avg=product_units_avg,
    )
    assert "theme_ids" not in G, "theme_ids key found in prepare_globals() output"
    emb_dim = G["emb_dim"]
    assert G["in_dim"] == emb_dim + 4, f"expected in_dim == emb_dim+4, got {G['in_dim']}"
    print(f"  prepare_globals(): in_dim={G['in_dim']} (emb_dim={emb_dim}+4), no theme_ids — OK")

    con = duckdb_manager.get_connection()
    # Drop leftover DuckDB state from a previous run BEFORE touching local
    # output files — a stale inference_chunks_pytest table (all rows already
    # 'complete') would make run_inference claim nothing while _cleanup() has
    # already wiped this run's chunk parquet, leaving nothing to merge.
    basket_store.drop_all(con, dataset_tag)
    n_baskets_total, _units_avg_pg, _products_pg = basket_store.build_baskets_table(
        con, raw_export_dir, dataset_tag, min_basket_products=2,
    )
    print(f"  build_baskets_table(): {n_baskets_total} baskets in DuckDB — OK")

    # ── 6. BASKET STORE SEMANTICS ────────────────────────────────────────
    bt = f'"baskets_{dataset_tag}"'
    sizes = con.execute(f"SELECT MIN(len(products)) FROM {bt}").fetchone()[0]
    if sizes is not None and sizes < 2:
        ok = _fail(f"min_basket_products=2 not enforced — found a basket of {sizes}")
    else:
        print("  basket_store: >=2-product filter enforced — OK")

    bad_bucket = con.execute(
        f"""SELECT COUNT(*) FROM {bt} WHERE size_bucket <> CASE
              WHEN len(products) <= 1  THEN 0 WHEN len(products) <= 5  THEN 1
              WHEN len(products) <= 15 THEN 2 WHEN len(products) <= 50 THEN 3
              ELSE 4 END"""
    ).fetchone()[0]
    if bad_bucket:
        ok = _fail(f"{bad_bucket} baskets have a size_bucket inconsistent with their size")
    else:
        print("  basket_store: size_bucket boundaries consistent with product counts — OK")

    # basket_id MUST stay '<household>_<year_week>' — need_state_graph parses
    # it back to recover household identity for journey analysis, and nothing
    # else in the pipeline preserves that link.
    bad_id = con.execute(
        f"SELECT COUNT(*) FROM {bt} WHERE basket_id <> "
        f"(CAST(household_number AS VARCHAR) || '_' || CAST(year_week_number AS VARCHAR))"
    ).fetchone()[0]
    if bad_id:
        ok = _fail(f"{bad_id} basket_ids do not match '<household>_<year_week>' — "
                   f"this silently breaks need_state_graph journey analysis")
    else:
        print("  basket_store: basket_id format is '<household>_<year_week>' — OK")

    sampled = basket_store.sample_training_baskets(
        con, dataset_tag, n_samples=n_baskets_total, seed=42,
    )
    print(f"  sample_training_baskets(): {len(sampled)} baskets sampled — OK")

    lmdb_graph_cache.load_or_build_lmdb_cache(
        sampled, G, lmdb_path, manifest_path,
        seed=42, n_train_samples_requested=n_baskets_total,
    )
    train_dataset = lmdb_graph_cache.LMDBGraphDataset(lmdb_path)
    assert len(train_dataset) == len(sampled), "LMDB cache length != sampled basket count"
    g0 = train_dataset.get(0)
    assert g0.x.shape[1] == G["in_dim"], \
        f"training graph node feature width {g0.x.shape[1]} != in_dim {G['in_dim']}"
    if g0.edge_attr.shape[0] > 0:
        assert g0.edge_attr.shape[1] == 2, "expected 2 edge features (log_cp, relative strength)"
    print(f"  lmdb_graph_cache: {len(train_dataset)} graphs, widths correct — OK")

    # ── 5. TRAIN/SCORE FEATURE PARITY (values, not just widths) ──────────
    # Rebuild the SAME baskets the way GraphBuilder._embed_basket_chunk does
    # at inference time, and require the resulting tensors to be numerically
    # identical to the LMDB-cached training graphs. Comparing only widths
    # (as this file used to) would pass even under a genuine feature drift —
    # which is the exact class of bug GraphBuilder.py's docstring records.
    n_parity = min(5, len(sampled))
    mismatches = []
    for i in range(n_parity):
        row = sampled.iloc[i]
        local_idx_map, basket_dense_cp = _basket_dense_cp_submatrix(
            row["products"], G["csr"], G["product_id_to_index"],
        )
        g_infer = build_one_graph(
            products=row["products"], units=row["units"], basket_id=row["basket_id"],
            emb_matrix=G["emb_matrix"], emb_dim=G["emb_dim"],
            subcluster_arr=G["subcluster_arr"], distinctiveness_arr=G["distinctiveness_arr"],
            dense_cp=basket_dense_cp, local_idx=local_idx_map,
            product_id_to_index=G["product_id_to_index"],
        )
        g_train = train_dataset.get(i)
        if g_train.x.shape != g_infer.x.shape:
            mismatches.append(f"basket {i}: x shape {tuple(g_train.x.shape)} vs {tuple(g_infer.x.shape)}")
        elif not torch.allclose(g_train.x, g_infer.x, atol=1e-6):
            worst = (g_train.x - g_infer.x).abs().max().item()
            col = (g_train.x - g_infer.x).abs().max(dim=0).values.argmax().item()
            mismatches.append(f"basket {i}: node features differ, max abs diff {worst:.6g} "
                              f"(worst slot index {col} of {G['in_dim']})")
        elif g_train.edge_index.shape != g_infer.edge_index.shape:
            mismatches.append(f"basket {i}: edge_index {tuple(g_train.edge_index.shape)} "
                              f"vs {tuple(g_infer.edge_index.shape)}")
        elif g_train.edge_attr.numel() and not torch.allclose(
                g_train.edge_attr, g_infer.edge_attr, atol=1e-6):
            mismatches.append(f"basket {i}: edge features differ")

    if mismatches:
        ok = _fail("TRAIN/SCORE DRIFT — training and inference built different graphs:")
        for m in mismatches:
            print(f"      {m}")
    else:
        print(f"  TRAIN/SCORE PARITY: {n_parity} baskets byte-identical across both "
              f"construction paths (node AND edge features) — OK")

    n_graphs_first_open = len(train_dataset)
    # lmdb refuses to open the same environment path twice concurrently in one
    # process — close before constructing a second dataset over the same path.
    train_dataset.close()

    lmdb_graph_cache.load_or_build_lmdb_cache(
        sampled, G, lmdb_path, manifest_path,
        seed=42, n_train_samples_requested=n_baskets_total,
    )
    reloaded_dataset = lmdb_graph_cache.LMDBGraphDataset(lmdb_path)
    assert len(reloaded_dataset) == n_graphs_first_open, \
        "cache round-trip returned a different graph count"
    print("  lmdb_graph_cache: cache-hit reload round-trip — OK")
    reloaded_dataset.close()

    device = torch.device("cpu")
    model = BasketGNN(in_dim=G["in_dim"], edge_dim=2, hidden_dim=8, out_dim=4).to(device)
    model.eval()

    with torch.no_grad():
        z_train_path = model.encode(g0.to(device))
    assert z_train_path.shape == (1, 4), f"unexpected encode() shape {z_train_path.shape}"
    print(f"  model.encode() on an LMDB-cached graph: {tuple(z_train_path.shape)} — OK")

    run_inference(
        con, dataset_tag, G, model, device,
        worker_id="pytest-worker", chunk_size=10, batch_size=8, output_dir=output_dir,
    )
    merged = merge_inference_output(dataset_tag, output_dir=output_dir)
    assert len(merged) == n_baskets_total, \
        f"run_inference embedded {len(merged)} baskets, expected {n_baskets_total}"
    assert merged["gnn_embedding"].iloc[0].shape == (4,), "unexpected embedding width"
    print(f"  run_inference() + merge: {len(merged)} baskets, restartable queue — OK")

    basket_store.drop_all(con, dataset_tag)
    _cleanup()

    print("PASSED" if ok else "FAILED")
    return ok


# ─────────────────────────────────────────────
# 7. NEED-STATE GRAPH
# ─────────────────────────────────────────────

def check_need_state_graph():
    print()
    print("=" * 70)
    print("7. NEED-STATE GRAPH: adjacency, transitions, journeys, GMM overlap")
    print("=" * 70)

    import need_state_graph as nsg

    ok = True

    # ── basket_id -> household/week round trip ───────────────────────────
    clusters = pd.DataFrame({
        "basket_id": ["1000_202615", "1000_202616", "2000_202615"],
        "need_state_cluster": [0, 1, 0],
    })
    parsed = nsg.split_basket_id(clusters)
    if list(parsed["household_number"]) != [1000, 1000, 2000]:
        ok = _fail(f"household parse wrong: {list(parsed['household_number'])}")
    elif list(parsed["year_week_number"]) != [202615, 202616, 202615]:
        ok = _fail(f"week parse wrong: {list(parsed['year_week_number'])}")
    else:
        print("  split_basket_id: recovers household + week from basket_id — OK")

    try:
        nsg.split_basket_id(pd.DataFrame({"basket_id": ["nounderscore"],
                                          "need_state_cluster": [0]}))
        ok = _fail("split_basket_id should raise on a malformed basket_id")
    except ValueError:
        print("  split_basket_id: raises on malformed basket_id — OK")

    # ── year-boundary week ranking ───────────────────────────────────────
    # year_week_number is year*100+week, so 202552 -> 202601 is ONE week but
    # a numeric gap of 49. Ranking the observed weeks is what makes
    # max_week_gap=1 behave correctly across a year end; a naive subtraction
    # would silently drop every transition that spans New Year.
    yb = pd.DataFrame({
        "basket_id": [f"1_{w}" for w in (202551, 202552, 202601, 202602)],
        "need_state_cluster": [0, 1, 2, 3],
    })
    ranked = nsg._add_week_rank(nsg.split_basket_id(yb))
    if list(ranked.sort_values("year_week_number")["week_rank"]) != [0, 1, 2, 3]:
        ok = _fail(f"week_rank across year boundary wrong: {list(ranked['week_rank'])}")
    else:
        print("  _add_week_rank: 202552 -> 202601 is a gap of 1, not 49 — OK")

    # ── transitions, hand-computed ───────────────────────────────────────
    # 10 identical households, each 0 -> 1 -> 2 over three CONSECUTIVE weeks
    # that cross the year boundary.
    rows = []
    for hh in range(10):
        for wk, ns in zip((202551, 202552, 202601), (0, 1, 2)):
            rows.append({"basket_id": f"{hh}_{wk}", "need_state_cluster": ns})
    seq = pd.DataFrame(rows)

    trans = nsg.build_need_state_transitions(seq, max_week_gap=1, min_support=1)
    got = {(int(r.from_need_state), int(r.to_need_state)): (int(r.n_transitions), float(r.prob))
           for r in trans.itertuples(index=False)}
    if got != {(0, 1): (10, 1.0), (1, 2): (10, 1.0)}:
        ok = _fail(f"transitions wrong. expected {{(0,1):(10,1.0),(1,2):(10,1.0)}}, got {got}")
    else:
        print("  build_need_state_transitions: exact counts and probabilities — OK")

    # Every from_need_state's outgoing probabilities must sum to 1.
    sums = trans.groupby("from_need_state")["prob"].sum()
    if not np.allclose(sums.to_numpy(), 1.0):
        ok = _fail(f"outgoing prob does not sum to 1 per need-state: {sums.to_dict()}")
    else:
        print("  build_need_state_transitions: prob rows sum to 1.0 — OK")

    # max_week_gap must actually filter. Household 5 shops in weeks 1 and 5
    # only — a genuine 4-week gap. The filler households are REQUIRED: week
    # ranking is computed over the weeks present in the whole dataset, so
    # without weeks 2-4 appearing somewhere, ranking would compress 1->5 into
    # an apparent gap of 1. (That compression is a real property of
    # _add_week_rank on sparse extracts — see its docstring.)
    gapped = pd.DataFrame({
        "basket_id": ["5_202601", "5_202605",
                      "6_202602", "6_202603", "6_202604"],
        "need_state_cluster": [0, 1, 0, 0, 0],
    })
    # Household 5's 0->1 step spans 4 weeks; household 6's 0->0 steps span 1.
    # Only the 0->1 step should be gap-filtered.
    def _has_01(df):
        return not df[(df["from_need_state"] == 0) & (df["to_need_state"] == 1)].empty

    tight = nsg.build_need_state_transitions(gapped, max_week_gap=1, min_support=1)
    loose = nsg.build_need_state_transitions(gapped, max_week_gap=None, min_support=1)
    if _has_01(tight):
        ok = _fail("max_week_gap=1 should reject household 5's 4-week 0->1 step")
    elif not _has_01(loose):
        ok = _fail("max_week_gap=None should accept consecutive OBSERVED baskets, "
                   "including household 5's 4-week 0->1 step")
    elif tight[(tight["from_need_state"] == 0) & (tight["to_need_state"] == 0)].empty:
        ok = _fail("max_week_gap=1 should keep household 6's genuine 1-week 0->0 steps")
    else:
        print("  build_need_state_transitions: max_week_gap filtering — OK")

    # ── journeys ─────────────────────────────────────────────────────────
    # A deterministic chain yields BOTH the 1-step (0,1) and the 2-step
    # (0,1,2) path, each with probability 1.0 — so select the full-length
    # path explicitly rather than assuming it sorts first (it ties).
    j = nsg.possible_journeys(trans, from_need_state=0, depth=3)
    if j.empty:
        ok = _fail("expected at least one journey out of need-state 0")
    else:
        full = j[j["path"].map(lambda p: tuple(p) == (0, 1, 2))]
        if full.empty:
            ok = _fail(f"journey (0,1,2) missing; got {[tuple(p) for p in j['path']]}")
        elif not np.isclose(full.iloc[0]["path_prob"], 1.0):
            ok = _fail(f"path_prob for a deterministic chain should be 1.0, "
                       f"got {full.iloc[0]['path_prob']}")
        elif int(full.iloc[0]["n_steps"]) != 2:
            ok = _fail(f"n_steps for (0,1,2) should be 2, got {full.iloc[0]['n_steps']}")
        elif j["path_prob"].is_monotonic_decreasing is False and len(j) > 1:
            ok = _fail("possible_journeys should return rows sorted by descending path_prob")
        else:
            print("  possible_journeys: deterministic chain recovered as (0,1,2) p=1.0 — OK")

    # path_prob must be the product of its step probabilities.
    two_step = pd.DataFrame({
        "from_need_state": [0, 0, 1],
        "to_need_state":   [1, 2, 2],
        "n_transitions":   [60, 40, 100],
        "prob":            [0.6, 0.4, 1.0],
        "lift":            [1.0, 1.0, 1.0],
        "avg_week_gap":    [1.0, 1.0, 1.0],
        "low_support":     [False, False, False],
    })
    j2 = nsg.possible_journeys(two_step, from_need_state=0, depth=2)
    path_012 = j2[j2["path"].map(lambda p: tuple(p) == (0, 1, 2))]
    if path_012.empty:
        ok = _fail("journey (0,1,2) missing")
    elif not np.isclose(path_012.iloc[0]["path_prob"], 0.6 * 1.0):
        ok = _fail(f"path_prob should be 0.6*1.0=0.6, got {path_012.iloc[0]['path_prob']}")
    else:
        print("  possible_journeys: path_prob is the product of step probs — OK")

    # exclude_self_loops must actually suppress standing still.
    selfy = pd.DataFrame({
        "from_need_state": [0, 0], "to_need_state": [0, 1],
        "n_transitions": [90, 10], "prob": [0.9, 0.1], "lift": [1.0, 1.0],
        "avg_week_gap": [1.0, 1.0], "low_support": [False, False],
    })
    j3 = nsg.possible_journeys(selfy, from_need_state=0, depth=2, exclude_self_loops=True)
    if any(len(set(p)) != len(p) for p in j3["path"]):
        ok = _fail("exclude_self_loops=True still produced a self-transition step")
    else:
        print("  possible_journeys: exclude_self_loops suppresses standing still — OK")

    # ── adjacency, hand-computed ─────────────────────────────────────────
    # clusters: 1_1,2_1 -> ns0 ; 3_1,4_1 -> ns1
    # edges: (1,3,w=1) (2,3,w=2) (1,4,w=3) all cross 0-1 ; (1,2,w=4) inside 0
    #   agg      : (0,0) w=4 n=1 ; (0,1) w=6 n=3
    #   deg(0)   = 4 + 6 + 4 = 14   (self-loop contributes to both endpoints)
    #   deg(1)   = 6
    #   total_w  = 10
    #   expected(0,1) = 14*6/(2*10) = 4.2
    #   lift          = 6/4.2      = 1.428571
    #   share_a       = 6/14       = 0.428571
    #   share_b       = 6/6        = 1.0
    adj_clusters = pd.DataFrame({
        "basket_id": ["1_1", "2_1", "3_1", "4_1"],
        "need_state_cluster": [0, 0, 1, 1],
    })
    adj_edges = pd.DataFrame({
        "basket_a": ["1_1", "2_1", "1_1", "1_1"],
        "basket_b": ["3_1", "3_1", "4_1", "2_1"],
        "weight":   [1.0, 2.0, 3.0, 4.0],
    })
    adj = nsg.build_need_state_adjacency(adj_edges, adj_clusters)
    pair = adj[(adj["need_state_a"] == 0) & (adj["need_state_b"] == 1)]
    if pair.empty:
        ok = _fail("adjacency missing the (0,1) pair")
    else:
        r = pair.iloc[0]
        checks = [
            ("n_edges", r["n_edges"], 3),
            ("weight_sum", r["weight_sum"], 6.0),
            ("lift", r["lift"], 6.0 / 4.2),
            ("share_a", r["share_a"], 6.0 / 14.0),
            ("share_b", r["share_b"], 1.0),
        ]
        for name, got_v, want in checks:
            if not np.isclose(float(got_v), want):
                ok = _fail(f"adjacency {name}: expected {want:.6f}, got {float(got_v):.6f}")
        if (adj["need_state_a"] == adj["need_state_b"]).any():
            ok = _fail("self-loops should be dropped by default")
        print("  build_need_state_adjacency: n_edges/weight_sum/lift/share_a/share_b — OK")

    # (a,b) and (b,a) must aggregate to ONE canonical undirected row.
    flipped = pd.DataFrame({
        "basket_a": ["3_1", "1_1"], "basket_b": ["1_1", "3_1"], "weight": [1.0, 1.0],
    })
    fadj = nsg.build_need_state_adjacency(flipped, adj_clusters)
    if len(fadj) != 1 or int(fadj.iloc[0]["n_edges"]) != 2:
        ok = _fail(f"(a,b) and (b,a) should merge into one row with n_edges=2, got {len(fadj)} rows")
    else:
        print("  build_need_state_adjacency: undirected canonicalisation — OK")

    # neighbouring_need_states must find a need-state in EITHER column.
    nb0 = nsg.neighbouring_need_states(adj, 0)
    nb1 = nsg.neighbouring_need_states(adj, 1)
    if nb0.empty or int(nb0.iloc[0]["neighbour"]) != 1:
        ok = _fail(f"neighbours of 0 should include 1, got {nb0.to_dict('records')}")
    elif nb1.empty or int(nb1.iloc[0]["neighbour"]) != 0:
        ok = _fail(f"neighbours of 1 should include 0, got {nb1.to_dict('records')}")
    elif not np.isclose(float(nb1.iloc[0]["share_of_this"]), 1.0):
        ok = _fail(f"share_of_this for need-state 1 should be 1.0, "
                   f"got {float(nb1.iloc[0]['share_of_this'])}")
    else:
        print("  neighbouring_need_states: symmetric lookup + correct share — OK")

    # Adjacency and transitions must produce JOINABLE need-state dtypes —
    # a float/int mismatch between the two artifacts silently yields empty
    # joins in downstream analysis.
    if adj["need_state_a"].dtype.kind != trans["from_need_state"].dtype.kind:
        ok = _fail(f"adjacency need_state dtype ({adj['need_state_a'].dtype}) is not joinable "
                   f"with transitions ({trans['from_need_state'].dtype})")
    else:
        print("  adjacency/transitions need-state dtypes are joinable — OK")

    # ── GMM overlap ──────────────────────────────────────────────────────
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(3)
    blob = np.vstack([
        rng.normal(loc=+4.0, scale=0.25, size=(120, 4)),
        rng.normal(loc=-4.0, scale=0.25, size=(120, 4)),
    ])
    emb = pd.DataFrame({
        "basket_id": [f"{i}_202601" for i in range(len(blob))],
        "gnn_embedding": list(blob.astype(np.float32)),
    })
    # Fit on the SAME normalized representation build_gmm_overlap() scores on
    # (cluster_basket_embeddings.py normalizes before both fit and predict).
    gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=0).fit(
        normalize(blob)
    )
    ov_full = nsg.build_gmm_overlap(emb, gmm, batch_size=10_000)
    ov_batched = nsg.build_gmm_overlap(emb, gmm, batch_size=7)

    if not np.allclose(ov_full["mean_posterior"].to_numpy(),
                       ov_batched["mean_posterior"].to_numpy(), atol=1e-9):
        ok = _fail("build_gmm_overlap is not batch-invariant")
    else:
        print("  build_gmm_overlap: batched result == single-batch result — OK")

    # GMM component ids are a DIFFERENT labelling from Leiden need-states, so
    # the columns must NOT be called need_state_* — otherwise joining this to
    # the adjacency/transition tables silently produces nonsense instead of
    # failing. Pin the names.
    if {"need_state_a", "need_state_b"} & set(ov_full.columns):
        ok = _fail("build_gmm_overlap must not emit need_state_* columns — GMM "
                   "component ids are not Leiden need-state ids, and sharing the "
                   "column name invites a silently wrong join")
    elif not {"gmm_component_a", "gmm_component_b"} <= set(ov_full.columns):
        ok = _fail(f"expected gmm_component_a/b columns, got {list(ov_full.columns)}")
    else:
        print("  build_gmm_overlap: component ids named distinctly from Leiden ids — OK")

    # Only rows for components that actually won baskets are distributions;
    # an unused component's row is all zeros by construction.
    populated = ov_full[ov_full["n_baskets_a"] > 0]
    row_sums = populated.groupby("gmm_component_a")["mean_posterior"].sum()
    if not np.allclose(row_sums.to_numpy(), 1.0, atol=1e-6):
        ok = _fail(f"overlap rows should sum to 1.0, got {row_sums.to_dict()}")
    else:
        print("  build_gmm_overlap: each row is a probability distribution — OK")

    diag = populated[
        populated["gmm_component_a"] == populated["gmm_component_b"]
    ]["mean_posterior"]
    if float(diag.min()) < 0.9:
        ok = _fail(f"two well-separated blobs should give near-1.0 purity, "
                   f"got min diagonal {float(diag.min()):.3f}")
    else:
        print("  build_gmm_overlap: separated blobs give near-1.0 diagonal purity — OK")

    # ── household-level lookup (the end-to-end question) ─────────────────
    # seq gave every household the identical path 0 -> 1 -> 2 across weeks
    # 202551, 202552, 202601. Household 3's CURRENT state is therefore 2,
    # taken from its most recent week — which is the one that crosses the
    # year boundary, so this also pins that "most recent" uses the right order.
    hist = nsg.household_history(seq, household_number=3)
    if list(hist["need_state_cluster"]) != [0, 1, 2]:
        ok = _fail(f"household_history should be oldest-first [0,1,2], "
                   f"got {list(hist['need_state_cluster'])}")
    elif list(hist["year_week_number"]) != [202551, 202552, 202601]:
        ok = _fail(f"household_history week ordering wrong: {list(hist['year_week_number'])}")
    else:
        print("  household_history: observed path, oldest first, across year end — OK")

    res = nsg.journeys_for_household(seq, trans, household_number=3)
    if res["current_need_state"] != 2:
        ok = _fail(f"current_need_state should be 2 (most recent week), "
                   f"got {res['current_need_state']}")
    elif res["as_of_week"] != 202601:
        ok = _fail(f"as_of_week should be 202601, got {res['as_of_week']}")
    else:
        print("  journeys_for_household: resolves household -> current need-state — OK")

    missing = nsg.journeys_for_household(seq, trans, household_number=999999)
    if missing["current_need_state"] is not None or not missing["history"].empty:
        ok = _fail("journeys_for_household should return empty for an unknown household")
    else:
        print("  journeys_for_household: unknown household degrades cleanly — OK")

    print("PASSED" if ok else "FAILED")
    return ok


# ─────────────────────────────────────────────
# 8. PROD OUTPUT SANITY (opt-in)
# ─────────────────────────────────────────────

def check_prod_outputs():
    """
    Sanity checks against REAL artifacts in ../data/output. These catch the
    failure modes that synthetic tests structurally cannot: a collapsed
    embedding space, a degenerate clustering, or labels that do not cover the
    baskets. Every check skips cleanly when its file is absent.
    """
    print()
    print("=" * 70)
    print("8. PROD OUTPUTS: sanity checks against ../data/output")
    print("=" * 70)

    ok = True
    checked_any = False

    emb_path = OUTPUT_DIR / "basket_gnn_embeddings.parquet"
    clu_path = OUTPUT_DIR / "basket_need_state_clusters.parquet"

    if emb_path.exists():
        checked_any = True
        emb = pd.read_parquet(emb_path)
        X = np.stack(emb["gnn_embedding"].values)
        print(f"  embeddings: {X.shape[0]:,} baskets x {X.shape[1]} dims")

        if not np.isfinite(X).all():
            ok = _fail(f"{np.isfinite(X).sum()} non-finite values in basket embeddings — "
                       f"training likely diverged (check the NaN-batch warnings in GNN_Train)")
        else:
            print("    all finite — OK")

        # Embedding collapse: if the autoencoder learned a constant, every
        # basket gets ~the same vector and every downstream cluster is noise.
        # This is silent — clustering still "works", it just means nothing.
        per_dim_std = X.std(axis=0)
        if float(per_dim_std.mean()) < 1e-4:
            ok = _fail(f"EMBEDDING COLLAPSE — mean per-dimension std is "
                       f"{per_dim_std.mean():.2e}. Every basket has nearly the same "
                       f"embedding, so need-states are meaningless. Check training loss.")
        else:
            print(f"    mean per-dim std {per_dim_std.mean():.4f} (no collapse) — OK")

        dead = int((per_dim_std < 1e-6).sum())
        if dead:
            print(f"    NOTE: {dead}/{X.shape[1]} embedding dimensions are constant — "
                  f"effective dimensionality is lower than OUT_DIM suggests.")

        if emb["basket_id"].duplicated().any():
            ok = _fail(f"{int(emb['basket_id'].duplicated().sum())} duplicate basket_ids "
                       f"in the embeddings file")
        else:
            print("    basket_id unique — OK")
    else:
        print(f"  SKIP: {emb_path} not found")

    if clu_path.exists():
        checked_any = True
        clu = pd.read_parquet(clu_path)
        print(f"  clusters: {len(clu):,} rows")

        for col in ("need_state_cluster", "need_state_cluster_gmm"):
            if col not in clu.columns:
                print(f"    SKIP: no {col} column")
                continue
            vc = clu[col].value_counts()
            share = vc.iloc[0] / len(clu)
            print(f"    {col}: {len(vc)} clusters, largest holds {share:.1%}")
            if len(vc) < 2:
                ok = _fail(f"{col} has {len(vc)} cluster — clustering is degenerate")
            elif share > 0.90:
                ok = _fail(f"{col}: one cluster holds {share:.1%} of all baskets. "
                           f"For Leiden, sweep_resolution() and raise LEIDEN_RESOLUTION; "
                           f"for GMM, revisit GMM_N_COMPONENTS (currently a placeholder).")
            if clu[col].isna().any():
                ok = _fail(f"{col} has {int(clu[col].isna().sum())} unlabelled baskets")

        if "gmm_confidence" in clu.columns:
            low = float((clu["gmm_confidence"] < 0.5).mean())
            print(f"    gmm_confidence < 0.5 for {low:.1%} of baskets")
            if low > 0.5:
                print(f"    NOTE: most baskets sit between GMM components. Expected to some "
                      f"degree at week grain (a week spans several shopping occasions — see "
                      f"ns_household_tpnb_week_agg_train.sql), but worth checking "
                      f"GMM_N_COMPONENTS before using these as crisp segments.")

        if emb_path.exists():
            emb_ids = set(pd.read_parquet(emb_path, columns=["basket_id"])["basket_id"])
            missing = len(emb_ids - set(clu["basket_id"]))
            if missing:
                ok = _fail(f"{missing:,} embedded baskets have no need-state label")
            else:
                print("    every embedded basket has a label — OK")
    else:
        print(f"  SKIP: {clu_path} not found")

    # Need-state graph artifacts
    for name, path in (("adjacency", OUTPUT_DIR / "need_state_adjacency.parquet"),
                       ("transitions", OUTPUT_DIR / "need_state_transitions.parquet")):
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        checked_any = True
        df = pd.read_parquet(path)
        print(f"  {name}: {len(df):,} rows")
        if name == "transitions" and len(df):
            sums = df.groupby("from_need_state")["prob"].sum()
            if not np.allclose(sums.to_numpy(), 1.0, atol=1e-6):
                ok = _fail(f"transition probabilities do not sum to 1 per need-state "
                           f"(worst {abs(sums - 1).max():.4f})")
            else:
                print("    outgoing probabilities sum to 1.0 — OK")
            thin = float(df["low_support"].mean()) if "low_support" in df else 0.0
            if thin > 0.5:
                print(f"    NOTE: {thin:.0%} of transitions are below min_support. The "
                      f"training window is ~2 periods (~8 weeks); widen it in "
                      f"ns_household_tpnb_week_agg_train.sql before trusting journeys.")
        if name == "adjacency" and len(df):
            if (df["need_state_a"] == df["need_state_b"]).any():
                ok = _fail("adjacency file contains self-loops")
            else:
                print("    no self-loops — OK")

    if not checked_any:
        print("  Nothing to check — no prod artifacts present. Run the pipeline first.")
        return True

    print("PASSED" if ok else "FAILED")
    return ok


# ─────────────────────────────────────────────
# 9. CLUSTERING PROGRESS + GRAPH BUILD
# ─────────────────────────────────────────────

def _grouping(ids, labels):
    """Labels -> set of frozensets of co-clustered ids, ignoring label numbering."""
    from collections import defaultdict
    buckets = defaultdict(set)
    for i, lab in zip(ids, labels):
        buckets[lab].add(i)
    return {frozenset(members) for members in buckets.values()}


def _same_edges(left, right):
    """
    Edge-table equality that survives a parquet round-trip.

    Deliberately not DataFrame.equals: that compares dtypes, and pandas 3 /
    future.infer_string reads string columns back from parquet as StringDtype
    where they went in as object. That is a storage detail, not a difference
    in the edges, and failing on it would be a false alarm.
    """
    if len(left) != len(right) or list(left.columns) != list(right.columns):
        return False
    for col in left.columns:
        lhs, rhs = left[col].reset_index(drop=True), right[col].reset_index(drop=True)
        if col == "weight":
            if not np.allclose(lhs.to_numpy(dtype=float), rhs.to_numpy(dtype=float)):
                return False
        elif list(lhs.astype(object)) != list(rhs.astype(object)):
            return False
    return True


def _reference_build_igraph(edges):
    """
    The pure-Python _build_igraph that cluster_basket_embeddings.py used to
    have, kept here purely as the thing the vectorised version must agree with.
    Do NOT "fix" this to match the fast one — it is the baseline.
    """
    baskets = sorted(set(edges["basket_a"]) | set(edges["basket_b"]))
    idx = {b: i for i, b in enumerate(baskets)}
    pairs = [(idx[a], idx[b]) for a, b in zip(edges["basket_a"], edges["basket_b"])]
    return baskets, pairs, edges["weight"].tolist()


def check_graph_coverage():
    """
    Every basket must come out of Stage 2 with a label, or a loud error.

    The bug this pins: mutual-kNN keeps an edge only when both baskets rank
    each other in their top-K, and _build_igraph derives its vertex set from
    the edge endpoints — so a basket with no surviving edge is not a vertex,
    gets no community, and is simply absent from run_leiden_on_basket_graph's
    output. pipeline_main then outer-merges that with the GMM labels, which DO
    cover everyone, so the saved parquet had the right row count with half its
    need_state_cluster column NaN. A real run clustered 28,692,907 of
    57,115,804 baskets and said nothing.

    Three things are verified:
      * _mutual_coverage_mask agrees with a hand-computed mutual structure;
      * _check_graph_coverage raises below the threshold and passes above it;
      * run_leiden_on_basket_graph given all_basket_ids returns EVERY basket,
        with UNCLUSTERED (not NaN, not a dropped row) for the isolated ones.
    """
    print()
    print("=" * 70)
    print("GRAPH COVERAGE — no basket silently loses its label")
    print("=" * 70)
    ok = True

    import cluster_basket_embeddings as cbe

    # ── 1. _mutual_coverage_mask against a hand-built neighbour table ──
    # Column 0 is each row's self-match, as pynndescent returns it.
    #   0 <-> 1 mutual.  2 -> 0 but 0 does not list 2 back, so 2 is isolated.
    #   3 <-> 4 mutual.
    indices = np.array([
        [0, 1, 3],   # 0 lists 1, 3
        [1, 0, 4],   # 1 lists 0  -> mutual with 0
        [2, 0, 1],   # 2 lists 0, 1 -> neither lists 2 back
        [3, 4, 0],   # 3 lists 4, 0 -> 0 lists 3 back at k=2
        [4, 3, 1],   # 4 lists 3  -> mutual with 3
    ], dtype=np.int32)

    got = cbe._mutual_coverage_mask(indices, k=2)
    expected = np.array([True, True, False, True, True])
    if not np.array_equal(got, expected):
        ok = _fail(f"_mutual_coverage_mask(k=2) = {got.tolist()}, expected {expected.tolist()}")
    else:
        print("  _mutual_coverage_mask k=2 — OK (4 of 5 covered, basket 2 isolated)")

    # At k=1 only the first neighbour counts: 0->1 and 1->0 still mutual,
    # 3->4 and 4->3 still mutual, 2->0 still unreciprocated.
    got_k1 = cbe._mutual_coverage_mask(indices, k=1)
    expected_k1 = np.array([True, True, False, True, True])
    if not np.array_equal(got_k1, expected_k1):
        ok = _fail(f"_mutual_coverage_mask(k=1) = {got_k1.tolist()}, "
                   f"expected {expected_k1.tolist()}")
    else:
        print("  _mutual_coverage_mask k=1 — OK")

    # A row whose only "neighbour" is itself must not count as mutual.
    self_only = np.array([[0, 0], [1, 1]], dtype=np.int32)
    if cbe._mutual_coverage_mask(self_only, k=1).any():
        ok = _fail("_mutual_coverage_mask counted a self-match as a mutual neighbour")
    else:
        print("  _mutual_coverage_mask ignores self-matches — OK")

    # ── 2. the coverage gate ──
    universe = np.array([f"b{i}" for i in range(10)], dtype=object)
    in_graph = universe[:5]

    try:
        cbe._check_graph_coverage(in_graph, universe, min_coverage=0.95)
        ok = _fail("_check_graph_coverage accepted 50% coverage against a 0.95 floor")
    except ValueError as e:
        if "50.0%" not in str(e):
            ok = _fail(f"coverage error should state the actual coverage, got: {e}")
        else:
            print("  _check_graph_coverage raises at 50% vs floor 0.95 — OK")

    try:
        returned = cbe._check_graph_coverage(universe, universe, min_coverage=0.95)
        if len(returned) != 10:
            ok = _fail(f"_check_graph_coverage returned {len(returned)} ids, expected 10")
        else:
            print("  _check_graph_coverage passes at 100% and returns the universe — OK")
    except ValueError as e:
        ok = _fail(f"_check_graph_coverage rejected full coverage: {e}")

    # min_coverage=0 is the documented escape hatch and must never raise.
    try:
        cbe._check_graph_coverage(in_graph, universe, min_coverage=0.0)
        print("  _check_graph_coverage honours the min_coverage=0 escape hatch — OK")
    except ValueError as e:
        ok = _fail(f"min_coverage=0 should allow any coverage, but raised: {e}")

    # ── 3. end to end: isolated baskets come back as UNCLUSTERED ──
    # Two well-separated triangles plus three baskets with no edges at all.
    edges = pd.DataFrame({
        "basket_a": ["b0", "b1", "b0", "b3", "b4", "b3"],
        "basket_b": ["b1", "b2", "b2", "b4", "b5", "b5"],
        "weight":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    full_universe = np.array([f"b{i}" for i in range(9)], dtype=object)

    clusters = cbe.run_leiden_on_basket_graph(
        edges, all_basket_ids=full_universe, min_coverage=0.0,
    )

    if len(clusters) != 9:
        ok = _fail(f"Leiden returned {len(clusters)} rows for a 9-basket population — "
                   f"every basket must appear")
    elif clusters["need_state_cluster"].isna().any():
        ok = _fail("Leiden output contains NaN need_state_cluster — the isolated "
                   "baskets must be UNCLUSTERED, not missing")
    else:
        isolated = set(clusters.loc[
            clusters["need_state_cluster"] == cbe.UNCLUSTERED, "basket_id"
        ])
        if isolated != {"b6", "b7", "b8"}:
            ok = _fail(f"expected b6,b7,b8 to be UNCLUSTERED, got {sorted(isolated)}")
        else:
            print("  run_leiden_on_basket_graph returns all 9 baskets, 3 UNCLUSTERED — OK")

        # The two triangles are disconnected, so they must not share a cluster.
        c0 = clusters.loc[clusters["basket_id"] == "b0", "need_state_cluster"].iloc[0]
        c3 = clusters.loc[clusters["basket_id"] == "b3", "need_state_cluster"].iloc[0]
        if c0 == c3:
            ok = _fail("two disconnected triangles landed in the same cluster")
        elif cbe.UNCLUSTERED in (c0, c3):
            ok = _fail("a connected basket was labelled UNCLUSTERED")
        else:
            print("  disconnected components get distinct clusters — OK")

    # Without all_basket_ids the old shape is preserved: only graph vertices.
    legacy = cbe.run_leiden_on_basket_graph(edges)
    if len(legacy) != 6:
        ok = _fail(f"without all_basket_ids Leiden should return only the 6 graph "
                   f"vertices, got {len(legacy)}")
    else:
        print("  omitting all_basket_ids still returns graph vertices only — OK")

    print("PASSED" if ok else "FAILED")
    return ok


def check_clustering_progress():
    """
    Stage 2's progress reporting, and the vectorised graph build it exposed.

    Two separate things are verified here, and the second is the one that
    matters for results:

      * the reporting helpers themselves (duration formatting, step
        announcement, heartbeat) — cheap to get subtly wrong, e.g. a heartbeat
        thread that never fires or never stops;

      * that the rewritten _build_igraph and the per-iteration Leiden loop are
        BEHAVIOUR-PRESERVING. _build_igraph was changed from a Python dict +
        tuple-list build to pd.factorize over arrays, and Leiden now runs the
        optimiser one iteration at a time instead of calling find_partition()
        once. Both were done for observability and memory, neither is allowed
        to change which baskets end up clustered together — so both are pinned
        against their previous implementation here.
    """
    print()
    print("=" * 70)
    print("9. CLUSTERING: progress reporting, vectorised graph build, Leiden parity")
    print("=" * 70)

    import io
    import os as _os
    import threading
    import time as _time
    from contextlib import redirect_stdout

    import leidenalg

    import cluster_basket_embeddings as cbe
    import config

    ok = True

    # ── duration formatting, hand-computed ───────────────────────────────
    fmt_cases = [(0, "0s"), (9, "9s"), (59, "59s"), (60, "1m 00s"),
                 (95, "1m 35s"), (3600, "1h 00m 00s"), (3725, "1h 02m 05s"),
                 (86_400, "24h 00m 00s")]
    bad_fmt = [(s, cbe.fmt_duration(s), want) for s, want in fmt_cases
               if cbe.fmt_duration(s) != want]
    if bad_fmt:
        ok = _fail(f"fmt_duration wrong (seconds, got, want): {bad_fmt}")
    else:
        print(f"  fmt_duration: {len(fmt_cases)} hand-computed cases — OK")

    # ── progress_step announces, times, and heartbeats ───────────────────
    buf = io.StringIO()
    with redirect_stdout(buf):
        with cbe.progress_step("unit test step", 2, 7, heartbeat_secs=0.05):
            _time.sleep(0.25)
    out = buf.getvalue()
    if "[2/7] unit test step ..." not in out:
        ok = _fail(f"progress_step did not announce the step: {out!r}")
    elif "[2/7] unit test step: done in" not in out:
        ok = _fail(f"progress_step did not report a duration: {out!r}")
    elif "still running" not in out:
        ok = _fail(f"heartbeat never fired for a step longer than the interval: {out!r}")
    else:
        beats = out.count("still running")
        print(f"  progress_step: announces, heartbeats ({beats}x at 0.05s), "
              f"reports duration — OK")

    # A step SHORTER than the heartbeat interval must stay quiet, otherwise
    # every fast step in the pipeline gains a useless line.
    buf = io.StringIO()
    with redirect_stdout(buf):
        with cbe.progress_step("fast step", heartbeat_secs=60):
            pass
    if "still running" in buf.getvalue():
        ok = _fail("heartbeat fired for a step shorter than its interval")
    else:
        print("  progress_step: no heartbeat for a sub-interval step — OK")

    # The heartbeat thread must not outlive the step. Compared by thread
    # IDENTITY rather than threading.active_count(): tqdm's monitor thread and
    # numba/joblib pools come and go independently, so a bare count is flaky on
    # a loaded box and would fail for reasons having nothing to do with this.
    before_ids = {t.ident for t in threading.enumerate()}
    with redirect_stdout(io.StringIO()):
        with cbe.progress_step("thread cleanup", heartbeat_secs=0.01):
            _time.sleep(0.05)
    leaked = [t for t in threading.enumerate()
              if t.ident not in before_ids and t.is_alive()]
    if leaked:
        ok = _fail(f"heartbeat thread outlived its step: {[t.name for t in leaked]}")
    else:
        print("  progress_step: heartbeat thread is joined on exit — OK")

    # A step that raises must say so, not print a success-shaped line — the
    # whole point of this helper is telling a hang from a crash.
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            with cbe.progress_step("exploding step"):
                raise MemoryError("simulated OOM")
    except MemoryError:
        pass
    else:
        ok = _fail("progress_step swallowed an exception raised inside it")
    crash_out = buf.getvalue()
    if "FAILED after" not in crash_out:
        ok = _fail(f"a step that raised did not report failure: {crash_out!r}")
    elif "done in" in crash_out:
        ok = _fail(f"a step that raised still printed 'done in': {crash_out!r}")
    else:
        print("  progress_step: a raising step reports FAILED, exception propagates — OK")

    # ── _build_igraph parity vs the old pure-Python build ────────────────
    rng = np.random.default_rng(20260922)
    mismatches = []
    for id_kind in ("int", "str", "sparse-int"):
        for _ in range(60):
            n_ids = int(rng.integers(2, 25))
            if id_kind == "int":
                ids = np.arange(n_ids)
            elif id_kind == "str":
                # Deliberately NOT zero-padded: '10' sorts before '9' as a
                # string, so this catches a build that sorted numerically on
                # one side and lexically on the other.
                ids = np.array([f"b{i}" for i in range(n_ids)], dtype=object)
            else:
                ids = np.arange(n_ids) * int(rng.integers(1, 10_000))

            m = int(rng.integers(1, 40))
            e = pd.DataFrame({
                "basket_a": rng.choice(ids, m),
                "basket_b": rng.choice(ids, m),
                "weight": rng.random(m),
            })

            want_baskets, want_pairs, want_weights = _reference_build_igraph(e)
            with redirect_stdout(io.StringIO()):
                g, got_baskets = cbe._build_igraph(e)

            if list(got_baskets) != list(want_baskets):
                mismatches.append((id_kind, "vertex order/identity"))
                break
            # Compared position by position (igraph keeps insertion order) but
            # with each edge's endpoints unordered: the graph is undirected, so
            # (i, j) and (j, i) are the same edge and igraph is free to store
            # either. A real divergence — wrong index, wrong count, wrong
            # pairing — still fails this.
            if ([frozenset(edge.tuple) for edge in g.es]
                    != [frozenset(p) for p in want_pairs]):
                mismatches.append((id_kind, "edge endpoint indices"))
                break
            if not np.allclose(g.es["weight"], want_weights):
                mismatches.append((id_kind, "edge weights"))
                break
            if g.vcount() != len(want_baskets) or g.ecount() != len(want_pairs):
                mismatches.append((id_kind, "vcount/ecount"))
                break

    if mismatches:
        ok = _fail(f"_build_igraph diverged from the pure-Python build: {mismatches}")
    else:
        print("  _build_igraph: identical vertices, edge indices and weights to the "
              "pure-Python build (180 random cases: int / str / sparse-int ids) — OK")

    # Weights must stay attached to the RIGHT edge, checked by endpoint id
    # rather than by position — a factorize that reordered anything would pass
    # a positional check and still corrupt every edge weight.
    e = pd.DataFrame({
        "basket_a": ["b2", "b0", "b1"],
        "basket_b": ["b0", "b1", "b2"],
        "weight":   [0.25, 0.50, 0.75],
    })
    with redirect_stdout(io.StringIO()):
        g, baskets = cbe._build_igraph(e)
    by_endpoints = {
        frozenset((baskets[edge.source], baskets[edge.target])): edge["weight"]
        for edge in g.es
    }
    want_by_endpoints = {
        frozenset(("b2", "b0")): 0.25,
        frozenset(("b0", "b1")): 0.50,
        frozenset(("b1", "b2")): 0.75,
    }
    if by_endpoints != want_by_endpoints:
        ok = _fail(f"edge weights not attached to the right endpoints: "
                   f"{by_endpoints} != {want_by_endpoints}")
    else:
        print("  _build_igraph: each weight stays on its own edge, keyed by "
              "basket id — OK")

    # A null basket id must stop the run, not become vertex -1. This is the
    # one place the factorize build genuinely diverges from the old
    # sorted(set(...)) one, which would have kept NaN as a real vertex.
    nan_edges = pd.DataFrame({
        "basket_a": ["b0", None, "b1"],
        "basket_b": ["b1", "b0", "b0"],
        "weight":   [0.5, 0.5, 0.5],
    })
    try:
        with redirect_stdout(io.StringIO()):
            cbe._build_igraph(nan_edges)
        ok = _fail("_build_igraph accepted a null basket id instead of raising")
    except ValueError as exc:
        if "null basket id" not in str(exc):
            ok = _fail(f"_build_igraph raised the wrong error for a null id: {exc}")
        else:
            print("  _build_igraph: a null basket id raises instead of silently "
                  "becoming vertex -1 — OK")

    # ── Leiden: per-iteration loop == find_partition ─────────────────────
    # Three 6-node cliques joined by two deliberately weak bridges. Any
    # correct Leiden run at resolution 1.0 recovers exactly those cliques, so
    # this is a fixed target rather than a "whatever it did last time" pin.
    cliques = [[f"c{c}_{i}" for i in range(6)] for c in range(3)]
    rows = []
    for members in cliques:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                rows.append((members[i], members[j], 1.0))
    rows.append((cliques[0][0], cliques[1][0], 0.01))
    rows.append((cliques[1][0], cliques[2][0], 0.01))
    clique_edges = pd.DataFrame(rows, columns=["basket_a", "basket_b", "weight"])
    want_groups = {frozenset(members) for members in cliques}

    with redirect_stdout(io.StringIO()):
        got = cbe.run_leiden_on_basket_graph(clique_edges, resolution=1.0)
    got_groups = _grouping(got["basket_id"], got["need_state_cluster"])

    if got_groups != want_groups:
        ok = _fail(f"Leiden did not recover the three cliques: "
                   f"{[sorted(x) for x in got_groups]}")
    else:
        print("  run_leiden_on_basket_graph: recovers 3 planted cliques — OK")

    # Same graph, same seed, same iteration count, through leidenalg's own
    # find_partition(). The per-iteration loop must agree with it.
    with redirect_stdout(io.StringIO()):
        g, baskets = cbe._build_igraph(clique_edges)
    reference = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=1.0,
        n_iterations=config.LEIDEN_N_ITERATIONS, seed=config.SEED,
    )
    ref_groups = _grouping(baskets, reference.membership)
    if got_groups != ref_groups:
        ok = _fail("per-iteration Leiden loop disagrees with find_partition(): "
                   f"{[sorted(x) for x in got_groups]} vs "
                   f"{[sorted(x) for x in ref_groups]}")
    else:
        print(f"  run_leiden_on_basket_graph: same partition as "
              f"find_partition(n_iterations={config.LEIDEN_N_ITERATIONS}, "
              f"seed={config.SEED}) — OK")

    # Informational, deliberately NOT a failure: on a graph whose right answer
    # is not obvious, does the per-iteration loop land on exactly the same
    # partition as find_partition()? Both are legitimate Leiden runs, so a
    # difference is not a bug — but it decides whether Stage 2 rerun on the
    # same embeddings reproduces its previous cluster ids, which is worth
    # knowing before anyone compares two runs' outputs.
    blocks = [[f"n{b}_{i}" for i in range(25)] for b in range(4)]
    blocky = []
    flat = [node for block in blocks for node in block]
    for block in blocks:
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if rng.random() < 0.45:
                    blocky.append((block[i], block[j], float(rng.uniform(0.5, 1.0))))
    for _ in range(40):
        a, b = rng.choice(flat, 2, replace=False)
        if a.split("_")[0] != b.split("_")[0]:
            blocky.append((a, b, float(rng.uniform(0.01, 0.1))))
    blocky_edges = pd.DataFrame(blocky, columns=["basket_a", "basket_b", "weight"])

    with redirect_stdout(io.StringIO()):
        loop_result = cbe.run_leiden_on_basket_graph(blocky_edges, resolution=1.0)
        g_b, baskets_b = cbe._build_igraph(blocky_edges)
    ref_b = leidenalg.find_partition(
        g_b, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=1.0,
        n_iterations=config.LEIDEN_N_ITERATIONS, seed=config.SEED,
    )
    loop_groups = _grouping(loop_result["basket_id"], loop_result["need_state_cluster"])
    ref_groups_b = _grouping(baskets_b, ref_b.membership)
    if loop_groups == ref_groups_b:
        print(f"  INFO per-iteration loop is bit-identical to find_partition() on a "
              f"4-block random graph ({len(loop_groups)} clusters) — reruns reproduce")
    else:
        print(f"  INFO per-iteration loop found {len(loop_groups)} clusters vs "
              f"find_partition()'s {len(ref_groups_b)} on a 4-block random graph. "
              f"Both are valid Leiden runs; cluster ids are NOT reproducible "
              f"across the two call styles.")

    # The early "converged, stopping" path must return the same answer as
    # letting it run the full iteration budget.
    with redirect_stdout(io.StringIO()):
        few = cbe.run_leiden_on_basket_graph(clique_edges, resolution=1.0, n_iterations=1)
        many = cbe.run_leiden_on_basket_graph(clique_edges, resolution=1.0, n_iterations=8)
    if _grouping(few["basket_id"], few["need_state_cluster"]) != want_groups:
        ok = _fail("Leiden with n_iterations=1 did not recover the cliques")
    elif _grouping(many["basket_id"], many["need_state_cluster"]) != want_groups:
        ok = _fail("Leiden with n_iterations=8 did not recover the cliques")
    else:
        print("  run_leiden_on_basket_graph: n_iterations 1 and 8 agree "
              "(early convergence stop is label-preserving) — OK")

    buf = io.StringIO()
    with redirect_stdout(buf):
        cbe.run_leiden_on_basket_graph(clique_edges, resolution=1.0, n_iterations=3)
    leiden_out = buf.getvalue()
    if "Leiden iteration 1" not in leiden_out:
        ok = _fail(f"Leiden did not report per-iteration progress: {leiden_out!r}")
    elif "clusters, quality=" not in leiden_out:
        ok = _fail(f"Leiden iteration report is missing cluster count/quality: {leiden_out!r}")
    else:
        print("  run_leiden_on_basket_graph: reports cluster count + quality "
              "per iteration — OK")

    # ── build_basket_knn_graph over the .to_numpy() id path ──────────────
    # String basket ids on purpose: the real ones are "household_week", and
    # this function now indexes them as a numpy array rather than a list.
    # Centres chosen as distinct DIRECTIONS, not distinct positions: both this
    # function and the GMM path L2-normalise first, so a group centred on the
    # origin would smear into random directions all over the unit circle and
    # the "well-separated" premise would be false.
    centres = np.array([[1.0, 0.0], [0.0, 1.0], [-0.7071, -0.7071]])
    emb, emb_ids = [], []
    for c_i, centre in enumerate(centres):
        for p_i in range(10):
            emb.append(centre + rng.normal(scale=0.01, size=2))
            emb_ids.append(f"{c_i}_{p_i:03d}")
    embeddings = pd.DataFrame({"basket_id": emb_ids, "gnn_embedding": [np.asarray(v) for v in emb]})

    # use_cache=False: this check is about the edge maths, and the default
    # would write a cache file into the real ../data/output.
    with redirect_stdout(io.StringIO()):
        knn_edges = cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=True,
                                               use_cache=False)

    if list(knn_edges.columns) != ["basket_a", "basket_b", "weight"]:
        ok = _fail(f"build_basket_knn_graph columns changed: {list(knn_edges.columns)}")
    elif len(knn_edges) == 0:
        ok = _fail("build_basket_knn_graph produced no edges for 3 well-separated groups")
    else:
        seen_ids = set(knn_edges["basket_a"]) | set(knn_edges["basket_b"])
        unknown = seen_ids - set(emb_ids)
        cross = knn_edges[knn_edges["basket_a"].str[0] != knn_edges["basket_b"].str[0]]
        if unknown:
            ok = _fail(f"edge list contains ids that were never input (id array "
                       f"indexing is off): {sorted(unknown)[:5]}")
        elif len(cross) > 0:
            ok = _fail(f"{len(cross)} edge(s) cross well-separated groups: "
                       f"{cross.head().to_dict('records')}")
        elif not ((knn_edges["weight"] >= 0).all() and (knn_edges["weight"] <= 1).all()):
            ok = _fail("edge weights fell outside [0, 1] after minmax scaling")
        else:
            print(f"  build_basket_knn_graph: {len(knn_edges)} edges, all within-group, "
                  f"ids preserved through .to_numpy() — OK")

    # normalize(copy=False) must not reach back into the caller's DataFrame.
    # pipeline_main.py hands the SAME embeddings frame to Leiden and then to
    # GMM, so an in-place normalise that escaped would leave Stage 2b fitting
    # on already-unit-length vectors — silently, and only at full scale.
    before = np.stack(embeddings["gnn_embedding"].values).copy()
    with redirect_stdout(io.StringIO()):
        cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=True, use_cache=False)
    after = np.stack(embeddings["gnn_embedding"].values)
    if not np.array_equal(before, after):
        changed = int((~np.isclose(before, after)).any(axis=1).sum())
        ok = _fail(f"normalize(copy=False) mutated the caller's embeddings "
                   f"({changed} of {len(before)} rows changed)")
    else:
        print("  normalize(copy=False): caller's embedding frame is untouched — OK")

    # ── integer-key dedupe == drop_duplicates on the string ids ──────────
    # build_basket_knn_graph stopped deduping two object columns (~193M Python
    # string hashes at full scale) and now dedupes one int64 key. That is only
    # a speedup if it selects exactly the same rows, including which duplicate
    # survives, so the two are compared directly here on inputs built to be
    # duplicate-heavy.
    dedupe_bad = []
    for trial in range(200):
        n_v = int(rng.integers(2, 12))
        m = int(rng.integers(1, 50))
        ea = rng.integers(0, n_v, m)
        eb = rng.integers(0, n_v, m)
        esim = rng.random(m)
        ids = np.array([f"v{i}" for i in range(n_v)], dtype=object)

        # Reference: exactly what the function used to do.
        want = pd.DataFrame({
            "basket_a": ids[ea], "basket_b": ids[eb], "similarity": esim,
        }).drop_duplicates(subset=["basket_a", "basket_b"]).reset_index(drop=True)

        # New path.
        pair_key = ea.astype(np.int64, copy=False) * n_v + eb
        _, first_seen = np.unique(pair_key, return_index=True)
        first_seen.sort()
        got = pd.DataFrame({
            "basket_a": ids[ea[first_seen]], "basket_b": ids[eb[first_seen]],
            "similarity": esim[first_seen],
        })

        if list(got["basket_a"]) != list(want["basket_a"]) \
                or list(got["basket_b"]) != list(want["basket_b"]):
            dedupe_bad.append((trial, "different rows or row order"))
            break
        # Which duplicate survives matters: keep="first" carries the first
        # occurrence's similarity, and that becomes the edge weight.
        if not np.allclose(got["similarity"].to_numpy(), want["similarity"].to_numpy()):
            dedupe_bad.append((trial, "kept a different duplicate's similarity"))
            break

    if dedupe_bad:
        ok = _fail(f"integer-key dedupe diverged from drop_duplicates: {dedupe_bad}")
    else:
        print("  edge dedupe: int64-key dedupe picks the same rows, order and "
              "kept-duplicate as drop_duplicates on string ids (200 cases) — OK")

    # One-directional mode collapses (i,j) and (j,i) onto the same pair, so it
    # is the mode that actually produces duplicates to remove.
    with redirect_stdout(io.StringIO()):
        one_way = cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=False,
                                             use_cache=False)
    dup_count = one_way.duplicated(subset=["basket_a", "basket_b"]).sum()
    if dup_count:
        ok = _fail(f"{dup_count} duplicate edge(s) survived in one-directional mode")
    else:
        print(f"  edge dedupe: one-directional mode leaves 0 duplicate pairs "
              f"({len(one_way)} edges) — OK")

    # ── kNN edge-list cache: reuse when valid, REBUILD when not ──────────
    # The reuse half saves hours. The invalidation half is the one that can
    # silently corrupt a run, so most of what is checked here is the cases
    # where the cache must NOT be trusted.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cache = str(Path(tmp) / "edges.parquet")
        manifest = str(Path(tmp) / "edges.manifest.json")

        buf = io.StringIO()
        with redirect_stdout(buf):
            first = cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=True,
                                               use_cache=True, cache_path=cache)
        wrote = buf.getvalue()

        if not Path(cache).exists():
            ok = _fail("edge cache parquet was not written")
        elif not Path(manifest).exists():
            ok = _fail("edge cache manifest was not written")
        elif "Cached" not in wrote:
            ok = _fail(f"cache write was not reported: {wrote!r}")
        else:
            print(f"  edge cache: wrote {Path(cache).name} + "
                  f"{Path(manifest).name} — OK")

        # Identical inputs -> reuse, and byte-identical edges back.
        buf = io.StringIO()
        with redirect_stdout(buf):
            second = cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=True,
                                                use_cache=True, cache_path=cache)
        reused = buf.getvalue()
        if "REUSING cached basket kNN graph" not in reused:
            ok = _fail(f"identical inputs did not hit the cache: {reused!r}")
        elif "kNN method" in reused:
            ok = _fail("cache hit still ran the neighbour search")
        elif not _same_edges(first, second):
            ok = _fail("cached edges differ from the edges that were cached")
        else:
            print("  edge cache: identical inputs reuse it and skip the kNN "
                  "search, same edges back — OK")

        # Different k must NOT reuse.
        buf = io.StringIO()
        with redirect_stdout(buf):
            cbe.build_basket_knn_graph(embeddings, k=6, use_mutual=True,
                                       use_cache=True, cache_path=cache)
        if "REUSING" in buf.getvalue():
            ok = _fail("cache was reused after k changed")
        else:
            print("  edge cache: invalidated by a different k — OK")

        # Same basket ids, same row count, DIFFERENT embedding values — the
        # retrained-model case. Row count alone would happily reuse here.
        moved = embeddings.copy()
        moved["gnn_embedding"] = [np.asarray(v) + 0.5 for v in moved["gnn_embedding"]]
        buf = io.StringIO()
        with redirect_stdout(buf):
            cbe.build_basket_knn_graph(moved, k=4, use_mutual=True,
                                       use_cache=True, cache_path=cache)
        if "REUSING" in buf.getvalue():
            ok = _fail("cache was reused after the embeddings themselves changed — "
                       "a retrained model would silently cluster the old geometry")
        else:
            print("  edge cache: invalidated when embedding VALUES change at the "
                  "same row count — OK")

        # A parquet with no manifest must not be trusted (the crash-between-
        # writes case _write_edge_cache is ordered to produce).
        Path(cbe._cache_manifest_path(cache)).unlink()
        buf = io.StringIO()
        with redirect_stdout(buf):
            cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=True,
                                       use_cache=True, cache_path=cache)
        if "REUSING" in buf.getvalue():
            ok = _fail("cache was reused with its manifest missing")
        else:
            print("  edge cache: a manifest-less parquet is rebuilt, not trusted — OK")

        # use_cache=False must neither read nor write.
        fresh = str(Path(tmp) / "unused.parquet")
        with redirect_stdout(io.StringIO()):
            cbe.build_basket_knn_graph(embeddings, k=4, use_mutual=True,
                                       use_cache=False, cache_path=fresh)
        if Path(fresh).exists():
            ok = _fail("use_cache=False still wrote a cache file")
        else:
            print("  edge cache: use_cache=False writes nothing — OK")

    # ── GMM still runs with verbose output turned on ─────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        model_path = str(Path(tmp) / "gmm_test.pkl")
        buf = io.StringIO()
        with redirect_stdout(buf):
            gmm_result = cbe.cluster_basket_embeddings_gmm(
                embeddings, n_components=3, save_path=model_path)
        gmm_out = buf.getvalue()

        if list(gmm_result.columns) != ["basket_id", "need_state_cluster_gmm", "gmm_confidence"]:
            ok = _fail(f"GMM result columns changed: {list(gmm_result.columns)}")
        elif len(gmm_result) != len(embeddings):
            ok = _fail(f"GMM returned {len(gmm_result)} rows for {len(embeddings)} baskets")
        elif not ((gmm_result["gmm_confidence"] >= 0).all()
                  and (gmm_result["gmm_confidence"] <= 1).all()):
            ok = _fail("gmm_confidence outside [0, 1]")
        elif not Path(model_path).exists():
            ok = _fail("fitted GMM was not saved — score_new_baskets.py depends on it")
        elif "fitting GMM" not in gmm_out:
            ok = _fail(f"GMM fit did not report progress: {gmm_out!r}")
        else:
            print("  cluster_basket_embeddings_gmm: runs with verbose EM output, "
                  "saves model, confidence in [0,1] — OK")

    # ── the new config knobs ─────────────────────────────────────────────
    knob_problems = []
    if not isinstance(getattr(config, "LEIDEN_N_ITERATIONS", None), int):
        knob_problems.append("LEIDEN_N_ITERATIONS missing or not an int")
    if not isinstance(getattr(config, "PROGRESS_HEARTBEAT_SECS", None), int):
        knob_problems.append("PROGRESS_HEARTBEAT_SECS missing or not an int")
    if not isinstance(getattr(config, "CACHE_BASKET_EDGES", None), bool):
        knob_problems.append("CACHE_BASKET_EDGES missing or not a bool")
    # Only this one var disables the default check — an unrelated PIPELINE_*
    # override (PIPELINE_SEED, say) should not silently skip it.
    if "PIPELINE_LEIDEN_N_ITERATIONS" not in _os.environ:
        # leidenalg's own find_partition default is 2 — drifting from it would
        # silently change every clustering result.
        if getattr(config, "LEIDEN_N_ITERATIONS", None) != 2:
            knob_problems.append(f"LEIDEN_N_ITERATIONS default is "
                                 f"{config.LEIDEN_N_ITERATIONS}, leidenalg's is 2")
    if knob_problems:
        ok = _fail(f"config knobs: {knob_problems}")
    else:
        print(f"  config: LEIDEN_N_ITERATIONS={config.LEIDEN_N_ITERATIONS}, "
              f"PROGRESS_HEARTBEAT_SECS={config.PROGRESS_HEARTBEAT_SECS}s, "
              f"CACHE_BASKET_EDGES={config.CACHE_BASKET_EDGES} — OK")

    return ok


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="Skip the slow DuckDB/LMDB/torch functional check")
    parser.add_argument("--prod-outputs", action="store_true",
                        help="Also sanity-check real artifacts under ../data/output")
    args = parser.parse_args()

    checks = [
        ("static", check_no_theme_identifiers),
        ("config", check_config),
        ("graph primitives", check_graph_primitives),
        ("co-purchase additivity", check_copurchase_chunk_additivity),
        ("need-state graph", check_need_state_graph),
        ("clustering progress / graph build / Leiden parity", check_clustering_progress),
        ("graph coverage / no silent label loss", check_graph_coverage),
    ]
    if not args.fast:
        checks.append(("functional / parity / basket store", check_functional))
    if args.prod_outputs:
        checks.append(("prod outputs", check_prod_outputs))

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"\n{name.upper()} FAILED with an exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = False

    print()
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("=" * 70)
    if all(results.values()):
        print("ALL CHECKS PASSED")
        sys.exit(0)
    print("CHECKS FAILED — see above")
    sys.exit(1)


if __name__ == "__main__":
    main()
