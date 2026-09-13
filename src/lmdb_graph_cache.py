"""
lmdb_graph_cache.py

Build-once, random-access-during-training graph cache for the ~300k-basket
training sample.

Why this exists: GraphBuilder.build_training_graphs() used to build all
sampled graphs into one Python list, because torch_geometric's DataLoader
needs random-access indexing across the whole sample to shuffle it every
epoch. That list is ~20-25GB at real basket sizes — the second-largest
memory risk in this pipeline after `baskets` itself. LMDB (a memory-mapped
key-value store) removes the need to hold that list in RAM: build each
graph once here, and read it back by index during training — the OS page
cache handles what stays "hot", not a pinned Python list.

Reuses GraphBuilder._basket_dense_cp_submatrix() and build_one_graph() —
the SAME per-basket functions embed_all_baskets_fast() uses at inference —
so nothing about feature/graph construction changes, only where the
results live.
"""

import json
import os
import pickle

import lmdb
from torch_geometric.data import Dataset as PyGDataset

from GraphBuilder import _basket_dense_cp_submatrix, build_one_graph, GRAPH_BUILDER_VERSION

_COMMIT_EVERY = 2000  # periodic commits during the build pass, bounding write-txn memory


def build_lmdb_cache(sampled_baskets_df, G, lmdb_path: str,
                      map_size: int = 200 * 1024 ** 3):
    """
    One-time build pass: for each row in sampled_baskets_df, builds its graph
    via the exact same functions build_training_graphs used to, serializes
    it, and writes it to LMDB keyed by its row index. Each graph is
    discarded immediately after being written — nothing accumulates in
    Python memory across the pass except the small per-basket ingredients
    (products/units lists), which is what sampled_baskets_df already is.

    `map_size` is a virtual address-space bound, not pre-allocated disk
    space — safe to size generously.
    """
    os.makedirs(lmdb_path, exist_ok=True)
    csr = G["csr"]
    pid2idx = G["product_id_to_index"]

    products_col = sampled_baskets_df["products"].tolist()
    units_col = sampled_baskets_df["units"].tolist()
    basket_ids = (
        sampled_baskets_df["basket_id"].tolist()
        if "basket_id" in sampled_baskets_df.columns
        else list(range(len(sampled_baskets_df)))
    )
    n = len(products_col)

    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True)
    try:
        txn = env.begin(write=True)
        try:
            for i in range(n):
                products = products_col[i]
                units = units_col[i]
                local_idx_map, basket_dense_cp = _basket_dense_cp_submatrix(products, csr, pid2idx)
                g = build_one_graph(
                    products=products, units=units, basket_id=basket_ids[i],
                    emb_matrix=G["emb_matrix"], emb_dim=G["emb_dim"],
                    subcluster_arr=G["subcluster_arr"],
                    distinctiveness_arr=G["distinctiveness_arr"],
                    dense_cp=basket_dense_cp, local_idx=local_idx_map,
                    product_id_to_index=pid2idx,
                )
                txn.put(str(i).encode(), pickle.dumps(g))
                del g

                if (i + 1) % _COMMIT_EVERY == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    print(f"  LMDB build: {i + 1:,}/{n:,} training graphs written")

            txn.put(b"__len__", str(n).encode())
            txn.commit()
        except Exception:
            txn.abort()
            raise
    finally:
        env.close()

    print(f"LMDB training-graph cache built: {n:,} graphs at {lmdb_path}")


def _manifest_matches(manifest_path: str, expected: dict) -> bool:
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return saved == expected


def _write_manifest(manifest_path: str, manifest: dict):
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp_path, manifest_path)   # atomic — never leaves a truncated manifest


def load_or_build_lmdb_cache(sampled_baskets_df, G, lmdb_path: str, manifest_path: str,
                               seed: int, n_train_samples_requested: int,
                               map_size: int = 200 * 1024 ** 3):
    """
    Staleness-guarded wrapper around build_lmdb_cache() — same pattern
    already used elsewhere in this codebase for product_subclusters.pkl /
    training_graphs.pkl (a cache that's silently wrong is worse than no
    cache). Compares GRAPH_BUILDER_VERSION, seed, sample count, and in_dim
    against a saved manifest before reusing an existing LMDB directory;
    any mismatch triggers a full rebuild with a loud printed reason.
    """
    expected_manifest = {
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "seed": seed,
        "n_graphs": len(sampled_baskets_df),
        "in_dim": G["in_dim"],
        "n_train_samples_requested": n_train_samples_requested,
    }

    if os.path.exists(lmdb_path) and _manifest_matches(manifest_path, expected_manifest):
        print(f"  Reusing cached LMDB training graphs at {lmdb_path} "
              f"(manifest matches: version={GRAPH_BUILDER_VERSION}, seed={seed}, "
              f"n_graphs={expected_manifest['n_graphs']:,})")
        return

    if os.path.exists(lmdb_path):
        print(f"  IGNORING stale LMDB cache at {lmdb_path}: manifest doesn't match this run "
              f"(graph builder version, seed, sample size, or in_dim changed) — rebuilding "
              f"from scratch rather than risking silently training on graphs built under a "
              f"different feature/edge layout.")
        import shutil
        shutil.rmtree(lmdb_path)

    build_lmdb_cache(sampled_baskets_df, G, lmdb_path, map_size=map_size)
    _write_manifest(manifest_path, expected_manifest)


class LMDBGraphDataset(PyGDataset):
    """
    Lazy, per-worker-safe random-access dataset over an LMDB-cached set of
    training graphs.

    Worker safety: the LMDB environment handle is NEVER opened in
    __init__ — only lazily, on first access, inside _ensure_env(). Windows
    DataLoader workers use the `spawn` start method, which pickles this
    Dataset object to send to each new worker process BEFORE that worker's
    first __getitem__ call; if a live env handle were already open at that
    moment, pickling would break or silently corrupt it. __getstate__
    strips `_env` back to None before pickling, so every worker's first
    `get()` call opens its own fresh, independent read-only handle.
    """

    def __init__(self, lmdb_path: str):
        super().__init__()
        self.lmdb_path = lmdb_path
        self._env = None
        # Short-lived handle just to read the length, closed immediately —
        # no environment handle survives past __init__ returning.
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=True)
        try:
            with env.begin(write=False) as txn:
                self._length = int(txn.get(b"__len__").decode())
        finally:
            env.close()

    def _ensure_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=True)
        return self._env

    def len(self):
        return self._length

    def get(self, idx):
        env = self._ensure_env()
        with env.begin(write=False) as txn:
            data_bytes = txn.get(str(idx).encode())
        return pickle.loads(data_bytes)

    def close(self):
        """
        Closes this instance's read handle, if one was ever opened. lmdb
        refuses to open the same environment path twice concurrently within
        one process — call this before constructing a second
        LMDBGraphDataset over the same lmdb_path in the same process (e.g.
        in a test, or a notebook/REPL session; a normal `python
        pipeline_main.py` run never needs this, since each run is its own
        process).
        """
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None   # never send a live LMDB handle across a process boundary
        return state
