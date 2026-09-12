"""
pipeline_main.py

End-to-end pipeline:  your warehouse (manual export)  ->  GNN basket embeddings  ->  need-state clustering

    Stage 0   Read the two parquet files exported from your workspace  (parquet_loader.py)
    Stage 1   Build baskets (whole basket, no category split), train the GNN
    Stage 2   Cluster the resulting basket embeddings — BOTH Leiden and GMM    (cluster_basket_embeddings.py)
    Stage 3   Save need-state output as parquet for manual reload into your warehouse

THEME-FREE: no theme, category, or product_theme concept feeds this
pipeline anywhere, upstream or downstream. This supersedes an earlier
version of this file that worked around GraphBuilder.py's old
product_theme requirement by grouping products into arbitrary fixed-size
chunks labeled like themes — that requirement doesn't exist anymore.
GraphBuilder.py's prepare_globals() now clusters products GLOBALLY, across
the whole catalog, with no pre-grouping of any kind, and takes no
product_theme argument at all. See REFACTOR_NOTES.md for the full account
of what changed in GraphBuilder.py and why.

Need-states are found across the WHOLE basket — split_basket_by_theme.py
is not used anywhere in this pipeline (it's fully decommissioned, not just
unused — see REFACTOR_NOTES.md).

BREAKING CHANGE — cached artifact schema: GraphBuilder.py's node feature
layout changed (in_dim = emb_dim + 4, was + 5) and its sub-clustering
algorithm changed (global MiniBatchKMeans, was per-theme KMeans). Delete
these before running this version, if they exist from an earlier run:
    product_subclusters.pkl, training_graphs.pkl, basket_gnn_model.pt,
    copurchase_sparse.npz, product_id_to_index.pkl, product_units_avg.pkl
There is no product_theme.pkl to delete — that file is never written by
this version, and never read by score_new_baskets.py either.

There's no live database connection anywhere in this file. Before running
this, you need to have already run build_product_embeddings.py (product
embeddings don't exist yet, so that script builds them from scratch — see
its own docstring).

Run this from inside the src/ folder, with a sibling data/ folder holding
your downloaded tables (see the folder layout each script's path constants
assume — PRODUCT_EMBEDDINGS_PARQUET / HOUSEHOLD_TPNB_PERIOD_PARQUET below
use "../data/..." on that assumption):

    project/
    ├── data/
    │   ├── ns_item_lookup_tpna/                    (folder — Spark export)
    │   ├── ns_tpnb_to_tpna_mapping/                (folder)
    │   ├── ns_household_tpnb_period_agg_train/     (folder)
    │   ├── ns_household_tpnb_period_agg_score/     (folder — used later by score_new_baskets.py)
    │   └── product_embeddings.parquet              (single file — written by build_product_embeddings.py)
    └── src/
        ├── build_product_embeddings.py
        ├── pipeline_main.py
        └── ... (the rest)

    cd src
    python build_product_embeddings.py
    python pipeline_main.py

Stage 2 runs both Leiden and GMM clustering and reports how much they agree,
since need-states here are decided by comparing/combining both methods
rather than picking one. GMM_N_COMPONENTS below is a placeholder — swap it
for your team's real best-K selection once that logic is available; in the
meantime cluster_basket_embeddings.select_k_via_bic() gives a reasonable BIC
sweep to eyeball.

Run:  python pipeline_main.py
"""

import os
import pickle
import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import parquet_loader
from GNN_Train import train_and_embed
from cluster_basket_embeddings import (
    cluster_basket_embeddings,
    cluster_basket_embeddings_gmm,
    compare_leiden_gmm,
)

# Drop degenerate 1-item baskets.
MIN_BASKET_PRODUCTS = 2

# Placeholder K for GMM — replace with your real best_k_value.py logic once
# shared. cluster_basket_embeddings.select_k_via_bic() sweeps a range and
# reports BIC/AIC per K if you want to eyeball it first.
GMM_N_COMPONENTS = 30

# ─────────────────────────────────────────────
# Cache-staleness guard
# ─────────────────────────────────────────────
# GraphBuilder.py caches two intermediates to FIXED filenames, reused as-is
# on any future run regardless of whether the underlying data OR CODE
# changed. If you're running this for the first time after the theme-free
# refactor and either of these files exists from before, IT WILL BE LOADED
# AS-IS — and being from before means it was built with a different node
# feature schema (in_dim = emb_dim + 5, per-theme sub-clustering), which
# will silently corrupt this run rather than error loudly. Delete both
# before your first run against this version.
for _stale_cache in ("product_subclusters.pkl", "training_graphs.pkl"):
    if os.path.exists(_stale_cache):
        print(f"WARNING: {_stale_cache} exists and will be REUSED AS-IS by GraphBuilder "
              f"(cached by fixed filename, not by data or code fingerprint). If this is "
              f"left over from before the theme-free refactor, or from a different "
              f"warehouse export, DELETE IT before continuing — the node feature schema "
              f"changed (in_dim = emb_dim + 4, was + 5) and a stale cache here will not "
              f"error, it will just silently produce embeddings that don't mean what you "
              f"think they mean.")

# ─────────────────────────────────────────────
# Stage 0: read the manually-downloaded warehouse exports
# ─────────────────────────────────────────────
# Point these at wherever you saved the two downloads, or edit the defaults
# in parquet_loader.py directly.

# Databricks/Spark exports a table as a FOLDER of part-files, not one single
# .parquet file — pandas reads that folder directly, so these point straight
# at the folder names as downloaded. Paths are relative to running this
# script from inside src/ (see the note at the top of this file).
PRODUCT_EMBEDDINGS_PARQUET    = "../data/output/product_embeddings.parquet"
HOUSEHOLD_TPNB_PERIOD_PARQUET = "../data/ns_household_tpnb_period_agg_train"

print("[Stage 0] Reading warehouse exports...")
product_df_2 = parquet_loader.load_product_embeddings(PRODUCT_EMBEDDINGS_PARQUET)
tpnb_x_hh    = parquet_loader.load_household_tpnb_period(HOUSEHOLD_TPNB_PERIOD_PARQUET)

# ─────────────────────────────────────────────
# Stage 1: build whole baskets (no category split), train the GNN
# ─────────────────────────────────────────────

print("\n[Stage 1] Building features...")
tpnb_x_hh["year_period_number"] = tpnb_x_hh["year_number"] * 100 + tpnb_x_hh["period_number"]

print("Building baskets (whole basket — every product a household bought in "
      "the period, no category/theme split)...")
baskets = (
    tpnb_x_hh
    .groupby(["household_number", "year_period_number"])
    .agg(products=("tpnb", list), units=("quantity", list))
    .reset_index()
)
baskets["basket_id"] = (
    baskets["household_number"].astype(str) + "_" +
    baskets["year_period_number"].astype(str)
)
before_filter = len(baskets)
baskets = baskets[baskets["products"].apply(len) >= MIN_BASKET_PRODUCTS].reset_index(drop=True)
print(f"  Full baskets: {len(baskets):,} "
      f"({before_filter - len(baskets):,} dropped with < {MIN_BASKET_PRODUCTS} products)")

# This DataFrame is held in memory for the ENTIRE rest of this script — it's
# needed both as GNN training input and, afterward, to embed every single
# basket. Its size doesn't shrink at any later point. A rough estimate: at
# ~30 products/basket, roughly 3-4KB per row once you account for real
# per-object Python overhead in the products/units list cells — 20 million+
# baskets is on the order of 50-100GB for this one object alone, before
# anything else the rest of the script needs. If you're not intentionally
# running the full, un-sampled population, check sql/04's
# MOD(household_number, N) = 0 filter is actually being applied — this
# number should look dramatically smaller than your household count.
BASKET_COUNT_WARN_THRESHOLD = 2_000_000
if len(baskets) > BASKET_COUNT_WARN_THRESHOLD:
    est_gb = len(baskets) * 3500 / 1e9
    print(f"  WARNING: {len(baskets):,} baskets is large enough to plausibly run this "
          f"machine out of memory later in the run (rough estimate: ~{est_gb:.0f}GB just "
          f"for this DataFrame, held for the rest of the script). If this wasn't "
          f"intentional, stop now (Ctrl+C) and check sql/04's household-sampling filter "
          f"is actually being applied to the data you downloaded.")

print("Building co-purchase matrix...")
basket_items = baskets.explode("products").rename(columns={"products": "product_id"})
basket_items = basket_items.drop_duplicates(["basket_id", "product_id"])
basket_codes, basket_uniques   = pd.factorize(basket_items["basket_id"])
product_codes, product_uniques = pd.factorize(basket_items["product_id"])
X = csr_matrix(
    (np.ones(len(basket_items), dtype=np.int32), (basket_codes, product_codes)),
    shape=(len(basket_uniques), len(product_uniques)),
)
copurchase_sparse    = X.T @ X
product_id_to_index  = {pid: idx for idx, pid in enumerate(product_uniques)}

# basket_items (and the intermediate X) are only needed to build
# copurchase_sparse — not used again. Freeing them here rather than letting
# them sit until the script ends matters at real data scale.
del basket_items, X, basket_codes, product_codes
gc.collect()

product_embedding = dict(zip(product_df_2["tpnb"], product_df_2["embedding"]))
product_units_avg = tpnb_x_hh.groupby("tpnb")["quantity"].mean().to_dict()

# tpnb_x_hh (the full raw purchase table — tens of millions of rows even at
# a 1% household sample) isn't needed again after this. baskets IS still
# needed later (passed into train_and_embed, then reused for embedding every
# basket after training), so that one stays.
del tpnb_x_hh
gc.collect()

# train_and_embed() only saves the model + training embeddings — it does NOT
# persist these three inputs, but score_new_baskets.py (scoring new baskets
# against the trained model later) needs exactly these three files on disk
# under exactly these names. Save them now while they're in scope. No
# product_theme to save — that concept doesn't exist in this pipeline.
print("Saving inference-time artifacts (needed later to score new baskets)...")
sp.save_npz("copurchase_sparse.npz", copurchase_sparse.tocsr())
with open("product_id_to_index.pkl", "wb") as f:
    pickle.dump(product_id_to_index, f)
with open("product_units_avg.pkl", "wb") as f:
    pickle.dump(product_units_avg, f)
print("  Saved copurchase_sparse.npz, product_id_to_index.pkl, product_units_avg.pkl")

print("\n[Stage 1] Training GNN + embedding baskets...")
basket_gnn_embeddings = train_and_embed(
    baskets              = baskets,
    product_embedding    = product_embedding,
    product_id_to_index  = product_id_to_index,
    copurchase_sparse    = copurchase_sparse,
    product_units_avg    = product_units_avg,
)

# ─────────────────────────────────────────────
# Stage 2: cluster the basket embeddings into need-states — Leiden AND GMM
# ─────────────────────────────────────────────

print("\n[Stage 2a] Leiden clustering...")
leiden_clusters = cluster_basket_embeddings(basket_gnn_embeddings)

print("\n[Stage 2b] GMM clustering (comparison / combination method)...")
gmm_clusters = cluster_basket_embeddings_gmm(basket_gnn_embeddings, n_components=GMM_N_COMPONENTS)

print("\n[Stage 2c] Comparing the two methods...")
compare_leiden_gmm(leiden_clusters, gmm_clusters)

# Both label sets kept side by side — need_state_cluster (Leiden) and
# need_state_cluster_gmm (GMM) — rather than collapsing to one, since the
# real decision (compare vs. combine, and how) isn't settled yet.
need_state_clusters = leiden_clusters.merge(gmm_clusters, on="basket_id", how="outer")
need_state_clusters.to_parquet("basket_need_state_clusters.parquet", index=False)
print(f"  Saved basket_need_state_clusters.parquet ({len(need_state_clusters):,} rows, "
      f"columns: need_state_cluster [Leiden], need_state_cluster_gmm [GMM])")

# ─────────────────────────────────────────────
# Stage 3: reload into your warehouse (manual)
# ─────────────────────────────────────────────
# No live write-back connection here, same as Stage 0. Two options:
#   1. Run sql/06_write_back_need_states.sql once to create the landing table,
#      then upload basket_need_state_clusters.parquet through your workspace
#      web tool's import feature.
#   2. If your web tool doesn't support parquet import, re-save as CSV first:
#        need_state_clusters.to_csv("basket_need_state_clusters.csv", index=False)

print("\n[Stage 3] Skipped — reload basket_need_state_clusters.parquet into "
      "your warehouse manually (see sql/06_write_back_need_states.sql).")

print("\nPipeline complete.")

# Themes, if wanted at all, are formed AFTER this point — by profiling the
# need-states this run produced (e.g. summarizing each need_state_cluster's
# dominant products/hierarchy) — never fed back in as an input. That
# profiling step isn't implemented in this repository yet.