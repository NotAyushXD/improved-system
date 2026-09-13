"""
score_new_baskets.py

Scores NEW baskets (a fresh period, a held-out sample — whatever "new"
means for your check) against the already-trained GNN and the need-states
already discovered by pipeline_main.py, without retraining anything.

THEME-FREE: does not read, write, or depend on any theme/category concept.
There is no product_theme.pkl anymore — an earlier version of this file
loaded one; that file is no longer written by pipeline_main.py and is not
needed here, because GraphBuilder.py's prepare_globals() no longer takes a
product_theme argument at all (see REFACTOR_NOTES.md).

Basket storage and inference now go through the SAME shared code path
pipeline_main.py uses (basket_store.py + GraphBuilder.run_inference), just
tagged "score" instead of "train" — this is deliberate: an earlier version
of this file had its own separate basket-building implementation, which is
exactly the kind of train/score drift this pipeline has been bitten by
before (see REFACTOR_NOTES.md's account of the old inference shortcut that
computed a different feature than training did). One shared implementation
means these two entry points can't silently diverge again.

Embeds new baskets via GraphBuilder.run_inference(), which builds a REAL
per-basket graph for every basket (using build_one_graph — the exact same
function training uses) and runs it through the model's full encode()
pipeline (node_encoder -> conv1 -> conv2 -> global_mean_pool -> proj) —
guaranteeing this file can't compute different features, or use a
different encoding path, than training did. Inference is restartable and
never holds more than one chunk's worth of baskets in memory, same as
training's inference pass.

Assigns each new basket to a need-state using BOTH methods, since that's
how need-states are decided here:
  - Leiden: via cluster_basket_embeddings.assign_new_baskets_to_clusters()
    (k-NN majority vote — Leiden has no native way to place a new point into
    an already-found community)
  - GMM: via the saved gmm_basket_model.pkl's own .predict() — a fitted GMM
    scores new points directly, no workaround needed

Requires these files to already exist under data/output/, from a prior
pipeline_main.py run:
    basket_gnn_model.pt
    product_id_to_index.pkl
    copurchase_sparse.npz
    product_units_avg.pkl
    basket_gnn_embeddings.parquet
    basket_need_state_clusters.parquet
    gmm_basket_model.pkl        (optional — GMM scoring skipped if absent)

Usage:
    python score_new_baskets.py --new-transactions ../data/ns_household_tpnb_week_agg_score
"""

import argparse
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize

import parquet_loader
import duckdb_manager
import basket_store
from GraphBuilder import prepare_globals, run_inference, merge_inference_output
from GNN_Train import BasketGNN
from cluster_basket_embeddings import assign_new_baskets_to_clusters

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# All pipeline-produced artifacts (from a prior pipeline_main.py run, and
# this script's own outputs) live under this folder, not the working directory.
OUTPUT_DIR = "../data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH         = os.path.join(OUTPUT_DIR, "basket_gnn_model.pt")
PRODUCT_IDX_PATH   = os.path.join(OUTPUT_DIR, "product_id_to_index.pkl")
COPURCHASE_PATH    = os.path.join(OUTPUT_DIR, "copurchase_sparse.npz")
PRODUCT_UNITS_PATH = os.path.join(OUTPUT_DIR, "product_units_avg.pkl")

MIN_BASKET_PRODUCTS = 2   # same floor pipeline_main.py applies at training time

# Same file used in training (Stage 0 of pipeline_main.py) — reusing it here,
# rather than a fresh pull, keeps product embeddings identical between train
# and score so nothing shifts underneath the model.
PRODUCT_EMBEDDINGS_PARQUET = os.path.join(OUTPUT_DIR, "product_embeddings.parquet")

EXISTING_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "basket_gnn_embeddings.parquet")
EXISTING_CLUSTERS_PATH   = os.path.join(OUTPUT_DIR, "basket_need_state_clusters.parquet")
GMM_MODEL_PATH           = os.path.join(OUTPUT_DIR, "gmm_basket_model.pkl")

NEW_EMBEDDINGS_OUT    = os.path.join(OUTPUT_DIR, "new_basket_gnn_embeddings.parquet")
NEW_CLUSTERS_OUT      = os.path.join(OUTPUT_DIR, "new_basket_need_states.parquet")
MERGED_EMBEDDINGS_OUT = os.path.join(OUTPUT_DIR, "basket_gnn_embeddings_merged.parquet")
MERGED_CLUSTERS_OUT   = os.path.join(OUTPUT_DIR, "basket_need_state_clusters_merged.parquet")

EMBED_BATCH_SIZE = 8192
DATASET_TAG = "score"


# ─────────────────────────────────────────────
# LOAD MODEL + GLOBALS
# ─────────────────────────────────────────────

def load_globals_and_model():
    print("Loading persisted training artifacts...")
    with open(PRODUCT_IDX_PATH, "rb") as f:
        product_id_to_index = pickle.load(f)
    with open(PRODUCT_UNITS_PATH, "rb") as f:
        product_units_avg = pickle.load(f)
    csr_raw = sp.load_npz(COPURCHASE_PATH)

    product_df_2 = parquet_loader.load_product_embeddings(PRODUCT_EMBEDDINGS_PARQUET)
    product_embedding = dict(zip(product_df_2["tpnb"], product_df_2["embedding"]))

    G = prepare_globals(
        product_embedding   = product_embedding,
        product_id_to_index = product_id_to_index,
        copurchase_sparse   = csr_raw,
        product_units_avg   = product_units_avg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GNN model from {MODEL_PATH} (device={device})...")
    state = torch.load(MODEL_PATH, map_location=device)

    # Infer architecture dims from the saved weights — avoids hardcoding
    # dims that could drift from whatever GNN_Train.py's CONFIG actually
    # used for this checkpoint.
    enc_key     = next(k for k in state if "node_encoder" in k and "weight" in k)
    in_dim_ckpt = int(state[enc_key].shape[1])
    hidden_dim  = int(state[enc_key].shape[0])
    proj_keys   = [k for k in state if "proj" in k and "weight" in k]
    out_dim     = int(state[proj_keys[-1]].shape[0])
    lin_key     = next((k for k in state if "conv1.lin" in k and "weight" in k), None)
    edge_dim    = int(state[lin_key].shape[1]) - hidden_dim if lin_key else 2

    if in_dim_ckpt != G["in_dim"]:
        print(f"  WARNING: prepare_globals() computed in_dim={G['in_dim']} but the "
              f"checkpoint expects in_dim={in_dim_ckpt}. If this checkpoint was trained "
              f"before the theme-free refactor (in_dim was emb_dim+5, now emb_dim+4), "
              f"it is NOT compatible — retrain rather than continuing. Otherwise this "
              f"usually means the product embedding table changed since training — "
              f"using the checkpoint's value, but double check product_embeddings.parquet "
              f"still matches what training used.")

    model = BasketGNN(
        in_dim=in_dim_ckpt, edge_dim=edge_dim,
        hidden_dim=hidden_dim, out_dim=out_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Model loaded: in_dim={in_dim_ckpt} hidden={hidden_dim} out={out_dim} edge_dim={edge_dim}")

    return G, model, device


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-transactions", required=True,
        help="Parquet downloaded after running ns_household_tpnb_week_agg_score.sql on your warehouse",
    )
    parser.add_argument(
        "--worker-id", default=None,
        help="Label for this process in the restartable inference chunk queue "
             "(defaults to hostname-pid) — see pipeline_main.py's --worker-id for details.",
    )
    args = parser.parse_args()

    G, model, device = load_globals_and_model()

    con = duckdb_manager.get_connection()

    print(f"Building baskets from {args.new_transactions}...")
    # Same whole-basket construction pipeline_main.py's Stage 0 uses — basket
    # grain is WEEK (household_number x year_week_number), not a true
    # single-visit basket. See the "BASKET GRAIN" note at the top of
    # pipeline_main.py and data/ns_household_tpnb_week_agg_train.sql.
    # DuckDB reads the parquet export directly — no separate load step.
    # product_units_avg returned here is discarded — this script already has
    # its own from training (PRODUCT_UNITS_PATH), and reusing that one
    # (rather than a fresh one computed only from the new/held-out data) is
    # what keeps product features consistent between train and score.
    n_baskets_total, _discarded_units_avg, _discarded_products = basket_store.build_baskets_table(
        con, args.new_transactions, DATASET_TAG, min_basket_products=MIN_BASKET_PRODUCTS,
    )
    print(f"  {n_baskets_total:,} new baskets built (before already-embedded exclusion)")

    # Same exclusion the old in-memory version did (`~isin(existing_ids)`),
    # but done as a SQL anti-join against a streamed basket_id-only table —
    # `existing_ids` at real scale is the entire training basket population
    # (tens of millions of rows), which is exactly the kind of object this
    # whole redesign exists to avoid materializing as a Python set.
    basket_store.exclude_existing_basket_ids(con, DATASET_TAG, EXISTING_EMBEDDINGS_PATH)
    n_to_score = basket_store.count_baskets(con, DATASET_TAG)
    print(f"  Need embedding: {n_to_score:,}")

    if n_to_score == 0:
        print("All baskets already embedded — nothing to score.")
        return

    print(f"\nEmbedding {n_to_score:,} baskets via full GNN encoding "
          f"(restartable, chunked from DuckDB)...")
    run_inference(
        con, DATASET_TAG, G, model, device,
        worker_id=args.worker_id, batch_size=EMBED_BATCH_SIZE,
    )
    new_embeddings = merge_inference_output(
        DATASET_TAG, output_dir=OUTPUT_DIR, final_path=NEW_EMBEDDINGS_OUT,
    )
    print(f"Saved {NEW_EMBEDDINGS_OUT} ({len(new_embeddings):,} rows)")

    print("\nAssigning new baskets to existing need-state clusters (Leiden, k-NN vote)...")
    reference_embeddings = pd.read_parquet(EXISTING_EMBEDDINGS_PATH)
    reference_clusters   = pd.read_parquet(EXISTING_CLUSTERS_PATH)
    new_clusters = assign_new_baskets_to_clusters(
        new_embeddings, reference_embeddings, reference_clusters
    )

    if Path(GMM_MODEL_PATH).exists():
        print(f"\nAssigning new baskets via the saved GMM model ({GMM_MODEL_PATH})...")
        gmm = joblib.load(GMM_MODEL_PATH)
        X = normalize(np.stack(new_embeddings["gnn_embedding"].values))
        gmm_labels = gmm.predict(X)
        gmm_probs  = gmm.predict_proba(X).max(axis=1)
        new_clusters["need_state_cluster_gmm"] = gmm_labels
        new_clusters["gmm_confidence"] = gmm_probs
        print(f"  GMM assigned {len(new_clusters):,} baskets across "
              f"{new_clusters['need_state_cluster_gmm'].nunique()} existing components")
    else:
        print(f"\n{GMM_MODEL_PATH} not found — skipping GMM assignment "
              f"(only Leiden clusters will be in the output).")

    new_clusters.to_parquet(NEW_CLUSTERS_OUT, index=False)
    print(f"Saved {NEW_CLUSTERS_OUT} ({len(new_clusters):,} rows)")

    print("\nMerging into master tables...")
    merged_emb = (
        pd.concat([reference_embeddings, new_embeddings], ignore_index=True)
        .drop_duplicates("basket_id", keep="last")
    )
    merged_emb.to_parquet(MERGED_EMBEDDINGS_OUT, index=False)

    merged_clusters = (
        pd.concat(
            [reference_clusters, new_clusters.drop(columns=["cluster_confidence"])],
            ignore_index=True,
        )
        .drop_duplicates("basket_id", keep="last")
    )
    merged_clusters.to_parquet(MERGED_CLUSTERS_OUT, index=False)

    print(f"  {MERGED_EMBEDDINGS_OUT}: {len(merged_emb):,} rows")
    print(f"  {MERGED_CLUSTERS_OUT}: {len(merged_clusters):,} rows")
    print("\nDone. Re-upload the *_merged parquet files through your workspace "
          "if you want the new baskets reflected in the warehouse too.")


if __name__ == "__main__":
    main()
