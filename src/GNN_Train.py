"""
GNN_Train.py — Sample + Inductive GNN

Theme-free: no theme/category input feeds training or scoring anywhere in
this file. Node features come from product embeddings, co-purchase
structure, and GLOBAL product sub-clustering (see GraphBuilder.py). Scoring
(GraphBuilder.run_inference / _embed_basket_chunk) builds real per-basket
graphs and runs the full encode() pipeline below — the same one training
uses — so training and scoring can't drift apart in either features or
encoding. See REFACTOR_NOTES.md for the full account of what changed and why.

Basket storage: `baskets` is never passed into this file as one in-memory
object anymore — it lives in a local embedded DuckDB database
(basket_store.py, duckdb_manager.py), addressed by `con` (the shared
connection object) + `dataset_tag`. The training sample is drawn there
(bounded, ~N_TRAIN_SAMPLES rows), cached once to LMDB (lmdb_graph_cache.py)
so DataLoader can randomly access it across epochs without holding all
built graphs in RAM, and inference streams the full population from DuckDB
in restartable chunks (GraphBuilder.run_inference) — restartable for a
single process; DuckDB has no row-level locking, so this isn't safe for
multiple concurrent processes claiming chunks at once (see
basket_store.claim_next_chunk).
"""

import json
import os
import socket
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import joblib

from GraphBuilder import (prepare_globals, run_inference, merge_inference_output,
                          GRAPH_BUILDER_VERSION)
import basket_store
import lmdb_graph_cache
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HIDDEN_DIM      = config.HIDDEN_DIM
OUT_DIM         = config.OUT_DIM
DROPOUT         = config.DROPOUT
EDGE_DIM        = config.EDGE_DIM
TRAIN_BATCH     = config.TRAIN_BATCH
EPOCHS          = config.EPOCHS
LR              = config.LR   # default 1e-4, reduced from 1e-3 — prevents divergence
WEIGHT_DECAY    = config.WEIGHT_DECAY
N_TRAIN_SAMPLES = config.N_TRAIN_SAMPLES  # default 300_000 — was 1_000_000 — reduced as an extra memory safety
                            # margin on a shared machine. With the LMDB-backed
                            # dataset (build once, random-access during
                            # training, never all resident in RAM) this is no
                            # longer a hard memory constraint — raise it back
                            # up if more training data is wanted, independent
                            # of RAM.
NUM_WORKERS     = config.NUM_WORKERS   # see the __main__ guard note
                            # in pipeline_main.py — nonzero DataLoader workers
                            # on Windows require that guard to be in place,
                            # since `spawn` re-imports/re-executes the
                            # launching module in every worker process.

# All pipeline-produced artifacts land here, not the working directory.
OUTPUT_DIR      = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

LMDB_TRAINING_GRAPHS_PATH = os.path.join(OUTPUT_DIR, "training_graphs.lmdb")
LMDB_MANIFEST_PATH        = os.path.join(OUTPUT_DIR, "training_graphs.lmdb.manifest.json")

# Sidecar recording what the saved model was trained under, so a rerun can
# reuse it rather than spending hours retraining — and so it is never reused
# after a setting that affects the weights has changed. Same pattern as the
# LMDB training-graph manifest.
MODEL_MANIFEST_PATH       = os.path.join(OUTPUT_DIR, "basket_gnn_model.manifest.json")


def _manifest_matches(path: str, expected: dict) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) == expected
    except (OSError, ValueError):
        return False


def _write_manifest(path: str, manifest: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    os.replace(tmp, path)   # atomic — never leaves a truncated manifest


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class BasketGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=128, out_dim=64, dropout=0.1):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)

        nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = F.relu(self.node_encoder(x))
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        g = global_mean_pool(x, batch)
        return self.proj(g)

    def forward(self, data):
        z     = self.encode(data)
        recon = self.decoder(z)
        return z, recon


def batch_graph_targets(data):
    x, batch   = data.x, data.batch
    num_graphs = int(batch.max().item()) + 1
    return torch.stack([x[batch == g].mean(dim=0) for g in range(num_graphs)])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def train_and_embed(
    con,
    dataset_tag,
    product_embedding,
    product_id_to_index,
    copurchase_sparse,
    product_units_avg,
    worker_id=None,
):
    worker_id = worker_id or _default_worker_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Weight init, dropout masks and DataLoader shuffling all draw from torch's
    # global RNG, which was previously left unseeded — so two runs on identical
    # data produced different models, and a lost checkpoint could never be
    # reconstructed. Seeding makes a retrain reproducible, which is what turns
    # "the model file was lost" from unrecoverable into merely annoying.
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # ── Step 1: Prepare globals ──
    print("[ 1 / 4 ] Preparing globals...")
    G = prepare_globals(
        product_embedding   = product_embedding,
        product_id_to_index = product_id_to_index,
        copurchase_sparse   = copurchase_sparse,
        product_units_avg   = product_units_avg,
    )
    in_dim = G["in_dim"]   # emb_dim + 4
    print(f"  Node feature dim: {in_dim}  (emb_dim={G['emb_dim']} + 4 extra features)")

    # ── Step 2: Sample training baskets (from DuckDB, bounded result size) ──
    print("\n[ 2 / 4 ] Sampling training baskets...")
    sampled = basket_store.sample_training_baskets(
        con, dataset_tag, n_samples=N_TRAIN_SAMPLES, seed=config.SEED,
    )

    # ── Step 3: Build (or reuse) the LMDB training-graph cache ──
    # Each basket's co-purchase submatrix is built on the fly, scoped to just
    # that basket's own products — see GraphBuilder._basket_dense_cp_submatrix
    # — so there's no whole-sample dense matrix to build (or free) here.
    print("\n[ 3 / 4 ] Building (or reusing) LMDB training-graph cache...")
    lmdb_graph_cache.load_or_build_lmdb_cache(
        sampled, G, LMDB_TRAINING_GRAPHS_PATH, LMDB_MANIFEST_PATH,
        seed=config.SEED, n_train_samples_requested=N_TRAIN_SAMPLES,
    )
    train_dataset = lmdb_graph_cache.LMDBGraphDataset(LMDB_TRAINING_GRAPHS_PATH)
    print(f"  Training graphs ready: {len(train_dataset):,} (LMDB-backed, lazy random access)")

    # ── Step 4: Train (or reuse an already-trained model) ──
    model_path = os.path.join(OUTPUT_DIR, "basket_gnn_model.pt")

    model = BasketGNN(
        in_dim     = in_dim,
        edge_dim   = EDGE_DIM,
        hidden_dim = HIDDEN_DIM,
        out_dim    = OUT_DIM,
        dropout    = DROPOUT,
    ).to(device)

    # Reuse a matching trained model instead of retraining from scratch.
    # Training takes hours, and inference (which follows it) takes longer still
    # — so ANY restart of the inference pass used to pay for a full retrain
    # first, even though a perfectly good checkpoint was sitting on disk. Worse,
    # retraining produces a DIFFERENT model, so chunks already embedded by the
    # previous model would be silently mixed with chunks from the new one.
    # The manifest covers everything the weights depend on: architecture,
    # optimiser settings, epochs, sample size, seed, and the graph fingerprint
    # (so a change to TOP_K or sub-clustering forces a retrain, exactly as it
    # forces an LMDB rebuild). Delete basket_gnn_model.pt to force a retrain.
    expected_model_manifest = {
        "in_dim": in_dim, "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM,
        "edge_dim": EDGE_DIM, "dropout": DROPOUT, "epochs": EPOCHS,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "train_batch": TRAIN_BATCH,
        "n_train_samples": len(train_dataset), "seed": config.SEED,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_fingerprint": config.graph_fingerprint(),
    }

    if os.path.exists(model_path) and _manifest_matches(MODEL_MANIFEST_PATH, expected_model_manifest):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n[ 4 / 4 ] REUSING trained model from {model_path} "
              f"(manifest matches — same architecture, settings, seed and graph "
              f"fingerprint). Skipping {EPOCHS} epochs of training.")
        print(f"          Delete {model_path} if you want to retrain from scratch.")
        train_loader = None
    else:
        if os.path.exists(model_path):
            print(f"\n  IGNORING {model_path}: its manifest doesn't match this run "
                  f"(architecture, training settings, seed, or graph construction "
                  f"changed) — retraining rather than embedding with a model built "
                  f"under different settings.")
        print(f"\n[ 4 / 4 ] Training GNN for {EPOCHS} epochs...")
        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH,
            shuffle=True, num_workers=NUM_WORKERS,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler(enabled=device.type == "cuda")
    model.train()

    epoch_bar = tqdm(range(EPOCHS if train_loader is not None else 0),
                     desc="Epochs", unit="epoch", disable=train_loader is None)
    for epoch in epoch_bar:
        total_loss  = 0.0
        nan_batches = 0
        batch_bar   = tqdm(train_loader, desc=f"  Epoch {epoch+1:>3}/{EPOCHS}",
                           unit="batch", leave=False)
        for batch_data in batch_bar:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            with autocast(enabled=device.type == "cuda"):
                z, recon = model(batch_data)
                target   = batch_graph_targets(batch_data)

                # Sanitize target — NaN in any node feature poisons the whole batch
                if not torch.isfinite(target).all():
                    nan_batches += 1
                    continue

                loss = F.mse_loss(recon, target)

            if not torch.isfinite(loss):
                nan_batches += 1
                continue

            scaler.scale(loss).backward()
            # Gradient clipping — prevents exploding gradients causing NaN weights
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / max(len(train_loader) - nan_batches, 1)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")
        if nan_batches > 0:
            print(f"  Warning: {nan_batches} NaN batches skipped in epoch {epoch+1}")

    # ── Inference (restartable, bounded-memory, one chunk at a time) ──
    # Streams the FULL basket population from DuckDB in claimed chunks,
    # builds a real graph per chunk, runs it through the SAME model.encode()
    # path used during training, and writes each chunk's embeddings straight
    # to disk — see GraphBuilder.run_inference. A crash partway through
    # resumes from whatever chunks are still pending/stale rather than
    # starting over (single-process only — see basket_store.claim_next_chunk).
    # ── Save model BEFORE inference ──
    # This used to happen AFTER run_inference() returned, which quietly
    # defeated the whole point of making inference restartable. Inference over
    # a full basket population takes hours-to-days; if the process died (or was
    # stopped to apply a fix) at any point during it, the trained weights —
    # which existed only in this process's memory — were lost for good. Every
    # chunk already embedded then became an orphan: its embeddings came from a
    # model that no longer exists and, with no torch seed set, cannot be
    # reproduced by retraining. The chunk queue would happily "resume", but the
    # remaining chunks would be embedded by DIFFERENT weights than the
    # completed ones, silently mixing two incompatible embedding spaces into
    # one clustering.
    #
    # Saving here makes the restartability real: the weights survive the
    # process, so a killed run genuinely resumes where it left off.
    if train_loader is not None:
        torch.save(model.state_dict(), model_path)
        _write_manifest(MODEL_MANIFEST_PATH, expected_model_manifest)
        print(f"\nModel saved to {model_path} BEFORE inference — a crash or stop "
              f"during the (long) inference pass can now resume against these exact "
              f"weights instead of losing them, AND a rerun will reuse this model "
              f"rather than retraining.")

    print("\n[ Inference ] Embedding all baskets via full GNN encoding "
          "(restartable, chunked from DuckDB)...")
    run_inference(
        con, dataset_tag, G, model, device,
        worker_id=worker_id,
    )
    basket_gnn_embeddings = merge_inference_output(dataset_tag, output_dir=OUTPUT_DIR)

    print("\nDone. Saved:")
    print(f"  {os.path.join(OUTPUT_DIR, 'basket_gnn_embeddings.parquet')}  "
          f"({len(basket_gnn_embeddings):,} rows)")
    print(f"  {model_path}")

    return basket_gnn_embeddings
