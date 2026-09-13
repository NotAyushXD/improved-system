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
object anymore — it lives in Postgres (basket_store.py), addressed by
`conn_uri` + `dataset_tag`. The training sample is drawn there (bounded,
~N_TRAIN_SAMPLES rows), cached once to LMDB (lmdb_graph_cache.py) so
DataLoader can randomly access it across epochs without holding all built
graphs in RAM, and inference streams the full population from Postgres in
restartable chunks (GraphBuilder.run_inference).
"""

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

from GraphBuilder import prepare_globals, run_inference, merge_inference_output
import basket_store
import lmdb_graph_cache

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HIDDEN_DIM      = 128
OUT_DIM         = 64
DROPOUT         = 0.1
EDGE_DIM        = 2
TRAIN_BATCH     = 256
EPOCHS          = 20
LR              = 1e-4   # reduced from 1e-3 — prevents divergence with new features
WEIGHT_DECAY    = 1e-5
N_TRAIN_SAMPLES = 300_000  # was 1_000_000 — reduced as an extra memory safety
                            # margin on a shared machine. With the LMDB-backed
                            # dataset (build once, random-access during
                            # training, never all resident in RAM) this is no
                            # longer a hard memory constraint — raise it back
                            # up if more training data is wanted, independent
                            # of RAM.
NUM_WORKERS     = 0 if os.name == "nt" else 4   # see the __main__ guard note
                            # in pipeline_main.py — nonzero DataLoader workers
                            # on Windows require that guard to be in place,
                            # since `spawn` re-imports/re-executes the
                            # launching module in every worker process.

# All pipeline-produced artifacts land here, not the working directory.
OUTPUT_DIR      = "../data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LMDB_TRAINING_GRAPHS_PATH = os.path.join(OUTPUT_DIR, "training_graphs.lmdb")
LMDB_MANIFEST_PATH        = os.path.join(OUTPUT_DIR, "training_graphs.lmdb.manifest.json")


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
    conn_uri,
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

    # ── Step 2: Sample training baskets (from Postgres, bounded result size) ──
    print("\n[ 2 / 4 ] Sampling training baskets...")
    sampled = basket_store.sample_training_baskets(
        conn_uri, dataset_tag, n_samples=N_TRAIN_SAMPLES, seed=42,
    )

    # ── Step 3: Build (or reuse) the LMDB training-graph cache ──
    # Each basket's co-purchase submatrix is built on the fly, scoped to just
    # that basket's own products — see GraphBuilder._basket_dense_cp_submatrix
    # — so there's no whole-sample dense matrix to build (or free) here.
    print("\n[ 3 / 4 ] Building (or reusing) LMDB training-graph cache...")
    lmdb_graph_cache.load_or_build_lmdb_cache(
        sampled, G, LMDB_TRAINING_GRAPHS_PATH, LMDB_MANIFEST_PATH,
        seed=42, n_train_samples_requested=N_TRAIN_SAMPLES,
    )
    train_dataset = lmdb_graph_cache.LMDBGraphDataset(LMDB_TRAINING_GRAPHS_PATH)
    print(f"  Training graphs ready: {len(train_dataset):,} (LMDB-backed, lazy random access)")

    # ── Step 4: Train ──
    print(f"\n[ 4 / 4 ] Training GNN for {EPOCHS} epochs...")
    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH,
        shuffle=True, num_workers=NUM_WORKERS,
    )

    model = BasketGNN(
        in_dim     = in_dim,
        edge_dim   = EDGE_DIM,
        hidden_dim = HIDDEN_DIM,
        out_dim    = OUT_DIM,
        dropout    = DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler(enabled=device.type == "cuda")
    model.train()

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")
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
    # Streams the FULL basket population from Postgres in claimed chunks,
    # builds a real graph per chunk, runs it through the SAME model.encode()
    # path used during training, and writes each chunk's embeddings straight
    # to disk — see GraphBuilder.run_inference. A crash partway through
    # resumes from whatever chunks are still pending/stale rather than
    # starting over.
    print("\n[ Inference ] Embedding all baskets via full GNN encoding "
          "(restartable, chunked from Postgres)...")
    run_inference(
        conn_uri, dataset_tag, G, model, device,
        worker_id=worker_id, batch_size=8192,
    )
    basket_gnn_embeddings = merge_inference_output(dataset_tag, output_dir=OUTPUT_DIR)

    # ── Save model ──
    model_path = os.path.join(OUTPUT_DIR, "basket_gnn_model.pt")
    torch.save(model.state_dict(), model_path)

    print("\nDone. Saved:")
    print(f"  {os.path.join(OUTPUT_DIR, 'basket_gnn_embeddings.parquet')}  "
          f"({len(basket_gnn_embeddings):,} rows)")
    print(f"  {model_path}")

    return basket_gnn_embeddings
