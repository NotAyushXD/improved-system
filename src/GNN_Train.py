"""
GNN_Train.py — Sample + Inductive GNN

Theme-free: no theme/category input feeds training or scoring anywhere in
this file. Node features come from product embeddings, co-purchase
structure, and GLOBAL product sub-clustering (see GraphBuilder.py). Scoring
(embed_all_baskets_fast, in GraphBuilder.py) builds real per-basket graphs
and runs the full encode() pipeline below — the same one training uses —
so training and scoring can't drift apart in either features or encoding.
See REFACTOR_NOTES.md for the full account of what changed and why.
"""

import os
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

from GraphBuilder import (
    prepare_globals,
    sample_baskets,
    build_dense_cp_submatrix,
    build_training_graphs,
    save_training_graphs,
    embed_all_baskets_fast,
)

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
                            # margin on a shared machine, on top of the
                            # dense_cp-release fix above. 300k graphs is still
                            # plenty for training; raise it back up once a run
                            # succeeds cleanly and you've confirmed headroom.
NUM_WORKERS     = 0 if os.name == "nt" else 4


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
    baskets,
    product_embedding,
    product_id_to_index,
    copurchase_sparse,
    product_units_avg,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Step 1: Prepare globals ──
    print("[ 1 / 5 ] Preparing globals...")
    G = prepare_globals(
        product_embedding   = product_embedding,
        product_id_to_index = product_id_to_index,
        copurchase_sparse   = copurchase_sparse,
        product_units_avg   = product_units_avg,
    )
    in_dim = G["in_dim"]   # emb_dim + 4
    print(f"  Node feature dim: {in_dim}  (emb_dim={G['emb_dim']} + 4 extra features)")

    # ── Step 2: Sample baskets ──
    print("\n[ 2 / 5 ] Sampling training baskets...")
    sampled = sample_baskets(baskets, n_samples=N_TRAIN_SAMPLES)

    # ── Step 3: Dense co-purchase submatrix ──
    print("\n[ 3 / 5 ] Building dense co-purchase submatrix...")
    dense_cp, local_idx, _ = build_dense_cp_submatrix(
        sampled, G["product_id_to_index"], G["csr"]
    )

    # ── Step 4: Build training graphs ──
    print("\n[ 4 / 5 ] Building training graphs...")
    graph_list = build_training_graphs(sampled, G, dense_cp, local_idx)
    print(f"  Training graphs ready: {len(graph_list):,}")

    # dense_cp can be a very large dense matrix (tens of GB at real data
    # scale) — it's not needed again after this point, but stayed alive here
    # for the rest of the function before this fix, right through the
    # disk-write below, which is exactly when peak memory is highest.
    del dense_cp, local_idx
    import gc
    gc.collect()

    save_training_graphs(graph_list)

    # ── Step 5: Train ──
    print(f"\n[ 5 / 5 ] Training GNN for {EPOCHS} epochs...")
    train_loader = DataLoader(
        graph_list, batch_size=TRAIN_BATCH,
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

        avg_loss = total_loss / len(train_loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")

    # ── Inductive inference ──
    # Builds a real graph per basket and runs the SAME model.encode() path
    # used during training (node_encoder -> conv1 -> conv2 -> pool -> proj) —
    # not a hand-rolled pooling shortcut. See GraphBuilder.embed_all_baskets_fast.
    print("\n[ Inference ] Embedding all baskets via full GNN encoding...")
    basket_ids, all_z = embed_all_baskets_fast(
        baskets, G, model, device, batch_size=8192
    )

    # ── Save ──
    basket_gnn_embeddings = pd.DataFrame({
        "basket_id":     basket_ids,
        "gnn_embedding": list(all_z),
    })
    basket_gnn_embeddings.to_parquet("basket_gnn_embeddings.parquet", index=False)
    torch.save(model.state_dict(), "basket_gnn_model.pt")

    print("\nDone. Saved:")
    print(f"  basket_gnn_embeddings.parquet  ({len(basket_gnn_embeddings):,} rows)")
    print(f"  basket_gnn_model.pt")

    return basket_gnn_embeddings