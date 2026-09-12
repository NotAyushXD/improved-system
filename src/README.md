# `src/` — Code Files Reference

## Overview

This is a GNN-based basket need-state clustering pipeline. It embeds retail shopping baskets using a Graph Neural Network trained on co-purchase relationships, then clusters the resulting embeddings into "need-states" (customer intent groups) using two complementary methods: Leiden community detection and Gaussian Mixture Models (GMM).

---

## Pipeline Execution Order

```
cd src/

# Step 1 — Build product embeddings (run once, or whenever the product catalogue changes)
python build_product_embeddings.py

# Step 2 — Run the full end-to-end pipeline (train GNN + cluster baskets)
python pipeline_main.py

# Step 3 (optional) — Score new baskets against the already-trained model
python score_new_baskets.py --new-transactions ../data/ns_household_tpnb_period_agg_score
```

> **Prerequisites:** Before running, export the required warehouse tables as parquet into `data/` (see each script's path constants). No live database connection is used anywhere.

---

## File-by-File Reference

---

### `pipeline_main.py`

**What it does:** The top-level orchestrator. Runs the complete end-to-end pipeline in three stages:

| Stage | Description |
|-------|-------------|
| **Stage 0** | Reads the two locally-downloaded warehouse exports (product embeddings + household transaction data) via `parquet_loader.py` |
| **Stage 1** | Builds whole-basket co-purchase graphs, trains the GNN, and produces a 64-dim embedding per basket |
| **Stage 2** | Clusters basket embeddings into need-states using **both** Leiden (graph community detection) and GMM, then compares them via Adjusted Rand Index |
| **Stage 3** | Saves output parquet for manual reload into the warehouse — no live write-back |

**Key outputs written to `src/`:**
- `basket_gnn_embeddings.parquet` — 64-dim embedding per basket
- `basket_gnn_model.pt` — trained GNN weights
- `basket_need_state_clusters.parquet` — Leiden + GMM cluster labels per basket
- `copurchase_sparse.npz`, `product_id_to_index.pkl`, `product_theme.pkl`, `product_units_avg.pkl` — artifacts needed by `score_new_baskets.py`

**How to run:**
```bash
cd src
python pipeline_main.py
```

**Important:** Delete `product_subclusters.pkl` and `training_graphs.pkl` before running if the warehouse data has changed since the last run — these caches are keyed by filename, not by data fingerprint.

---

### `build_product_embeddings.py`

**What it does:** A one-time (or periodic) pre-step that builds `product_embeddings.parquet` from scratch. This file is the input product feature store for the GNN.

- Reads product attributes from `ns_item_lookup_tpna/` (TPNA grain — style level)
- Builds text strings per product: `description | department | class | subclass` (brand deliberately excluded — found to dominate similarity)
- Embeds text using a local `MiniLM-L6-v2` sentence transformer model
- Optionally applies anisotropy correction (removes top 2 principal components) to improve clustering quality
- Joins embeddings down from TPNA (style) to TPNB (SKU) grain via `ns_tpnb_to_tpna_mapping/`
- Saves to `../data/output/product_embeddings.parquet`

**How to run:**
```bash
cd src
python build_product_embeddings.py
```

**Key config (top of file):**
| Constant | Default | Description |
|----------|---------|-------------|
| `PRODUCT_ATTRIBUTES_TPNA_PARQUET` | `../data/ns_item_lookup_tpna` | Input: item lookup table |
| `TPNB_TO_TPNA_MAPPING_PARQUET` | `../data/ns_tpnb_to_tpna_mapping` | Input: SKU→style mapping |
| `OUTPUT_PARQUET` | `../data/output/product_embeddings.parquet` | Output path |
| `EMBEDDING_MODEL_NAME` | `D:\GIT\models\MiniLM-L6-v2` | Local model path |
| `APPLY_ANISOTROPY_CORRECTION` | `True` | Whether to remove top PCs |
| `N_TOP_PCS_TO_REMOVE` | `2` | How many PCs to remove |

---

### `parquet_loader.py`

**What it does:** A thin I/O utility. Reads and validates the two main input parquet files, returning DataFrames in the exact shapes expected by the rest of the pipeline.

- `load_product_embeddings(path)` — returns `DataFrame[tpnb, embedding (np.ndarray)]`
- `load_household_tpnb_period(path)` — returns `DataFrame[household_number, tpnb, year_number, period_number, quantity]`

Handles both array-typed and comma-delimited-string embedding formats (different warehouse export tools produce different shapes). Raises descriptive errors if expected columns are missing.

**Not run directly** — imported by `pipeline_main.py` and `score_new_baskets.py`.

---

### `GNN_Train.py`

**What it does:** Defines and trains the `BasketGNN` model, then uses it to embed all baskets. Called from `pipeline_main.py` via `train_and_embed()`.

**Architecture — `BasketGNN`:**
- Node encoder: `Linear(in_dim → hidden_dim=128)`
- Two `GINEConv` message-passing layers (Graph Isomorphism Network with Edge features)
- Dropout (p=0.1)
- Projection head: `Linear(128 → 64)` — the basket embedding space
- Decoder: `Linear(64 → 128 → in_dim)` — for reconstruction loss during training

**Training objective:** MSE reconstruction loss — the model learns to compress each basket graph into a 64-dim vector that can reconstruct the mean node features of that basket.

**Key config:**
| Constant | Value | Description |
|----------|-------|-------------|
| `HIDDEN_DIM` | 128 | GNN hidden layer size |
| `OUT_DIM` | 64 | Final basket embedding dimension |
| `EPOCHS` | 20 | Training epochs |
| `LR` | 1e-4 | Learning rate |
| `N_TRAIN_SAMPLES` | 1,000,000 | Max baskets sampled for training |
| `TRAIN_BATCH` | 256 | Batch size |

**Outputs:** `basket_gnn_embeddings.parquet`, `basket_gnn_model.pt`

**Not run directly** — imported by `pipeline_main.py` and `score_new_baskets.py`.

---

### `GraphBuilder.py`

**What it does:** Builds the PyTorch Geometric `Data` graph objects used to train the GNN. Contains all graph construction, feature engineering, and product sub-clustering logic. Called internally by `GNN_Train.py`.

**Key functions:**

| Function | Description |
|----------|-------------|
| `prepare_globals(...)` | Converts all product dicts to numpy arrays; runs product sub-clustering; returns a global feature dict `G` used throughout graph building |
| `build_product_subclusters(...)` | Clusters products within each theme using KMeans (K chosen by silhouette score, range 2–10). Cached to `product_subclusters.pkl` after first run |
| `sample_baskets(...)` | Stratified sample of up to 1M baskets for training, stratified by theme × basket size |
| `build_dense_cp_submatrix(...)` | Extracts the dense co-purchase submatrix for the sampled product set |
| `build_one_graph(...)` | Builds a single basket's PyG `Data` object with 6-group node features and top-K co-purchase edges |
| `build_training_graphs(...)` | Builds all training graphs; cached to `training_graphs.pkl` |
| `embed_all_baskets_fast(...)` | Volume-weighted mean pooling inference — embeds all baskets without building individual graphs |

**Node feature layout per product (in_dim = embedding_dim + 5):**

| Feature | Description |
|---------|-------------|
| `embedding[0:D]` | Product text embedding (from `build_product_embeddings.py`) |
| `cp_score` | Normalised co-purchase score with other basket items |
| `theme_score` | Frequency of this product's theme across the basket |
| `sub_cluster_id` | Normalised KMeans sub-cluster id within its product theme |
| `distinctiveness` | How exclusively the product belongs to its sub-cluster (0=generic, 1=distinct) |
| `log_units` | log1p(quantity) — purchase volume weight |

**Edge features (2-dim):**

| Feature | Description |
|---------|-------------|
| `log_copurchase` | log1p(co-purchase count between two products) |
| `same_theme` | 1 if both products share the same theme, 0 otherwise |

**Not run directly** — imported by `GNN_Train.py` and `score_new_baskets.py`.

---

### `cluster_basket_embeddings.py`

**What it does:** Stage 2 of the pipeline — takes basket GNN embeddings and clusters them into need-states. Implements two clustering methods and comparison utilities.

**Key functions:**

| Function | Description |
|----------|-------------|
| `build_basket_knn_graph(...)` | Builds a mutual k-NN similarity graph over basket embeddings (cosine distance, k=15 by default) |
| `cluster_basket_embeddings(...)` | **Leiden path** — builds kNN graph → Leiden community detection → returns `basket_id, need_state_cluster` |
| `cluster_basket_embeddings_gmm(...)` | **GMM path** — fits a Gaussian Mixture Model → returns `basket_id, need_state_cluster_gmm, gmm_confidence`. Saves fitted model to `gmm_basket_model.pkl` for later scoring |
| `assign_new_baskets_to_clusters(...)` | Assigns new (unseen) baskets to Leiden clusters via k-NN majority vote — Leiden is transductive and can't natively score new points |
| `compare_leiden_gmm(...)` | Computes Adjusted Rand Index between the two methods — diagnostic for whether they agree |
| `sweep_resolution(...)` | Sweeps Leiden resolution parameter and reports cluster count/modularity per value |
| `select_k_via_bic(...)` | BIC/AIC sweep for GMM K selection |
| `diagnose_connectivity(...)` | Reports graph connected components — run before trusting a resolution sweep |

**Key config:**
| Constant | Default | Description |
|----------|---------|-------------|
| `BASKET_KNN_K` | 15 | Neighbors in the kNN graph |
| `USE_MUTUAL_KNN` | True | Only keep edges where both baskets rank each other |
| `LEIDEN_RESOLUTION` | 1.0 | Leiden resolution — sweep with `sweep_resolution()` before trusting |
| `GMM_N_COMPONENTS` | set in `pipeline_main.py` | Number of GMM components (K) |
| `GMM_COVARIANCE` | `"diag"` | GMM covariance type |

**Not run directly** — imported by `pipeline_main.py` and `score_new_baskets.py`.

---

### `score_new_baskets.py`

**What it does:** Scores a fresh batch of baskets against the already-trained GNN and already-discovered need-states — without retraining. Produces Leiden cluster assignments (via k-NN majority vote) and GMM assignments (via the saved GMM model's `.predict()`).

**How to run:**
```bash
cd src
python score_new_baskets.py --new-transactions ../data/ns_household_tpnb_period_agg_score
```

**Prerequisites (must already exist in `src/`):**

| File | Created by |
|------|-----------|
| `basket_gnn_model.pt` | `pipeline_main.py` |
| `product_id_to_index.pkl` | `pipeline_main.py` |
| `copurchase_sparse.npz` | `pipeline_main.py` |
| `product_theme.pkl` | `pipeline_main.py` |
| `product_units_avg.pkl` | `pipeline_main.py` |
| `basket_gnn_embeddings.parquet` | `pipeline_main.py` |
| `basket_need_state_clusters.parquet` | `pipeline_main.py` |
| `gmm_basket_model.pkl` | `pipeline_main.py` (optional — GMM scoring skipped if absent) |

**Outputs:**

| File | Description |
|------|-------------|
| `new_basket_gnn_embeddings.parquet` | GNN embeddings for the new baskets |
| `new_basket_need_states.parquet` | Leiden + GMM cluster assignments for new baskets |
| `basket_gnn_embeddings_merged.parquet` | New + existing embeddings combined |
| `basket_need_state_clusters_merged.parquet` | New + existing clusters combined |

---

### `split_basket_by_theme.py`

**What it does:** A utility (no longer called in the main pipeline) that splits whole baskets into per-theme sub-baskets. Originally used to run need-state discovery per category. Kept for reference — the current pipeline finds need-states across the **whole** basket instead.

Core function: `split_baskets_by_theme(baskets_theme, product_theme, min_products=2)`

- Groups products within each basket by theme
- Assigns unmapped products to the dominant-theme sub-basket
- Drops sub-baskets with fewer than `MIN_PRODUCTS` (default 2) products
- Returns a DataFrame with additional columns: `theme_name`, `sub_basket_id`, `n_products`, `n_unmapped`

Can be run standalone:
```bash
python split_basket_by_theme.py
```
(Reads `baskets_theme.parquet`, writes `baskets_by_theme.parquet`)

---

### `requirements.txt`

Lists all Python dependencies. Install with:
```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch-geometric`, `sentence-transformers`, `leidenalg`, `igraph`, `numba`, `pynndescent`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib`, `tqdm`.

---

## Folder Layout Assumed by the Scripts

```
project/
├── data/
│   ├── ns_item_lookup_tpna/                  ← Spark-exported folder (parquet parts)
│   ├── ns_tpnb_to_tpna_mapping/              ← Spark-exported folder
│   ├── ns_household_tpnb_period_agg_train/   ← Spark-exported folder (training transactions)
│   ├── ns_household_tpnb_period_agg_score/   ← Spark-exported folder (new transactions)
│   └── output/
│       └── product_embeddings.parquet        ← Written by build_product_embeddings.py
└── src/
    ├── build_product_embeddings.py
    ├── pipeline_main.py
    ├── parquet_loader.py
    ├── GNN_Train.py
    ├── GraphBuilder.py
    ├── cluster_basket_embeddings.py
    ├── score_new_baskets.py
    ├── split_basket_by_theme.py
    └── requirements.txt
```

> All scripts assume they are run from inside `src/`, with `data/` as a sibling folder.
