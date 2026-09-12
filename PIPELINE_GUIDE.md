# Need-State Pipeline — End-to-End Guide

This document explains what the code in `improved-system/` does and the exact
sequence of steps to run to go from raw warehouse transaction data to
**need-state clusters** (data-driven groupings of shopping baskets that
represent "why the customer shopped," independent of product category).

The pipeline is **theme-free**: no category/theme concept is used anywhere
to pre-group products or baskets. Need-states emerge purely from a graph
neural network (GNN) trained on co-purchase behavior, followed by clustering
of the resulting basket embeddings.

---

## 1. Big-picture flow

```
 WAREHOUSE (SQL, run manually)                LOCAL PYTHON PIPELINE
 ──────────────────────────────                ──────────────────────

 ns_item_lookup_tpna.sql          ──┐
 ns_tpnb_to_tpna_mapping.sql      ──┼──►  build_product_embeddings.py
                                     │        (text embeds each product
                                     │         style, TPNA grain)
                                     │              │
                                     │              ▼
                                     │     data/product_embeddings.parquet
                                     │
 ns_household_tpnb_period_agg_train.sql ──►  pipeline_main.py  (Stage 0)
   (household x product x period)            │
                                              ▼
                                   Stage 1: build whole baskets,
                                   co-purchase matrix, train GNN
                                   (GraphBuilder.py + GNN_Train.py)
                                              │
                                              ▼
                                   Stage 2: cluster basket embeddings
                                   — Leiden AND GMM
                                   (cluster_basket_embeddings.py)
                                              │
                                              ▼
                              basket_need_state_clusters.parquet
                                              │
                                              ▼
                                   Stage 3: manually re-upload into
                                   the warehouse (need_state_cluster,
                                   need_state_cluster_gmm per basket)

 ns_household_tpnb_period_agg_score.sql ──►  score_new_baskets.py
   (later/held-out period)                  (scores NEW baskets against
                                              the already-trained model —
                                              no retraining)
```

Need-states are **discovered after the fact** by profiling what's dominant
in each cluster — themes/categories are never fed in as an input anywhere.

---

## 2. Folder layout the code expects

```
improved-system/
├── data/
│   ├── ns_item_lookup_tpna/                    (Spark/Databricks export — a folder of part-files)
│   ├── ns_tpnb_to_tpna_mapping/                 (folder)
│   ├── ns_household_tpnb_period_agg_train/      (folder)
│   ├── ns_household_tpnb_period_agg_score/      (folder — only needed for scoring new baskets)
│   └── product_embeddings.parquet               (single file — written by build_product_embeddings.py)
└── src/
    ├── build_product_embeddings.py
    ├── pipeline_main.py
    ├── GraphBuilder.py
    ├── GNN_Train.py
    ├── cluster_basket_embeddings.py
    ├── parquet_loader.py
    ├── score_new_baskets.py
    └── test_theme_free_pipeline.py
```

⚠️ **Path mismatch to fix before running**: `build_product_embeddings.py`
writes `../data/product_embeddings.parquet`, and `score_new_baskets.py` reads
from that same path — but `pipeline_main.py` currently points at
`../data/output/product_embeddings.parquet` (line 126). Either move the file
into a `data/output/` subfolder after building it, or edit that constant in
`pipeline_main.py` so it matches where `build_product_embeddings.py` actually
wrote the file. Otherwise Stage 0 of `pipeline_main.py` will raise
`FileNotFoundError`.

---

## 3. Step 0 — Run the SQL against your warehouse

These live in `improved-system/data/*.sql` and must be run manually (no live
DB connection exists anywhere in the Python code — everything is
file-based). Run each, then download the resulting table as parquet into the
matching `data/` folder above.

| # | File | Produces (grain) | Feeds |
|---|------|----|----|
| 1 | `ns_item_lookup_tpna.sql` | one row per TPNA style (description, brand, hierarchy) — scoped to TPNAs actually purchased in the training period | `build_product_embeddings.py` |
| 2 | `ns_tpnb_to_tpna_mapping.sql` | one row per TPNB → TPNA, full/unfiltered mapping | `build_product_embeddings.py` |
| 3 | `ns_product_theme_mapping.sql` | TPNB → category (`theme_name`) | **Not consumed anywhere in the current pipeline** — kept only for optional post-hoc profiling of clusters (see §7). Safe to skip for the core run. |
| 4 | `ns_household_tpnb_period_agg_train.sql` | household × TPNB × year_period, summed quantity/orders/sales | `pipeline_main.py` (training baskets) |
| 5 | `ns_household_tpnb_period_agg_score.sql` | same grain as #4, but a **later/held-out period** | `score_new_baskets.py` (scoring new baskets) — only needed once you want to score new data |

**Important — keep period ranges consistent:**
- File #1's `<START_YEAR_PERIOD>`/`<END_YEAR_PERIOD>` placeholders **must match** file #4's training period range. If they don't, some products in your training baskets will have no embedding, and `GraphBuilder.py` silently falls back to a zero vector for them rather than erroring.
- File #5 should use a period range **after** file #4's, since it's meant to be data the model never trained on.
- If you haven't confirmed the real period values yet, there's a mention of a `diagnostic_check_periods.sql` check to run first (referenced in file #1's comments, not included in this folder — write one if needed, or just query `MIN/MAX(year_number*100+period_number)` on the source table).

There is no `06_write_back_need_states.sql` file currently in this folder —
`pipeline_main.py`'s Stage 3 comment references it, but you'll need to write
that landing-table DDL yourself (or just upload the parquet/CSV through your
workspace's import tool without a pre-created table, if that's supported).

---

## 4. Step 1 — Build product embeddings

```
cd src
python build_product_embeddings.py
```

What it does (`build_product_embeddings.py`):
1. Loads `ns_item_lookup_tpna` (TPNA attributes) and drops duplicate TPNAs.
2. Builds a text string per TPNA: `description | department | class | subclass` — **brand is deliberately excluded** (found to dominate similarity for short/generic descriptions, causing clustering by brand rather than by what the product actually is).
3. Embeds that text with `sentence-transformers` (`all-MiniLM-L6-v2` by default).
4. Applies an "all-but-the-top" anisotropy correction (removes the top 2 principal components after mean-centering) — a carried-over default, not re-validated for this specific product set; worth checking downstream clustering quality with/without it.
5. Loads the TPNB→TPNA mapping and **broadcasts** each TPNA's embedding down to every sibling TPNB (every size/color variant of one style gets an identical vector).
6. Writes `data/product_embeddings.parquet` (columns: `tpnb`, `embedding`).

Requires `pip install sentence-transformers` (see `requirements.txt` note about installing `torch`/`torch-geometric` first, matching your platform/CUDA version).

---

## 5. Step 2 — Run the main pipeline

```
python pipeline_main.py
```

### Cache-staleness warning (read this first)

`GraphBuilder.py` caches two intermediates to **fixed filenames**
(`product_subclusters.pkl`, `training_graphs.pkl`), reused as-is on any
future run regardless of whether the underlying data or code changed. If any
of these exist from a previous run (especially a pre-refactor run), **delete
them before your first run**, or they will silently produce embeddings that
don't mean what you think they mean:

```
product_subclusters.pkl
training_graphs.pkl
basket_gnn_model.pt
copurchase_sparse.npz
product_id_to_index.pkl
product_units_avg.pkl
```

### Stage 0 — Read warehouse exports (`parquet_loader.py`)

Loads:
- `product_embeddings.parquet` → `product_df_2` (tpnb, embedding)
- `ns_household_tpnb_period_agg_train` → `tpnb_x_hh` (household_number, tpnb, year_number, period_number, quantity)

### Stage 1 — Build baskets, train the GNN

1. **Whole-basket construction**: every product a household bought in a
   given year-period becomes one basket (`household_number` + `year_period_number`
   → `basket_id`). No category/theme split. Baskets with fewer than
   `MIN_BASKET_PRODUCTS = 2` items are dropped.
   - ⚠️ Watch the `BASKET_COUNT_WARN_THRESHOLD` (2M) check — if you exceed
     it, the full basket table can run into 50–100GB in memory. If that's
     not intentional, verify the household-sampling filter you expect
     (e.g. `MOD(household_number, N) = 0`) is actually applied at the SQL
     stage.

2. **Co-purchase matrix**: builds a sparse product×basket incidence matrix,
   then `X.T @ X` → a product×product co-purchase count matrix
   (`copurchase_sparse`), plus a `product_id_to_index` map and
   `product_units_avg` (mean quantity per product). These three are saved to
   disk immediately (`copurchase_sparse.npz`, `product_id_to_index.pkl`,
   `product_units_avg.pkl`) — they're what `score_new_baskets.py` needs later.

3. **Train + embed** (`GNN_Train.train_and_embed`, delegating heavily to
   `GraphBuilder.py`):
   - `prepare_globals()` — builds the node feature arrays: product embedding
     matrix, plus one global **product sub-clustering** pass
     (`MiniBatchKMeans` over the *entire catalog*, k chosen by silhouette
     score from `SUBCL_K_CANDIDATES = [50, 100, 200, 400]`), a
     "distinctiveness" score (how close each product sits to its assigned
     centroid vs. the farthest centroid), and per-product average units.
     Final node feature width: `in_dim = emb_dim + 4`
     (`[embedding | co-purchase score | sub_cluster_id | distinctiveness | log_units]`).
   - `sample_baskets()` — stratified sampling **by basket size only** (bins:
     1, 2-5, 6-15, 16-50, 50+), capped at `N_TRAIN_SAMPLES = 300,000` (set in
     `GNN_Train.py`; `GraphBuilder.py`'s own default of 1,000,000 is
     overridden by this).
   - `build_dense_cp_submatrix()` — a dense co-purchase submatrix restricted
     to only the products appearing in the sampled training baskets (memory
     safety check: falls back to sparse if the dense version would exceed
     ~200GB).
   - `build_training_graphs()` — builds one PyTorch Geometric graph per
     sampled basket via `build_one_graph()`: nodes = products in the basket
     (with the 4 extra features above), edges = each product's top-`TOP_K=10`
     co-purchase partners **within that basket**, with edge features
     `[log(co-purchase count), relative strength vs. that node's own
     strongest link]`. Cached to `training_graphs.pkl`.
   - **Model** (`BasketGNN` in `GNN_Train.py`): a graph autoencoder —
     `node_encoder (Linear) → GINEConv → GINEConv → global_mean_pool → proj`
     produces the basket embedding (`out_dim = 64`); a `decoder` reconstructs
     the mean node feature vector, and the training loss is MSE between
     reconstruction and the actual per-basket mean node features. Trained for
     `EPOCHS = 20` with Adam (`lr=1e-4`), gradient clipping, and AMP
     (autocast/GradScaler) if CUDA is available. NaN-producing batches are
     detected and skipped rather than corrupting the model.
   - **Inductive inference** (`embed_all_baskets_fast()` in
     `GraphBuilder.py`) — after training, builds a **real graph for every
     basket** (not just the sampled training subset) using the exact same
     `build_one_graph()` function, and runs it through the model's full
     `encode()` path. This guarantees training and scoring can never compute
     features differently.
   - Saves `basket_gnn_embeddings.parquet` (`basket_id`, `gnn_embedding`) and
     `basket_gnn_model.pt`.

### Stage 2 — Cluster basket embeddings into need-states (`cluster_basket_embeddings.py`)

Both methods are run and compared/combined — this pipeline doesn't pick one
upfront:

- **2a — Leiden** (`cluster_basket_embeddings`): builds a mutual-kNN graph
  over the L2-normalized basket embeddings (`BASKET_KNN_K = 15`, cosine
  similarity via `pynndescent` if installed, else sklearn
  `NearestNeighbors`), then runs Leiden community detection
  (`LEIDEN_RESOLUTION = 1.0` — described as a starting point; use
  `sweep_resolution()` to check other values before trusting this one).
  Output column: `need_state_cluster`.
- **2b — GMM** (`cluster_basket_embeddings_gmm`): fits a
  `GaussianMixture(n_components=GMM_N_COMPONENTS, covariance_type="diag")`
  directly on the normalized embeddings. `GMM_N_COMPONENTS = 30` in
  `pipeline_main.py` is explicitly called out as a **placeholder** — swap it
  for a real best-K selection once available (`select_k_via_bic()` gives a
  BIC/AIC sweep to eyeball a better value). Saves the fitted model to
  `gmm_basket_model.pkl` (needed later to score new baskets directly, since
  GMM natively supports `.predict()` on new points — Leiden does not).
  Output columns: `need_state_cluster_gmm`, `gmm_confidence`.
- **2c — Compare** (`compare_leiden_gmm`): Adjusted Rand Index between the
  two label sets — close to 1 means they agree, close to 0 means they're
  finding different structure. Purely diagnostic, printed to console.

Both label sets are merged (outer join on `basket_id`) into one file —
neither is discarded, since which one (or how to combine them) isn't decided
yet.

### Stage 3 — Reload into the warehouse (manual)

Output: `basket_need_state_clusters.parquet`
(`basket_id`, `need_state_cluster`, `need_state_cluster_gmm`). No live
write-back connection exists — upload this parquet (or re-save as CSV first
if your workspace tool doesn't support parquet import) through your
warehouse's manual import feature.

---

## 6. Scoring new baskets later (no retraining)

Once you have a period of new/held-out data (`ns_household_tpnb_period_agg_score.sql`
→ downloaded parquet), score it against the already-trained model:

```
python score_new_baskets.py --new-transactions ../data/ns_household_tpnb_period_agg_score/... 
```

(Path arg should point at wherever you saved that download — the docstring
example uses `data/new_basket_source.parquet`, matching the naming in
`05_new_basket_source.sql`'s header comment even though the actual file in
this repo is `ns_household_tpnb_period_agg_score.sql`.)

Requires these artifacts to already exist from a prior `pipeline_main.py`
run: `basket_gnn_model.pt`, `product_id_to_index.pkl`,
`copurchase_sparse.npz`, `product_units_avg.pkl`,
`basket_gnn_embeddings.parquet`, `basket_need_state_clusters.parquet`, and
optionally `gmm_basket_model.pkl` (GMM assignment is skipped, not an error,
if that file is missing).

What it does:
1. Reloads the trained model + globals (`prepare_globals()`), inferring the
   architecture's dimensions from the checkpoint itself rather than trusting
   current config — and **warns loudly** if the checkpoint's `in_dim` doesn't
   match what `prepare_globals()` computes now (a strong signal of a
   pre-refactor checkpoint, i.e. retrain rather than continue).
2. Builds new whole-baskets from the new transactions, using the exact same
   logic as `pipeline_main.py` Stage 1.
3. Filters out any basket_id that's already been embedded before (dedup
   against `basket_gnn_embeddings.parquet`).
4. Embeds the new baskets via `embed_all_baskets_fast()` — same function,
   same code path as training.
5. Assigns each new basket to a need-state via **both** methods:
   - Leiden: `assign_new_baskets_to_clusters()` — Leiden has no native way to
     place a new point into an existing community, so this does a k-NN
     (k=15) majority vote among already-clustered reference baskets in
     embedding space, plus a `cluster_confidence` (fraction of neighbors
     that agreed).
   - GMM: the saved `gmm_basket_model.pkl`'s own `.predict()` /
     `.predict_proba()` — a fitted GMM scores new points directly, no
     workaround needed.
6. Saves `new_basket_gnn_embeddings.parquet`, `new_basket_need_states.parquet`,
   and merges everything into `basket_gnn_embeddings_merged.parquet` /
   `basket_need_state_clusters_merged.parquet` for re-upload.

---

## 7. After clustering: naming/profiling need-states (not yet implemented)

`pipeline_main.py`'s closing comment is explicit that **themes, if wanted at
all, are formed after clustering** — by profiling each `need_state_cluster`'s
dominant products/category hierarchy — and never fed back in as a pipeline
input. That profiling step doesn't exist in this repo yet. If/when you build
it, `ns_product_theme_mapping.sql`'s TPNB→category mapping (currently unused
by any Python file) is exactly what you'd join against each cluster's basket
contents to describe what each need-state actually represents in
human-readable terms.

---

## 8. Sanity-check the pipeline itself

```
python test_theme_free_pipeline.py
```

Two checks, both must pass (exit code 0):
1. **Static** — walks the AST of every active pipeline file and fails if any
   function/parameter/variable name contains "theme" (guards against
   reintroducing category-based logic), and fails if the old
   `split_basket_by_theme.py` file has reappeared.
2. **Functional** — builds a tiny synthetic catalog/baskets and actually
   runs `prepare_globals()` → `build_training_graphs()` →
   `embed_all_baskets_fast()`, asserting both paths produce node feature
   tensors of identical width and that both exercise the full GNN encoder
   (not a shortcut). Good to run once after any environment setup, before
   trusting a real run.

---

## 9. Quick-reference: run order

```bash
cd src

# 0. Run all SQL files in data/ against your warehouse first, download
#    each result as parquet into the matching data/ subfolder.

# 1. Sanity check the code (optional but recommended first time)
python test_theme_free_pipeline.py

# 2. Build product embeddings (only needs to be re-run if the product
#    catalog / attributes change)
python build_product_embeddings.py

# 3. Delete stale caches if this is a first run after any refactor:
#    product_subclusters.pkl, training_graphs.pkl, basket_gnn_model.pt,
#    copurchase_sparse.npz, product_id_to_index.pkl, product_units_avg.pkl

# 4. Train the GNN + cluster into need-states
python pipeline_main.py
#    -> basket_need_state_clusters.parquet  (upload this to the warehouse)

# 5. (Later, periodically) score new/held-out periods without retraining
python score_new_baskets.py --new-transactions <path to new period's parquet>
```

---

## 10. Key knobs you'll likely want to revisit

| Setting | Where | Current value | Note |
|---|---|---|---|
| `MIN_BASKET_PRODUCTS` | `pipeline_main.py` / `score_new_baskets.py` | 2 | Drops 1-item baskets |
| `N_TRAIN_SAMPLES` | `GNN_Train.py` | 300,000 | Reduced from 1,000,000 for memory safety; raise once you've confirmed headroom |
| `SUBCL_K_CANDIDATES` | `GraphBuilder.py` | [50, 100, 200, 400] | Global product sub-cluster K candidates — tune to catalog size |
| `TOP_K` | `GraphBuilder.py` | 10 | Co-purchase edges kept per node per basket graph |
| `EPOCHS` / `LR` | `GNN_Train.py` | 20 / 1e-4 | GNN training |
| `LEIDEN_RESOLUTION` | `cluster_basket_embeddings.py` | 1.0 | Run `sweep_resolution()` first rather than trusting this |
| `GMM_N_COMPONENTS` | `pipeline_main.py` | 30 | Explicit placeholder — replace with real best-K logic, or eyeball `select_k_via_bic()` |
| Training/scoring period ranges | `ns_household_tpnb_period_agg_train.sql` / `..._score.sql` | `202603–202604` | Must match `ns_item_lookup_tpna.sql`'s range for training; scoring range should be later |
