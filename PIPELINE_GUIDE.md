# Need-State Pipeline — End-to-End Guide

This document explains what the code in `improved-system/` does and the exact
sequence of steps to run to go from raw warehouse transaction data to
**need-state clusters** (data-driven groupings of shopping baskets that
represent "why the customer shopped," independent of product category).

The pipeline is **theme-free**: no category/theme concept is used anywhere
to pre-group products or baskets. Need-states emerge purely from a graph
neural network (GNN) trained on co-purchase behavior, followed by clustering
of the resulting basket embeddings.

⚠️ **Basket grain is WEEK, not a true single-visit basket.** None of the
tables in this warehouse (see `data/TABLE_REFERENCE.md`) carry a
transaction/order/checkout identifier — `cltv_hh_metrics_tpnb_base` (the
household-purchase source table) is already pre-aggregated to week grain,
and its `orders` column is a count, not a preserved per-visit ID. So a
"basket" here means *everything one household bought in one week*, not one
shopping trip. Week is the finest grain available; see the comment at the
top of `data/ns_household_tpnb_week_agg_train.sql` for the full reasoning.

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
                                     │     data/output/product_embeddings.parquet
                                     │
 ns_household_tpnb_week_agg_train.sql ──►  pipeline_main.py  (Stage 0)
   (household x product x WEEK)             │
                                              ▼
                                   Stage 1: build whole baskets (week grain),
                                   co-purchase matrix, train GNN
                                   (GraphBuilder.py + GNN_Train.py)
                                              │
                                              ▼
                                   Stage 2: cluster basket embeddings
                                   — Leiden AND GMM
                                   (cluster_basket_embeddings.py)
                                              │
                                              ▼
                        data/output/basket_need_state_clusters.parquet
                                              │
                                              ▼
                                   Stage 3: manually re-upload into
                                   the warehouse (need_state_cluster,
                                   need_state_cluster_gmm per basket)

 ns_household_tpnb_week_agg_score.sql ──►  score_new_baskets.py
   (later/held-out weeks)                  (scores NEW baskets against
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
│   ├── ns_household_tpnb_week_agg_train/        (folder)
│   ├── ns_household_tpnb_week_agg_score/        (folder — only needed for scoring new baskets)
│   ├── *.sql                                    (run manually against the warehouse; see §3)
│   ├── TABLE_REFERENCE.md                       (what each source table actually contains)
│   └── output/                                  (ALL pipeline-written artifacts — nothing is written
│                                                  to the working directory. product_embeddings.parquet,
│                                                  caches, the trained model, embeddings, cluster output)
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

---

## 3. Step 0 — Run the SQL against your warehouse

These live in `improved-system/data/*.sql` and must be run manually (no live
DB connection exists anywhere in the Python code — everything is
file-based). Run each, then download the resulting table as parquet into the
matching `data/` folder above. See `data/TABLE_REFERENCE.md` for what the
underlying source tables (`product.product`, `cltv_hh_metrics_tpnb_base`,
etc.) actually contain.

| File | Produces (grain) | Feeds |
|---|----|----|
| `ns_item_lookup_tpna.sql` | one row per TPNA style (description, brand, hierarchy) — scoped to TPNAs actually purchased in the training window | `build_product_embeddings.py` |
| `ns_tpnb_to_tpna_mapping.sql` | one row per TPNB → TPNA, full/unfiltered mapping | `build_product_embeddings.py` |
| `ns_product_theme_mapping.sql` | TPNB → category (`theme_name`) | **Not consumed anywhere in the current pipeline** — kept only for optional post-hoc profiling of clusters (see §7). Safe to skip for the core run. |
| `ns_household_tpnb_week_agg_train.sql` | household × TPNB × year_week, summed quantity/orders/sales | `pipeline_main.py` (training baskets) |
| `ns_household_tpnb_week_agg_score.sql` | same grain, but later/held-out weeks | `score_new_baskets.py` (scoring new baskets) — only needed once you want to score new data |

**Important — keep period ranges consistent:**
- `ns_item_lookup_tpna.sql`'s `<START_YEAR_PERIOD>`/`<END_YEAR_PERIOD>` placeholders **must match** `ns_household_tpnb_week_agg_train.sql`'s training period range. If they don't, some products in your training baskets will have no embedding, and `GraphBuilder.py` silently falls back to a zero vector for them rather than erroring.
- `ns_household_tpnb_week_agg_score.sql` should use a period range **after** the training one, since it's meant to be data the model never trained on.
- If you haven't confirmed the real period values yet, query `MIN/MAX(year_number*100+period_number)` on `cltv_hh_metrics_tpnb_base` first.

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
6. Writes `data/output/product_embeddings.parquet` (columns: `tpnb`, `embedding`).

Requires `pip install sentence-transformers` (see `requirements.txt` note about installing `torch`/`torch-geometric` first, matching your platform/CUDA version).

---

## 5. Step 2 — Run the main pipeline

```
python pipeline_main.py
```

### Cache-staleness warning (read this first)

Several files under `data/output/` are derived from **basket composition**
(which products appear together in which basket) — if you change the input
data or the basket grain (e.g. the period→week change described above),
every one of these is stale and must be deleted before your next run, or
they'll silently produce results that don't mean what you think they mean:

```
data/output/training_graphs.pkl
data/output/basket_gnn_model.pt
data/output/copurchase_sparse.npz
data/output/product_id_to_index.pkl
data/output/product_units_avg.pkl
data/output/basket_gnn_embeddings.parquet
data/output/gmm_basket_model.pkl
data/output/basket_need_state_clusters.parquet
```

`data/output/product_subclusters.pkl` and `data/output/product_embeddings.parquet`
do **not** need deleting for a basket-grain or transaction-data change — both
are derived purely from product text attributes, never from basket/purchase
data. They only go stale if the *product catalog or embedding logic* changes.

The co-purchase-matrix build's own checkpoint
(`copurchase_sparse.checkpoint.npz` + `.progress.txt`) is self-protecting:
it stores a fingerprint of the basket/product counts it was built from, and
automatically rebuilds from scratch if a stale checkpoint doesn't match the
current run, rather than silently resuming into mismatched data. The list
above is not fingerprinted, though — those must be deleted manually.

### Stage 0 — Read warehouse exports (`parquet_loader.py`)

Loads:
- `data/output/product_embeddings.parquet` → `product_df_2` (tpnb, embedding) — one shot, small file
- `ns_household_tpnb_week_agg_train` → **streamed**, not loaded as one table (see Stage 1)

### Stage 1 — Build baskets, train the GNN

1. **Whole-basket construction, streamed**: `parquet_loader.stream_build_baskets_and_units_avg()`
   reads the household×tpnb×week export in bounded batches
   (`batch_size`, default 20,000,000 rows) via `pyarrow.dataset`, and builds
   `baskets` and `product_units_avg` directly from the stream — the raw flat
   table is never materialized as one object. This matters because at week
   grain this export routinely reaches billions of rows (no longer
   collapsed across weeks the way the old period grain was), large enough
   that a single `pd.read_parquet()` can fail outright
   (`pyarrow.lib.ArrowMemoryError`) even on a machine with plenty of total
   RAM free — pyarrow needs one big contiguous allocation to convert the
   whole table to pandas in one go. Every product a household bought in a
   given **week** becomes one basket (`household_number` + `year_week_number`
   → `basket_id`) — see the grain warning at the top of this doc. No
   category/theme split. Baskets with fewer than `MIN_BASKET_PRODUCTS = 2`
   items are dropped.
   - ⚠️ Watch the `BASKET_COUNT_WARN_THRESHOLD` (2M) check — if you exceed
     it, the full basket table can run into 50–100GB in memory. If that's
     not intentional, verify the household-sampling filter you expect
     (e.g. `MOD(household_number, N) = 0`) is actually applied at the SQL
     stage.

2. **Co-purchase matrix**: built in row-chunks of `COPURCHASE_CHUNK_BASKETS`
   baskets at a time (default 500,000), checkpointed to disk after every
   chunk so a crash resumes instead of restarting — mathematically identical
   to computing `X.T @ X` over the whole population in one shot, not an
   approximation. Produces `copurchase_sparse` (product×product co-purchase
   counts), `product_id_to_index`, and `product_units_avg` (mean quantity per
   product) — saved to `data/output/` immediately since `score_new_baskets.py`
   needs them later.

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
     `GNN_Train.py`).
   - `build_training_graphs()` — builds one PyTorch Geometric graph per
     sampled basket via `build_one_graph()`: nodes = products in the basket
     (with the 4 extra features above), edges = each product's top-`TOP_K=10`
     co-purchase partners **within that basket**, with edge features
     `[log(co-purchase count), relative strength vs. that node's own
     strongest link]`. Each basket's co-purchase values come from a small
     **per-basket** dense submatrix sliced from the sparse global co-purchase
     matrix on the fly (bounded by that basket's own product count squared —
     independent of catalog size, no size threshold anywhere). Cached to
     `data/output/training_graphs.pkl`.
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
     `encode()` path. Processed in chunks (`graph_chunk_size`, default
     50,000 baskets) so the whole population's graphs are never all held in
     memory at once — only the final embeddings accumulate. This guarantees
     training and scoring can never compute features differently.
   - Saves `data/output/basket_gnn_embeddings.parquet` (`basket_id`,
     `gnn_embedding`) and `data/output/basket_gnn_model.pt`.

### Stage 2 — Cluster basket embeddings into need-states (`cluster_basket_embeddings.py`)

Both methods are run and compared/combined — this pipeline doesn't pick one
upfront:

- **2a — Leiden** (`cluster_basket_embeddings`): builds a mutual-kNN graph
  over the L2-normalized basket embeddings (`BASKET_KNN_K = 15`, cosine
  similarity via `pynndescent` if installed, else sklearn
  `NearestNeighbors`; edge construction is fully vectorized with NumPy, not a
  per-pair Python loop), then runs Leiden community detection
  (`LEIDEN_RESOLUTION = 1.0` — described as a starting point; use
  `sweep_resolution()` to check other values before trusting this one).
  Output column: `need_state_cluster`.
- **2b — GMM** (`cluster_basket_embeddings_gmm`): fits a
  `GaussianMixture(n_components=GMM_N_COMPONENTS, covariance_type="diag")`
  directly on the normalized embeddings. `GMM_N_COMPONENTS = 30` in
  `pipeline_main.py` is explicitly called out as a **placeholder** — swap it
  for a real best-K selection once available (`select_k_via_bic()` gives a
  BIC/AIC sweep to eyeball a better value). Saves the fitted model to
  `data/output/gmm_basket_model.pkl` (needed later to score new baskets
  directly, since GMM natively supports `.predict()` on new points — Leiden
  does not). Output columns: `need_state_cluster_gmm`, `gmm_confidence`.
- **2c — Compare** (`compare_leiden_gmm`): Adjusted Rand Index between the
  two label sets — close to 1 means they agree, close to 0 means they're
  finding different structure. Purely diagnostic, printed to console.

Both label sets are merged (outer join on `basket_id`) into one file —
neither is discarded, since which one (or how to combine them) isn't decided
yet.

### Stage 3 — Reload into the warehouse (manual)

Output: `data/output/basket_need_state_clusters.parquet`
(`basket_id`, `need_state_cluster`, `need_state_cluster_gmm`). No live
write-back connection exists — upload this parquet (or re-save as CSV first
if your workspace tool doesn't support parquet import) through your
warehouse's manual import feature.

---

## 6. Scoring new baskets later (no retraining)

Once you have held-out weeks of new data (`ns_household_tpnb_week_agg_score.sql`
→ downloaded parquet), score it against the already-trained model:

```
python score_new_baskets.py --new-transactions ../data/ns_household_tpnb_week_agg_score/...
```

(Path arg should point at wherever you saved that download.)

Requires these artifacts to already exist under `data/output/` from a prior
`pipeline_main.py` run: `basket_gnn_model.pt`, `product_id_to_index.pkl`,
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
2. Builds new whole-baskets (week grain) from the new transactions, using
   the exact same logic as `pipeline_main.py` Stage 1.
3. Filters out any basket_id that's already been embedded before (dedup
   against `basket_gnn_embeddings.parquet`).
4. Embeds the new baskets via `embed_all_baskets_fast()` — same function,
   same code path, same chunking as training.
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
   `basket_need_state_clusters_merged.parquet` (all under `data/output/`) for
   re-upload.

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

# 3. Delete stale caches under data/output/ if this is a first run after any
#    data or basket-grain change: training_graphs.pkl, basket_gnn_model.pt,
#    copurchase_sparse.npz, product_id_to_index.pkl, product_units_avg.pkl,
#    basket_gnn_embeddings.parquet, gmm_basket_model.pkl,
#    basket_need_state_clusters.parquet
#    (product_subclusters.pkl / product_embeddings.parquet do NOT need
#    deleting — they're product-text-derived, not basket-derived)

# 4. Train the GNN + cluster into need-states
python pipeline_main.py
#    -> data/output/basket_need_state_clusters.parquet  (upload this to the warehouse)

# 5. (Later, periodically) score new/held-out weeks without retraining
python score_new_baskets.py --new-transactions <path to new weeks' parquet>
```

---

## 10. Key knobs you'll likely want to revisit

| Setting | Where | Current value | Note |
|---|---|---|---|
| `MIN_BASKET_PRODUCTS` | `pipeline_main.py` / `score_new_baskets.py` | 2 | Drops 1-item baskets |
| `N_TRAIN_SAMPLES` | `GNN_Train.py` | 300,000 | Reduced from 1,000,000 for memory safety; raise once you've confirmed headroom |
| `COPURCHASE_CHUNK_BASKETS` | `pipeline_main.py` | 500,000 | Baskets per co-purchase-matrix chunk; lower if still memory-constrained |
| `batch_size` (of `stream_build_baskets_and_units_avg`) | `parquet_loader.py` | 20,000,000 | Rows per streamed read batch from the household×tpnb×week export; lower if still memory-constrained at Stage 0/1 |
| `SUBCL_K_CANDIDATES` | `GraphBuilder.py` | [50, 100, 200, 400] | Global product sub-cluster K candidates — tune to catalog size |
| `TOP_K` | `GraphBuilder.py` | 10 | Co-purchase edges kept per node per basket graph |
| `EPOCHS` / `LR` | `GNN_Train.py` | 20 / 1e-4 | GNN training |
| `LEIDEN_RESOLUTION` | `cluster_basket_embeddings.py` | 1.0 | Run `sweep_resolution()` first rather than trusting this |
| `GMM_N_COMPONENTS` | `pipeline_main.py` | 30 | Explicit placeholder — replace with real best-K logic, or eyeball `select_k_via_bic()` |
| Training/scoring period ranges | `ns_household_tpnb_week_agg_train.sql` / `..._score.sql` | `202603–202604` | Must match `ns_item_lookup_tpna.sql`'s range for training; scoring range should be later |

See `data/TABLE_REFERENCE.md` for what each underlying warehouse table
actually contains, and why basket grain landed on WEEK rather than a true
single-visit basket.
