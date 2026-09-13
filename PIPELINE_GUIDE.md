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
                                   local self-contained Postgres (pg_manager.py)
                                   — baskets_train table (basket_store.py)
                                              │
                                              ▼
                                   Stage 1: co-purchase matrix (streamed from
                                   Postgres), train GNN (LMDB-cached training
                                   graphs), embed every basket (restartable,
                                   chunked inference)
                                   (GraphBuilder.py + GNN_Train.py + lmdb_graph_cache.py)
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
│   ├── pgdata/                                  (the local self-contained Postgres instance's
│   │                                              data directory — pg_manager.py manages this)
│   └── output/                                  (ALL pipeline-written artifacts — nothing is written
│                                                  to the working directory. product_embeddings.parquet,
│                                                  caches, the trained model, the LMDB training-graph
│                                                  cache, embeddings, cluster output)
└── src/
    ├── build_product_embeddings.py
    ├── pipeline_main.py
    ├── pg_manager.py
    ├── basket_store.py
    ├── lmdb_graph_cache.py
    ├── GraphBuilder.py
    ├── GNN_Train.py
    ├── cluster_basket_embeddings.py
    ├── parquet_loader.py
    ├── score_new_baskets.py
    └── test_theme_free_pipeline.py
```

---

## 3. Step 0 — Run the SQL against your warehouse

These live in `improved-system/data/*.sql` and must be run manually against
your company's WAREHOUSE (no live connection to that warehouse exists
anywhere in the Python code — everything is file-based: run the SQL by
hand, download the result as parquet). Run each, then download the
resulting table as parquet into the matching `data/` folder above. See
`data/TABLE_REFERENCE.md` for what the underlying source tables
(`product.product`, `cltv_hh_metrics_tpnb_base`, etc.) actually contain.

(Separately, the Python pipeline *does* talk to a database — a private,
local, self-contained Postgres instance under `data/pgdata/`, managed
automatically by `pg_manager.py`. That's an implementation detail of this
pipeline's own basket storage, not a connection to your warehouse.)

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
data/output/basket_gnn_model.pt
data/output/copurchase_sparse.npz
data/output/product_id_to_index.pkl
data/output/product_units_avg.pkl
data/output/basket_gnn_embeddings.parquet
data/output/gmm_basket_model.pkl
data/output/basket_need_state_clusters.parquet
data/output/embeddings_chunk_*.parquet
data/output/training_graphs.lmdb/   (and its .manifest.json)
```

`data/output/product_subclusters.pkl` and `data/output/product_embeddings.parquet`
do **not** need deleting for a basket-grain or transaction-data change — both
are derived purely from product text attributes, never from basket/purchase
data. They only go stale if the *product catalog or embedding logic* changes.

The co-purchase-matrix build's own checkpoint
(`copurchase_sparse.checkpoint.npz` + `.progress.txt`) and the LMDB
training-graph cache (`training_graphs.lmdb/` + its `.manifest.json`) are
both self-protecting: they fingerprint what they were built from and
rebuild automatically on mismatch rather than silently resuming into
mismatched data. The list above is not fingerprinted, though — those must
be deleted manually. Basket storage itself now lives in a local, self-
contained Postgres instance (`data/pgdata/`, see below) rather than a
Python object or a plain file — if you want a fully clean rebuild for a
given `dataset_tag`, `basket_store.drop_all(conn_uri, dataset_tag)` drops
its Postgres tables too.

### Stage 0 — Read warehouse exports, load baskets into Postgres

- `parquet_loader.load_product_embeddings()` → `product_df_2` (tpnb,
  embedding) — one shot, small file, unchanged from before.
- `pg_manager.get_connection_uri()` starts (or reuses) a **self-contained
  local Postgres instance** — via `pgserver`, a pip-installed dependency
  that bundles real Postgres binaries, so there's no manual install, no
  `pg_ctl`, no port/config wrangling. Its data directory lives at
  `data/pgdata/`.
- `basket_store.load_raw_export_to_postgres()` streams the household×tpnb×week
  export in bounded batches (`batch_size`, default 20,000,000 rows) via
  `pyarrow.dataset`, piping each batch straight into a Postgres staging
  table via `COPY` — the raw flat table is never materialized as one Python
  object, not even chunked-and-discarded, just never assembled client-side
  at all. This is what replaces the old `pyarrow.lib.ArrowMemoryError` risk:
  at week grain this export routinely reaches billions of rows, large
  enough that a single `pd.read_parquet()` could fail outright even with
  plenty of total RAM free.

### Stage 1 — Build baskets in Postgres, train the GNN, embed every basket

1. **Basket aggregation, in Postgres**: `basket_store.build_baskets_table()`
   runs one SQL pass — `GROUP BY household_number, year_week_number` with
   `array_agg(tpnb)` / `array_agg(quantity)` — that replaces the old Python
   dict-accumulator entirely. Postgres's own disk-spilling hash aggregate
   handles "finalize a basket only once every one of its rows has been
   seen" natively, with no assumption about row order and no per-basket
   accumulator ever held in Python memory. Every product a household bought
   in a given **week** becomes one basket (`household_number` +
   `year_week_number` → `basket_id`) — see the grain warning at the top of
   this doc. No category/theme split. Baskets with fewer than
   `MIN_BASKET_PRODUCTS = 2` items are dropped. The resulting `baskets_train`
   table is never pulled into Python as one DataFrame — every consumer below
   streams bounded chunks or a bounded sample from it instead.
   - The old `BASKET_COUNT_WARN_THRESHOLD` RAM-estimate warning is gone —
     basket count alone no longer risks an OOM crash the way it used to,
     since `baskets_train` lives in Postgres, not a Python object. If the
     basket count still looks unexpectedly large, that's a data-sampling
     question (check `ns_household_tpnb_week_agg_train.sql`'s
     `MOD(household_number, N) = 0` filter), not a memory one.

2. **Co-purchase matrix**: built in row-chunks of `COPURCHASE_CHUNK_BASKETS`
   baskets at a time (default 500,000), streamed from Postgres via
   `basket_store.stream_basket_chunks()` (a bounded keyset-paginated read,
   not a slice of an in-memory frame) and checkpointed to disk after every
   chunk so a crash resumes instead of restarting — mathematically identical
   to computing `X.T @ X` over the whole population in one shot, not an
   approximation. Produces `copurchase_sparse` (product×product co-purchase
   counts), `product_id_to_index`, and `product_units_avg` (mean quantity per
   product, itself computed via one bounded SQL aggregate, not a Python
   accumulator) — saved to `data/output/` immediately since
   `score_new_baskets.py` needs them later.

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
     The co-purchase matrix is kept in its native dtype here (no whole-matrix
     `.astype(float32)` copy) — the per-basket submatrix casts to float32 on
     its own small slice instead, avoiding a redundant multi-GB copy of the
     entire matrix.
   - `basket_store.sample_training_baskets()` — stratified sampling **by
     basket size only** (bins: 1, 2-5, 6-15, 16-50, 50+), capped at
     `N_TRAIN_SAMPLES = 300,000` (set in `GNN_Train.py`), computed via SQL
     against the indexed `size_bucket` column `build_baskets_table()`
     persisted — returns one bounded DataFrame, the only basket-shaped
     object that's ever fully resident in Python memory anywhere in this
     pipeline.
   - `lmdb_graph_cache.load_or_build_lmdb_cache()` — builds one PyTorch
     Geometric graph per sampled basket via `build_one_graph()` (nodes =
     products in the basket with the 4 extra features above, edges = each
     product's top-`TOP_K=10` co-purchase partners **within that basket**,
     edge features `[log(co-purchase count), relative strength vs. that
     node's own strongest link]`, each basket's co-purchase values from a
     small **per-basket** dense submatrix sliced from the sparse global
     matrix on the fly) — but instead of holding all built graphs in one
     Python list (~20-25GB at real sample sizes), each graph is written
     straight to an LMDB key-value store (`data/output/training_graphs.lmdb/`)
     and discarded. `LMDBGraphDataset` then gives `DataLoader` lazy,
     random-access reads by index during training, with the OS page cache
     — not a pinned Python list — deciding what stays "hot". A manifest
     (`graph_builder_version`, seed, sample count, `in_dim`) guards against
     silently training on a stale cache built under a different feature
     layout.
   - **Model** (`BasketGNN` in `GNN_Train.py`): a graph autoencoder —
     `node_encoder (Linear) → GINEConv → GINEConv → global_mean_pool → proj`
     produces the basket embedding (`out_dim = 64`); a `decoder` reconstructs
     the mean node feature vector, and the training loss is MSE between
     reconstruction and the actual per-basket mean node features. Trained for
     `EPOCHS = 20` with Adam (`lr=1e-4`), gradient clipping, and AMP
     (autocast/GradScaler) if CUDA is available. NaN-producing batches are
     detected and skipped rather than corrupting the model. `DataLoader`
     workers (`NUM_WORKERS`) are safe to enable on Windows now that
     `pipeline_main.py` runs under an `if __name__ == "__main__":` guard —
     each worker opens its own LMDB read handle lazily rather than
     inheriting one from the parent process.
   - **Inductive inference** (`GraphBuilder.run_inference()` +
     `merge_inference_output()`) — after training, builds a **real graph for
     every basket** in the full population (not just the sampled training
     subset) using the exact same `build_one_graph()` function, and runs it
     through the model's full `encode()` path. Unlike training's LMDB
     approach, this is restartable rather than cached: it claims one chunk
     at a time from a Postgres work-queue table
     (`inference_chunks_train`, via `FOR UPDATE SKIP LOCKED`), writes that
     chunk's embeddings straight to its own parquet file, and marks the
     chunk complete — so a crash partway through a long run (say, 8 hours
     in) resumes from whatever's left pending rather than starting over.
     `merge_inference_output()` concatenates every completed chunk's file
     into the final `basket_gnn_embeddings.parquet` once nothing is left
     claimable. The chunk-queue design is also what makes it possible,
     later, for more than one machine to claim chunks against the same
     Postgres instance — today it just runs as one process.
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
2. Builds new whole-baskets (week grain) from the new transactions using the
   **same shared `basket_store.py` functions** `pipeline_main.py` Stage 0/1
   uses (`load_raw_export_to_postgres` + `build_baskets_table`), tagged
   `dataset_tag="score"` instead of `"train"` so the two never collide in
   the same Postgres instance — one implementation, not two that can drift.
3. Filters out any basket_id that's already been embedded before
   (`basket_store.exclude_existing_basket_ids()` — a SQL anti-join against a
   streamed `basket_id`-only table, not a Python set built from the entire
   training population).
4. Embeds the new baskets via `GraphBuilder.run_inference()` — same
   restartable, chunked-from-Postgres function training's inference stage
   uses.
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
2. **Functional** — builds a tiny synthetic catalog and a synthetic raw
   household×tpnb×week export, and actually runs the real pipeline against a
   real (throwaway) local Postgres instance: `prepare_globals()` →
   `basket_store.load_raw_export_to_postgres()` + `build_baskets_table()` →
   `sample_training_baskets()` → `lmdb_graph_cache.load_or_build_lmdb_cache()`
   → `GraphBuilder.run_inference()` + `merge_inference_output()`, asserting
   training and inference produce node feature tensors of identical width
   and that both exercise the full GNN encoder (not a shortcut). Requires
   `pgserver` installed. Good to run once after any environment setup,
   before trusting a real run.

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
#    data or basket-grain change: basket_gnn_model.pt,
#    copurchase_sparse.npz, product_id_to_index.pkl, product_units_avg.pkl,
#    basket_gnn_embeddings.parquet, gmm_basket_model.pkl,
#    basket_need_state_clusters.parquet, embeddings_chunk_*.parquet,
#    training_graphs.lmdb/ (and its .manifest.json)
#    (product_subclusters.pkl / product_embeddings.parquet do NOT need
#    deleting — they're product-text-derived, not basket-derived. The LMDB
#    cache also self-checks a manifest and rebuilds automatically on
#    mismatch, so deleting it manually is a belt-and-suspenders step, not
#    strictly required.)

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
| `N_TRAIN_SAMPLES` | `GNN_Train.py` | 300,000 | With the LMDB-backed training cache this is no longer a hard memory ceiling — raise it for more training data whenever you want, independent of RAM |
| `COPURCHASE_CHUNK_BASKETS` | `pipeline_main.py` | 500,000 | Baskets per co-purchase-matrix chunk, streamed from Postgres; lower if still memory-constrained |
| `INFERENCE_CHUNK_BASKETS` | `pipeline_main.py` | 50,000 | Baskets per restartable inference chunk (see `GraphBuilder.run_inference`) |
| `batch_size` (of `load_raw_export_to_postgres`) | `basket_store.py` | 20,000,000 | Rows per streamed read batch from the household×tpnb×week export into Postgres; lower if still memory-constrained at Stage 0 |
| `PG_WORK_MEM` / `PG_MAINTENANCE_WORK_MEM` | `basket_store.py` | 2GB | Postgres session tuning for the basket aggregation/sampling queries — a speed knob, not a correctness ceiling (Postgres spills to disk past this regardless) |
| `SUBCL_K_CANDIDATES` | `GraphBuilder.py` | [50, 100, 200, 400] | Global product sub-cluster K candidates — tune to catalog size |
| `TOP_K` | `GraphBuilder.py` | 10 | Co-purchase edges kept per node per basket graph |
| `EPOCHS` / `LR` | `GNN_Train.py` | 20 / 1e-4 | GNN training |
| `LEIDEN_RESOLUTION` | `cluster_basket_embeddings.py` | 1.0 | Run `sweep_resolution()` first rather than trusting this |
| `GMM_N_COMPONENTS` | `pipeline_main.py` | 30 | Explicit placeholder — replace with real best-K logic, or eyeball `select_k_via_bic()` |
| Training/scoring period ranges | `ns_household_tpnb_week_agg_train.sql` / `..._score.sql` | `202603–202604` | Must match `ns_item_lookup_tpna.sql`'s range for training; scoring range should be later |

See `data/TABLE_REFERENCE.md` for what each underlying warehouse table
actually contains, and why basket grain landed on WEEK rather than a true
single-visit basket.
