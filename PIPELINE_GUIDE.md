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
                                   local embedded DuckDB (duckdb_manager.py)
                                   — baskets_train table (basket_store.py),
                                   built directly from the parquet export
                                              │
                                              ▼
                                   Stage 1: co-purchase matrix (streamed from
                                   DuckDB), train GNN (LMDB-cached training
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
                                   Stage 2.5: need-state GRAPHS
                                   (need_state_graph.py — see §6b)
                                    ├─ adjacency   : which need-states border
                                    │                 each other (undirected)
                                    └─ transitions : where households actually
                                                     go next (directed)
                                              │
                                              ▼
                                   Stage 3: manually re-upload into
                                   the warehouse (need_state_cluster,
                                   need_state_cluster_gmm per basket)

 ns_household_tpnb_week_agg_score.sql ──►  score_new_baskets.py
   (later/held-out weeks)                  (scores NEW baskets against
                                              the already-trained model —
                                              no retraining)

 Every parameter above comes from .env via src/config.py — see §2b.
```

Need-states are **discovered after the fact** by profiling what's dominant
in each cluster — themes/categories are never fed in as an input anywhere.

**Three things that surprise most readers**, each covered in detail below:

1. **A basket is a household's whole WEEK**, not a shopping trip — no
   transaction ID exists in any source table (see the warning above).
2. **Households are never nodes in any graph.** The GNN's graphs have
   *product* nodes; households only meet each other in the basket kNN graph,
   and only as (household, week) pairs. See `HOUSEHOLD_GRAPH_FLOW.md`.
3. **A need-state is an occasion, not a customer segment.** The same
   household appears in several need-states across different weeks.

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
│   ├── pipeline.duckdb                          (the local embedded DuckDB database file —
│   │                                              duckdb_manager.py manages this)
│   └── output/                                  (ALL pipeline-written artifacts — nothing is written
│                                                  to the working directory. product_embeddings.parquet,
│                                                  caches, the trained model, the LMDB training-graph
│                                                  cache, embeddings, cluster output)
├── .env                                        (YOUR local settings — gitignored, edit freely)
├── .env.example                                (committed template documenting all 50 parameters)
└── src/
    ├── config.py                    Every tunable parameter, typed + validated. Reads .env. See §2b.
    ├── build_product_embeddings.py  Step 1 — product text vectors (run once)
    ├── pipeline_main.py             Steps 2-4 — baskets, co-purchase, GNN, clustering, need-state graphs
    ├── duckdb_manager.py            Opens the local embedded DuckDB file
    ├── basket_store.py              Basket table, sampling, restartable chunk queue
    ├── lmdb_graph_cache.py          Build-once training-graph cache (disk-backed)
    ├── GraphBuilder.py              Per-basket graph construction + chunked inference
    ├── GNN_Train.py                 The BasketGNN model, training, model reuse
    ├── cluster_basket_embeddings.py Leiden + GMM over basket embeddings
    ├── need_state_graph.py          Stage 2.5 — adjacency / transitions / journeys  (see §6b)
    ├── parquet_loader.py            Reads product_embeddings.parquet
    ├── score_new_baskets.py         Score NEW weeks against the trained model
    ├── benchmark_inference.py       Measure per-basket cost before a long run  (see §11)
    ├── reset_inference.py           Clear state left by an interrupted run     (see §11)
    └── test_pipeline.py             The test suite — run before any long run   (see §8)
```

All source files are **UTF-8** and contain non-ASCII characters (box-drawing,
em dashes) in comments. Every file this code reads is opened with an explicit
encoding for that reason — on a Western-European Windows install the locale
default is cp1252, which cannot decode them. If you add a file that gets read
at runtime, specify `encoding="utf-8"`. Your `.env` must be saved as UTF-8
too (a BOM is tolerated).

---

## 2b. Configuration — everything lives in `.env`

Every tunable parameter is defined once in `src/config.py` and set in `.env`.
Nothing is hardcoded in the pipeline files any more.

**Precedence:** a real environment variable beats `.env`, which beats the
built-in default.

```powershell
cd src
python config.py                              # what WILL be used, and where each value came from
$env:PIPELINE_EPOCHS=50; python pipeline_main.py   # override for one run, no file edit
```

Always run `python config.py` before a long job. It prints all 50 values,
marks overrides with `*`, and prints the two cache fingerprints.

### Getting started

`cp .env.example .env`, then edit. `.env.example` documents every parameter
with its default and the reasoning behind it; `.env` is your working copy and
is gitignored.

### ⚠ Some parameters invalidate cached artifacts

This is the nuance that matters most. Several values change the *meaning* of
files the pipeline caches on disk. Before `config.py` existed, changing one
required a code edit; now it is one line in `.env`, so the pipeline
fingerprints them and rebuilds automatically rather than silently reusing
something built under different settings.

| Fingerprint | Covers | Protects |
|---|---|---|
| `graph_fingerprint()` | `TOP_K`, all `SUBCL_*`, `SEED` | the LMDB training-graph cache **and** the trained model |
| `subcluster_fingerprint()` | all `SUBCL_*`, `SEED` | `product_subclusters.pkl` |

So changing `PIPELINE_TOP_K` forces a training-graph rebuild *and* a retrain,
because both depend on it. Changing `PIPELINE_EPOCHS` forces a retrain but not
a graph rebuild. Changing `PIPELINE_LEIDEN_RESOLUTION` forces neither — it
runs after embeddings exist. `test_pipeline.py` asserts exactly this, in both
directions: over-invalidating would cost needless hours.

Artifacts **not** fingerprinted, which you must delete by hand if their inputs
change: `basket_gnn_embeddings.parquet`, `gmm_basket_model.pkl`,
`basket_need_state_clusters.parquet`, and the DuckDB tables
(`basket_store.drop_all(con, "train")`).

### Validation happens at import

Bad values fail immediately with a clear message, not six hours into a run:

```
ValueError: Config error: PIPELINE_GMM_COVARIANCE='banana' is not a valid
choice from ['diag', 'full', 'spherical', 'tied'].
```

Cross-parameter checks run too — `GMM_K_MIN > GMM_K_MAX` is rejected, and
`EDGE_DIM != 2` is rejected because `_build_edges_numba` emits exactly two
edge features and anything else would crash at the first `GINEConv`.

### The one deliberately-unset parameter

`PIPELINE_NUM_WORKERS` is the only parameter whose default is platform-aware:
**0 on Windows, 4 elsewhere**. On Windows the `spawn` start method re-imports
the launching module in every worker. `pipeline_main.py` has the required
`if __name__ == "__main__"` guard so workers are safe — but hardcoding a
number in `.env` would silently change behaviour depending on which OS the
file is copied to. Leave it unset unless tuning a known machine.

---

## 3. Step 0 — Run the SQL against your warehouse

These live in `improved-system/data/*.sql` and must be run manually against
your company's WAREHOUSE (no live connection to that warehouse exists
anywhere in the Python code — everything is file-based: run the SQL by
hand, download the result as parquet). Run each, then download the
resulting table as parquet into the matching `data/` folder above. See
`data/TABLE_REFERENCE.md` for what the underlying source tables
(`product.product`, `cltv_hh_metrics_tpnb_base`, etc.) actually contain.

(Separately, the Python pipeline *does* use a database — a private, local,
embedded DuckDB file at `data/pipeline.duckdb`, managed automatically by
`duckdb_manager.py`. That's an implementation detail of this pipeline's own
basket storage, not a connection to your warehouse. An earlier version of
this used a self-contained Postgres instance instead — dropped after its
`initdb` setup step was blocked by a Windows permission restriction on the
target machine; DuckDB needs no such privileged step.)

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
be deleted manually. Basket storage itself now lives in a local, embedded
DuckDB database (`data/pipeline.duckdb`, see below) rather than a Python
object or a plain file — if you want a fully clean rebuild for a given
`dataset_tag`, `basket_store.drop_all(con, dataset_tag)` drops its DuckDB
tables too.

### Stage 0 — Read warehouse exports, build baskets directly from parquet

- `parquet_loader.load_product_embeddings()` → `product_df_2` (tpnb,
  embedding) — one shot, small file, unchanged from before.
- `duckdb_manager.get_connection()` opens (or reuses) a **local embedded
  DuckDB database** at `data/pipeline.duckdb` — no server process, no
  privileged setup step (an earlier version of this used a self-contained
  Postgres instance via `pgserver` instead — dropped after `initdb`'s
  directory-permission step was blocked outright on the target machine,
  even as Administrator; DuckDB never does anything like that).
- `basket_store.build_baskets_table()` reads the household×tpnb×week export
  **directly from its parquet files** via DuckDB's `read_parquet()` —
  DuckDB streams and aggregates straight off disk, so there's no separate
  "load raw rows into a staging table" step at all (the Postgres-based
  design needed one; DuckDB's native parquet scanning makes it
  unnecessary). This is what replaces the old `pyarrow.lib.ArrowMemoryError`
  risk: at week grain this export routinely reaches billions of rows, large
  enough that a single `pd.read_parquet()` could fail outright even with
  plenty of total RAM free — DuckDB never materializes the raw flat table
  as one object, in Python or otherwise.

### Stage 1 — Build baskets in DuckDB, train the GNN, embed every basket

1. **Basket aggregation, in DuckDB**: `basket_store.build_baskets_table()`
   runs one SQL pass over `read_parquet(...)` — `GROUP BY household_number,
   year_week_number` with `list(tpnb)` / `list(quantity)` — that replaces
   the old Python dict-accumulator entirely. DuckDB's own disk-spilling
   aggregate handles "finalize a basket only once every one of its rows has
   been seen" natively, with no assumption about row order and no
   per-basket accumulator ever held in Python memory. Every product a
   household bought in a given **week** becomes one basket
   (`household_number` + `year_week_number` → `basket_id`) — see the grain
   warning at the top of this doc. No category/theme split. Baskets with
   fewer than `MIN_BASKET_PRODUCTS = 2` items are dropped. The resulting
   `baskets_train` table is never pulled into Python as one DataFrame —
   every consumer below streams bounded chunks or a bounded sample from it
   instead.
   - The old `BASKET_COUNT_WARN_THRESHOLD` RAM-estimate warning is gone —
     basket count alone no longer risks an OOM crash the way it used to,
     since `baskets_train` lives in DuckDB, not a Python object. If the
     basket count still looks unexpectedly large, that's a data-sampling
     question (check `ns_household_tpnb_week_agg_train.sql`'s
     `MOD(household_number, N) = 0` filter), not a memory one.

2. **Co-purchase matrix**: built in row-chunks of `COPURCHASE_CHUNK_BASKETS`
   baskets at a time (default 500,000), streamed from DuckDB via
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
     at a time from a DuckDB work-queue table (`inference_chunks_train`),
     writes that chunk's embeddings straight to its own parquet file, and
     marks the chunk complete — so a crash partway through a long run (say,
     8 hours in) resumes from whatever's left pending rather than starting
     over. `merge_inference_output()` concatenates every completed chunk's
     file into the final `basket_gnn_embeddings.parquet` once nothing is
     left claimable. **Single-process only**: DuckDB has no row-level
     locking (no equivalent of Postgres's `FOR UPDATE SKIP LOCKED`), so this
     claim loop is correct for one process claiming chunks sequentially but
     is not safe if ever called concurrently from multiple processes at
     once — a real limitation versus the Postgres-based design this
     replaced, kept honest here rather than papered over.
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
   **same shared `basket_store.py` function** `pipeline_main.py` Stage 0/1
   uses (`build_baskets_table`, reading the new transactions parquet
   directly), tagged `dataset_tag="score"` instead of `"train"` so the two
   never collide in the same DuckDB database — one implementation, not two
   that can drift.
3. Filters out any basket_id that's already been embedded before
   (`basket_store.exclude_existing_basket_ids()` — a SQL anti-join against a
   streamed `basket_id`-only table, not a Python set built from the entire
   training population).
4. Embeds the new baskets via `GraphBuilder.run_inference()` — same
   restartable, chunked-from-DuckDB function training's inference stage
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

## 6b. Stage 2.5 — need-state graphs (`need_state_graph.py`)

Stage 2 produces need-state *labels*. Stage 2.5 produces the **edges between
need-states**, which earlier versions computed and threw away:
`cluster_basket_embeddings()` built the basket kNN edge list, handed it to
Leiden, and let it go out of scope. `pipeline_main.py` now calls
`build_basket_knn_graph()` and `run_leiden_on_basket_graph()` separately so
the edge list survives.

**Two edge sets, and they are not interchangeable.** This is the single most
important nuance in this stage:

| | Adjacency (undirected) | Transitions (directed) |
|---|---|---|
| built from | contracting the basket kNN graph | sequencing each household's own weeks |
| means | "these two occasions border each other" | "after A, households go to B next week" |
| use for | substitutable / easily-confused need-states | **journeys**, next-best-action |
| file | `need_state_adjacency.parquet` | `need_state_transitions.parquet` |

**Similarity is not movement.** Households do not travel along adjacency
edges. Reading journeys off the adjacency graph gives a plausible-looking
answer that is wrong.

```python
import pandas as pd, need_state_graph as nsg
clusters = pd.read_parquet("../data/output/basket_need_state_clusters.parquet")
trans    = pd.read_parquet("../data/output/need_state_transitions.parquet")

nsg.journeys_for_household(clusters, trans, household_number=4213)
# -> current_need_state, as_of_week, history, next_steps, journeys
```

Nuances worth knowing:

- **`need_state_gmm_overlap.parquet` uses GMM component ids**, a *different*
  labelling from Leiden's `need_state_cluster`. Its columns are named
  `gmm_component_a/b` deliberately so a join against the other tables fails
  loudly instead of silently returning nonsense. Relate the two through
  `basket_id` in `basket_need_state_clusters.parquet`, which carries both.
- **`basket_knn_edges.parquet` is capped** at `PIPELINE_BASKET_EDGE_WRITE_CAP`
  (50M). At full population the kNN graph can exceed 1B edges — hundreds of
  GB — so above the cap the write is skipped with a message. Expect this at
  57M baskets; it is correct behaviour, and the need-state-grain tables are
  always written.
- **Journeys are first-order Markov** — they assume where a household goes
  next depends only on where it is now. That is untested and, on an ~8-week
  window, untestable. Read multi-step paths as plausible routes, not
  forecasts, and check `low_support_steps` first.
- **`PIPELINE_TRANSITION_MAX_WEEK_GAP`** decides how a household's weeks
  become transitions. `1` counts only consecutive calendar weeks; `none`
  counts consecutive *observed* baskets however far apart. Households do not
  shop every week, so on a short window `1` yields very few transitions.
  `.env` ships with `none` for this reason — the trade-off is that a 1-week
  and a 6-week step are treated alike, which `avg_week_gap` on every row lets
  you check.
- **A household is never a node.** It survives only as a prefix inside
  `basket_id`, and nothing links a household's own weeks to each other.
  See `HOUSEHOLD_GRAPH_FLOW.md` for the full picture.

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

**Run this before any long job.** It is fast, and it has caught real bugs
that would otherwise have surfaced hours into a run.

```powershell
python .\test_pipeline.py              # everything
python .\test_pipeline.py --fast       # skip the DuckDB/LMDB/torch functional check
python .\test_pipeline.py --prod-outputs   # also check real artifacts in data/output
```

Six groups, all must pass (exit code 0):

| # | Group | What it protects |
|---|---|---|
| 1 | **Static** | No `theme`-named identifier can reappear; `split_basket_by_theme.py` stays deleted |
| 1b | **Config** | Defaults match the original hardcoded values; types are real; **cache fingerprints move when they should and stay put when they shouldn't**; bad values fail at import; `.env` precedence and UTF-8 parsing; `.env.example` documents every key |
| 2 | **Graph primitives** | `_minmax` edge cases, the memory-safe distance expansion, top-K edge selection, **edge relative-strength scaling**, the **bit-identical** per-basket submatrix rewrite, exact node-feature slot layout, duplicate-product unit summing, **zero/negative quantities (returns)** |
| 3 | **Co-purchase additivity** | Proves chunked `XᵀX` equals single-shot exactly — every edge feature depends on this and it was previously only a claim in a comment |
| 7 | **Need-state graph** | Adjacency maths pinned to hand-computed values, year-boundary week ranking, transition counts/probabilities, journey beam search, GMM overlap batch-invariance and column naming |
| 4-6 | **Functional / parity / basket store** | Real DuckDB + real LMDB on synthetic data; **train/score parity by VALUE, not just width**; basket grain semantics and the `basket_id` format journeys depend on |
| 8 | **Prod outputs** (opt-in) | Embedding collapse, degenerate clusters, label coverage — things synthetic data structurally cannot catch |

Two limits worth knowing. The functional check uses 25 baskets, so it cannot
catch **memory behaviour at production chunk sizes** — a 50,000-graph
accumulation is invisible at that scale, and that is exactly the bug that
once produced "0 chunks complete" after hours of running. And `--prod-outputs`
only checks artifacts that already exist; it skips cleanly otherwise.

---

## 9. Quick-reference: run order

```powershell
cd src

# 0. Run the SQL in data/ against your warehouse, download each result as
#    parquet into the matching data/ subfolder.

# 1. Configure, and check what will actually be used
copy ..\.env.example ..\.env      # first time only, then edit
python .\config.py

# 2. Verify the code before spending hours on it
python .\test_pipeline.py

# 3. Product embeddings (re-run only if the product catalog/attributes change)
python .\build_product_embeddings.py

# 4. OPTIONAL — measure per-basket inference cost before committing to it
python .\benchmark_inference.py --n-baskets 500

# 5. The main run. Tee the log: it is long, and the per-chunk ETA lines
#    are what you will want to read back.
python .\pipeline_main.py 2>&1 | Tee-Object -FilePath run.log
#    -> data/output/basket_need_state_clusters.parquet   (upload to warehouse)
#    -> data/output/need_state_adjacency.parquet         (Stage 2.5)
#    -> data/output/need_state_transitions.parquet       (Stage 2.5)

# 6. Confirm the output actually means something
python .\test_pipeline.py --fast --prod-outputs

# 7. (Later, periodically) score new weeks without retraining
python .\score_new_baskets.py --new-transactions ..\data\ns_household_tpnb_week_agg_score
```

**On a rerun, steps 3-5 reuse what they can:** the co-purchase matrix, the
LMDB training-graph cache, the trained model, and every completed inference
chunk. You do not need to delete anything by hand unless a fingerprint check
tells you it is stale — and then it rebuilds automatically. The old advice to
"delete stale caches before running" no longer applies; the fingerprints do
that job now, and do it more reliably than a checklist.

If a run died and the **model file is missing**, see `reset_inference.py` in
§10b before rerunning.

---

## 10b. Running at production scale — operations

Measured on the real data: **57,115,804 baskets**, **154,597 products** in the
co-purchase matrix, **1,474,348,846 non-zeros** (mean 9,537 per product row),
198,652 products with embeddings. CPU-only Windows box. These numbers drive
everything below.

### What each stage costs

| Stage | Cost | Reused on rerun? |
|---|---|---|
| Stage 0 — build basket table | ~30-60 min | No, always rebuilds |
| Stage 1a — co-purchase matrix | hours | **Yes** — fingerprinted `.meta.json` sidecar |
| LMDB training-graph cache | ~3 min | Yes, unless `graph_fingerprint` changed |
| Training (20 epochs, 300k graphs) | ~2h25m | **Yes** — model manifest |
| Inference (57M baskets) | the long pole | Yes, per-chunk |
| Stage 2 — clustering | see the warning below | No |

### Watch the first chunk

```
chunk 0 done in 71s (704 baskets/s, 1,420 us/basket) | 1/1,143 chunks | ETA 22.5 h
```

That line prints the moment chunk 0 finishes, with an ETA from this run's
rolling average. A per-basket progress bar appears within seconds. **If the
rate looks wrong, stop immediately** — do not discover it on day nine.

### Memory is bounded by batch size, not chunk size

`PIPELINE_INFERENCE_CHUNK_BASKETS` controls how much work is *checkpointed*;
`PIPELINE_INFERENCE_BATCH_SIZE` controls how much is held in **RAM**. Graphs
are built into a buffer of batch-size, encoded, and discarded. The co-purchase
matrix alone is ~11.8 GB resident, so:

| `INFERENCE_BATCH_SIZE` | approximate peak |
|---:|---:|
| 8,192 | ~12.6 GB |
| 2,048 (recommended here) | ~12.2 GB |

**If you hit memory pressure, halve the batch size.** Lowering the *chunk*
size does not help.

### If a run dies

Inference is restartable: completed chunks are skipped, the interrupted one
returns to the queue. The trained model is saved **before** inference starts,
so the weights survive. Just rerun `pipeline_main.py`.

**Only reset if the model file is missing.** `reset_inference.py` handles
this and refuses to delete when a model exists:

```powershell
python .\reset_inference.py          # dry run
python .\reset_inference.py --yes
```

Why it matters: a model that no longer exists cannot be reproduced without
the same seed, so chunks it embedded can never be safely mixed with chunks
from a new model. Two incompatible embedding spaces in one clustering fails
silently — nothing errors, the need-states are just wrong.

### Verify the output means something

```powershell
python .\test_pipeline.py --fast --prod-outputs
```

Checks the real artifacts for **embedding collapse** (if the autoencoder
learned a near-constant, every basket lands in the same place and clustering
is meaningless while appearing to succeed), degenerate clusters, and label
coverage. Worth running the moment inference finishes — a low training loss
does *not* rule collapse out.

### ⚠ Stage 2 at 57M baskets is not solved

| | at 57M × 64 dims |
|---|---|
| embeddings in memory | ~15 GB |
| `normalize()` copy | **~29 GB peak** |
| mutual-kNN edges | ~300M undirected, several GB |
| GMM working array | **~14 GB per EM iteration**, ×3 inits |

On a memory-constrained box Stage 2 may not complete. The intended fix —
cluster a few-million-basket sample, then assign the rest with
`assign_new_baskets_to_clusters` / the saved GMM's `predict` (both already
exist, that is what `score_new_baskets.py` does) — **is not yet wired into
the main run.**

### Known data nuance: returns

`quantity` is a weekly SUM including refunds, so it can be 0 or negative. A
product bought and returned in the same week nets to 0; a net refund goes
negative. `log1p` is undefined at/below −1, so those become `0.0` in the
`log_units` node feature — the same value as "bought nothing". That is
inherited behaviour, not a deliberate modelling choice. Quantify it before
relying on that feature:

```sql
SELECT COUNT(*) FILTER (WHERE quantity <= 0) AS non_positive, COUNT(*) AS total
FROM read_parquet('../data/ns_household_tpnb_week_agg_train/*.parquet');
```

If the share is large, decide deliberately in SQL whether returns should be
filtered, rather than inheriting a `nan_to_num` side effect.

---

## 11. Diagnostic scripts

**`benchmark_inference.py`** — run before committing to a long inference pass.
Reports old-vs-new equivalence of the per-basket submatrix lookup on *your*
matrix, a per-stage cost breakdown in microseconds, and a projected wall
clock. Two minutes.

```powershell
python .\benchmark_inference.py --n-baskets 500 --remaining-baskets 57115804
```

**`reset_inference.py`** — clears orphaned chunk state (see §10b). Dry-run by
default; refuses to act when a model file exists unless `--force`.

---

## 12. Key knobs you'll likely want to revisit

Every one of these now lives in `.env`, not in a source file. Run
`python config.py` to see the effective value of all 50.

| `.env` key | Default | Note |
|---|---|---|
| `PIPELINE_MIN_BASKET_PRODUCTS` | 2 | Below 2 a basket has no co-purchase pair at all |
| `PIPELINE_N_TRAIN_SAMPLES` | 300,000 | LMDB-backed, so no longer a RAM ceiling — raise freely |
| `PIPELINE_COPURCHASE_CHUNK_BASKETS` | 500,000 | Lower first under memory pressure. Does not change results |
| `PIPELINE_INFERENCE_CHUNK_BASKETS` | 50,000 | Checkpoint granularity — **not** a memory lever |
| `PIPELINE_INFERENCE_BATCH_SIZE` | 8,192 (2,048 here) | **The** inference memory lever |
| `PIPELINE_TOP_K` | 10 | ⚠ rebuilds graph cache + forces retrain |
| `PIPELINE_SUBCL_K_CANDIDATES` | 50,100,200,400 | ⚠ same. On this catalog k=400 was picked at the range ceiling with silhouette still rising — consider extending |
| `PIPELINE_EPOCHS` / `PIPELINE_LR` | 20 / 1e-4 | ⚠ forces retrain. LR reduced from 1e-3 to prevent divergence |
| `PIPELINE_LEIDEN_RESOLUTION` | 1.0 | Placeholder — run `sweep_resolution()` first |
| `PIPELINE_GMM_N_COMPONENTS` | 30 | Placeholder — see `select_k_via_bic()` |
| `PIPELINE_TRANSITION_MAX_WEEK_GAP` | `none` here | The only value in `.env` deviating from code default; see §6b |
| `PIPELINE_NUM_WORKERS` | unset | Platform-aware; leave unset (see §2b) |
| `PIPELINE_DUCKDB_PATH` | `../data/pipeline.duckdb` | Keep on a fast local disk, not a synced folder |
| Training/scoring SQL period ranges | `202603-202604` | In the `.sql` files, not `.env`. ~8 weeks — the binding constraint on journey quality |

⚠ = changing it invalidates a cached artifact, which rebuilds automatically.

See `data/TABLE_REFERENCE.md` for what each underlying warehouse table
actually contains, and why basket grain landed on WEEK rather than a true
single-visit basket.
