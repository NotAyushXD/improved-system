# Context for reviewing this codebase

I'm uploading the full contents of a data pipeline. Please read this context
first, then wait for my actual question/task at the end (or in my next
message) before proposing changes.

## What this pipeline does

This is a **need-state discovery pipeline** for retail transaction data: it
takes raw household-purchase data from a warehouse, trains a Graph Neural
Network (GNN) to embed each household's shopping basket into a vector, then
clusters those basket embeddings to discover "need-states" — data-driven
groupings of shopping baskets representing *why* a customer shopped,
independent of product category. Grocery/retail domain; products are
identified by `tpnb` (item-level barcode) and `tpna` (style/archetype level,
grouping size/color variants of the same product).

**It is deliberately theme-free**: no product category/theme concept is used
anywhere to pre-group products or baskets. There used to be a
category/"theme" based version of this pipeline; it was refactored away on
purpose (see repeated "theme-free" comments throughout the code, and
`test_pipeline.py`, which fails CI if any theme/category concept
is reintroduced). Do not suggest reintroducing category-based grouping
upstream of clustering unless I explicitly ask for it.

## Folder structure

```
project/
├── data/
│   ├── ns_item_lookup_tpna/                    (input — Spark/Databricks export, folder of part-files)
│   ├── ns_tpnb_to_tpna_mapping/                 (input — folder)
│   ├── ns_household_tpnb_week_agg_train/       (input — folder; THIS IS THE BIG ONE, see scale below)
│   ├── ns_household_tpnb_week_agg_score/       (input — folder, used later by score_new_baskets.py)
│   ├── *.sql                                     (SQL run manually against the warehouse to produce the folders above)
│   ├── TABLE_REFERENCE.md                        (what each underlying warehouse table contains)
│   └── output/                                   (ALL pipeline-written artifacts land here — nothing is written
│                                                    to the working directory. Includes: product_embeddings.parquet,
│                                                    product_subclusters.pkl, training_graphs.pkl,
│                                                    copurchase_sparse.npz, product_id_to_index.pkl,
│                                                    product_units_avg.pkl, basket_gnn_model.pt,
│                                                    basket_gnn_embeddings.parquet, gmm_basket_model.pkl,
│                                                    basket_need_state_clusters.parquet, and score_new_baskets.py's
│                                                    new_*/*_merged outputs)
└── src/
    ├── build_product_embeddings.py   Embeds product text (TPNA grain, brand excluded) via sentence-transformers;
    │                                 writes data/output/product_embeddings.parquet. Run BEFORE pipeline_main.py.
    ├── parquet_loader.py             Reads the two warehouse exports (product embeddings, household x tpnb x
    │                                 week purchases) into the DataFrame shapes the rest of the pipeline expects.
    ├── pipeline_main.py              Main entry point / orchestrator. Stage 0: read exports. Stage 1: build whole
    │                                 baskets (no category split), build the co-purchase matrix, train the GNN via
    │                                 GNN_Train.py. Stage 2: cluster basket embeddings (Leiden AND GMM, both kept).
    │                                 Stage 3: (manual) reload need-states into the warehouse.
    ├── GraphBuilder.py               Builds per-basket PyTorch Geometric graphs; global (whole-catalog) product
    │                                 sub-clustering; the co-purchase-derived edge features; inductive inference
    │                                 (embed_all_baskets_fast) used by both training and scoring.
    ├── GNN_Train.py                  BasketGNN model (GINEConv graph autoencoder) + training loop
    │                                 (train_and_embed()), which also calls GraphBuilder functions.
    ├── cluster_basket_embeddings.py  Clusters basket embeddings via mutual-kNN + Leiden AND via GaussianMixture
    │                                 (GMM); compares the two (Adjusted Rand Index); also handles assigning new
    │                                 baskets to existing clusters.
    ├── score_new_baskets.py          Scores new/held-out-period baskets against an already-trained model, no
    │                                 retraining — reuses GraphBuilder's exact graph-construction code.
    ├── test_pipeline.py   Sanity test: (1) AST-walks all active files, fails if any theme/category
    │                                 identifier exists anywhere; (2) functional check building synthetic data and
    │                                 asserting training/scoring paths produce consistent graph features.
    └── requirements.txt              torch/torch-geometric must be installed manually first (platform/CUDA
                                      specific), everything else is normal pip.
```

## Real data scale (this matters — several fixes below exist BECAUSE of these numbers)

- **198,645** distinct products (tpnb) in the catalog
- **1,025,332,391** raw household×tpnb×**period** rows in the training export
  (this was measured before the basket-grain fix below moved from period to
  week grain — expect roughly the same row count at week grain too, since
  the source table `cltv_hh_metrics_tpnb_base` is already at week grain and
  the period version was just summing weeks together)
- **21,978,316 baskets** at the OLD period grain (~4 weeks/basket). The
  week-grain rebuild was predicted to give "roughly 4x more baskets"; the
  measured figure is **57,115,804**, so ~2.6x. Confirmed against a real run,
  not a guess.
- **~47 products/basket on average at period grain** (noticeably fewer per
  basket at week grain, since each basket now spans 1 week instead of ~4)
- Running on a Windows machine: `E:\ayp\improved-system`, PowerShell, venv at
  `src\.venv`, **Python 3.13, 64 logical cores, 512 GB RAM, no GPU**

## Basket grain — WEEK, not a true single-visit basket

None of the 4 tables available in this warehouse (`product.product`,
`LAB_INSIGHT_CUSTOMER_ANALYTICS.IN22915286_UPDATED_PRODUCT_TABLE`,
`lab_customer_value_analytics.cltv_hh_metrics_tpnb_base`,
`product.buyer_hierarchy` — see `data/TABLE_REFERENCE.md`) carry a
transaction/order/checkout identifier. `cltv_hh_metrics_tpnb_base` (the only
household-purchase table) is already pre-aggregated to WEEK grain, and its
`orders` column is a COUNT of separate orders folded into that week's total,
not a preserved per-order identity. A true single-visit basket cannot be
built from what's available. Basket construction was therefore changed from
household×PERIOD (~4 weeks, an earlier design) to household×WEEK (the
finest grain the data supports) — `ns_household_tpnb_period_agg_train.sql` /
`_score.sql` were renamed to `ns_household_tpnb_week_agg_train.sql` /
`_score.sql`, and `pipeline_main.py` / `score_new_baskets.py` / `parquet_loader.py`
were updated to match (`year_period_number` → `year_week_number` throughout,
`load_household_tpnb_period` → `load_household_tpnb_week`). This is a real
improvement (4x less occasion-mixing) but still NOT a true single-visit
basket — flag this if the user asks about basket-level accuracy or need-state
quality, and don't assume week-grain fully resolves it.

## Recent engineering history — already fixed, please don't re-suggest these

The pipeline originally crashed (silently, no traceback — consistent with an
OOM kill) partway through `pipeline_main.py`, right after printing
`"Building co-purchase matrix..."`. Root cause: `X.T @ X` over the full
~1 billion-row incidence matrix, plus a redundant "build basket lists then
explode them straight back into long-form" round trip. Since then, the
following fixes have been made (all verified by code review, but **not yet
run end-to-end on real data** — no representative dataset or matching Python
environment was available in the environment these fixes were made in):

1. **Co-purchase matrix** (`pipeline_main.py`): now built in row-chunks
   (`COPURCHASE_CHUNK_BASKETS`, default 500,000 baskets/chunk), checkpointed
   to disk after every chunk (resumable on crash), skips the
   explode-then-refactorize round trip. Mathematically identical to a
   single-shot `X.T @ X` (row-disjoint chunks are additive) — not an
   approximation.
2. **Dense co-purchase submatrix removed entirely** (`GraphBuilder.py`): a
   previous version pre-built ONE dense product×product matrix covering
   every product touched anywhere in the ~300K-basket training sample, with
   a `>200GB -> fall back to sparse` guard. Given the catalog size above, a
   random training sample touches close to the *entire* catalog (basic
   coupon-collector argument), so that matrix was heading toward
   `198,645² × 4 bytes ≈ 157GB` — under the 200GB guard, so it wouldn't even
   have triggered the fallback. **Fixed by removing the need for any such
   matrix at all**: both training (`build_training_graphs`) and inference
   (`embed_all_baskets_fast`) now build a tiny **per-basket** dense
   submatrix (bounded by that basket's own product count squared, ~47×47),
   sliced from the sparse global co-purchase matrix on the fly, via a shared
   helper `_basket_dense_cp_submatrix()`. There is no size threshold
   anywhere in this path anymore — it doesn't scale with catalog size.
3. **`embed_all_baskets_fast()`** (`GraphBuilder.py`, used by both
   `GNN_Train.py` at the end of training — over ALL 21.98M baskets — and by
   `score_new_baskets.py`): previously built a full PyG graph object for
   every basket and held ALL of them in one Python list before running any
   through the model (would need multiple **terabytes** at this N). Now
   processes baskets in chunks (`graph_chunk_size`, default 50,000),
   encoding and discarding each chunk's graphs before moving to the next.
   Only the final embeddings (22M × 64 floats ≈ 5.6GB) accumulate.
4. **`sample_baskets()`** (`GraphBuilder.py`): previously did
   `baskets.copy()` on the full 22M-row basket table (which holds
   list-typed columns) just to add two small helper columns. Now computes
   size/bin as standalone Series and only ever materializes the FINAL
   sampled subset (e.g. 300K rows) out of `baskets`.
5. **Leiden's mutual-kNN graph construction**
   (`cluster_basket_embeddings.py`, `build_basket_knn_graph()`): previously
   a pure-Python double loop over ~22M baskets × 15 neighbors, accumulating
   into a plain dict that could reach hundreds of millions of entries (both
   a severe memory cost and an impractically slow pure-Python loop at this
   N). Rewritten to be fully vectorized with NumPy (including a
   sorted-array `np.searchsorted` lookup replacing the dict-based mutual-kNN
   membership check) — same neighbors, same similarities, same mutual-kNN
   rule, just computed as arrays instead of per-pair Python.
   *(Superseded in scope: the mutual-kNN rule itself is no longer used — see
   item 15 below — and the basket count is now 57.1M, not 22M. The
   vectorisation described here still stands and still runs.)*
6. **All output paths consolidated** to `data/output/` (previously
   scattered across the working directory with one path mismatch between
   what `build_product_embeddings.py` wrote and what `pipeline_main.py` /
   `score_new_baskets.py` read).

## What HAS been verified — on the real machine, against real data

**Measured production scale** (this supersedes any earlier numbers in this
document): 57,115,804 baskets; 154,597 products in the co-purchase matrix
with 1,474,348,846 non-zeros (mean 9,537 per product row); 198,652 products
with embeddings; global product sub-clustering selected k=400.

**The machine:** CPU-only Windows box, **64 logical cores, 512 GB RAM**,
Python 3.13. Earlier revisions of this document called it
"memory-constrained" — that was wrong, and it misdirected a round of work. A
full Stage 2 run peaks near 78 GB, about 15% of the box. The binding
constraints are single-threaded libraries and graph construction, not RAM.

**Reached end to end:** Stage 0 (basket build), Stage 1a (co-purchase
matrix), product sub-clustering, LMDB training-graph cache (299,999 graphs),
20 epochs of GNN training (2h25m, final avg_loss 3.2e-5), **the full inference
pass embedding all 57,115,804 baskets** (1,143 chunks), and **Stage 2a
clustering**.

**`test_pipeline.py` passes all six groups on the target machine**, including
checks that are not toy-scale in nature: the per-basket submatrix rewrite is
bit-identical to the original over random matrices; chunked co-purchase
accumulation equals single-shot exactly; training and inference build
byte-identical graphs (node *and* edge features); config fingerprints move
when a graph-affecting parameter changes and stay put otherwise.

**Fixed after being observed in production, not in theory:**

7. **Per-basket co-purchase slicing was ~21ms/basket** (≈9 days for 57M).
   `csr[rows][:, cols]` materialised entire rows — ~9,537 non-zeros each —
   to keep ~900. Replaced with a binary search over sorted indices;
   bit-identical, verified over 3,000 random matrices.
8. **Edge relative-strength used the wrong denominator** for any node with
   ≤ TOP_K neighbours (i.e. every basket of ≤11 products), producing values
   above 1.0. `GRAPH_BUILDER_VERSION` bumped to 3.
9. **The model was saved AFTER the multi-hour inference pass**, so any
   interruption destroyed the weights and orphaned every completed chunk
   while the queue happily "resumed" into a different model. Now saved
   before, and training is seeded.
10. **Inference built all 50,000 graphs in a chunk before encoding any**
    (~3.5GB on top of an 11.8GB matrix). This silently killed a production
    run — 0 chunks after hours, no traceback. Now streams through a
    batch-sized buffer; batch size is the memory lever, chunk size is not.
11. **Reruns rebuilt the co-purchase matrix and retrained the model** even
    though both were on disk. Both now have fingerprinted sidecars.
12. **All config moved to `config.py` + `.env`** (50 parameters, typed and
    validated at import, with cache fingerprints so a `.env` tweak cannot
    silently reuse an incompatible cache).
13. **Windows encoding**: every runtime file read now specifies UTF-8. The
    locale default is cp1252 there, and every source file contains non-ASCII
    comment characters.
14. **Stage 2.5 added** (`need_state_graph.py`): need-state adjacency,
    household transition graph, and journey queries — recovering edge
    structure the pipeline previously computed and discarded.
15. **Mutual-kNN was silently discarding half the population.** Graph vertices
    are derived from edge endpoints, so a basket left with no reciprocated
    edge never became a vertex. A k=15 run built a 28,692,907-vertex graph
    from 57,115,804 baskets and wrote the missing half out as NaN after an
    outer merge with the GMM labels — correct row count, half the column
    empty, nothing in the log. Coverage is now counted and printed on every
    build, checked against `PIPELINE_MIN_GRAPH_COVERAGE` **before** Leiden
    runs, and uncovered baskets get an explicit `-1` rather than a blank.
    Measured coverage under mutual-kNN: 41.0% at k=15, 50.2% at k=30, 53.3%
    at k=50 — it never approaches usable, so the graph is now one-directional.
16. **Leiden moved to NetworKit** (`cluster_leiden_networkit.py`). `leidenalg`
    is single-threaded and did not complete one optimiser iteration overnight
    on 28.7M vertices. `ParallelLeiden` clusters the larger 57.1M-vertex /
    508M-edge graph in ~36 minutes across 64 threads. Stage 2a therefore runs
    as its own process; `pipeline_main.py` loads its fingerprinted output and
    refuses to proceed on a mismatch rather than silently re-clustering.
17. **pynndescent's unfilled neighbour slots** — the "Failed to correctly find
    n_neighbors" warning that fires on every full-scale run — are padded with
    an out-of-range index and an infinite distance. Those indices alias onto
    real pairs under `src * n + dst` key arithmetic. Now filtered explicitly.

## What has NOT been verified yet

- **Stage 2b (GMM) has not completed at full scale.** `init_params='kmeans'`,
  sklearn's default, fits a complete k-means over all 57.1M points before EM
  iteration 1 and repeats it per `n_init` restart; it ran 40+ minutes without
  reaching the first iteration. `PIPELINE_GMM_INIT_PARAMS` now exists to skip
  that. `PIPELINE_GMM_N_COMPONENTS=30` remains an unvalidated placeholder.
- **Stages 2c and 2.5 have not run at full scale**, being downstream of 2b.
- **Embedding quality is unchecked.** Training loss is very low (3.2e-5),
  which is consistent with either a good autoencoder or a collapsed one.
  `test_pipeline.py --prod-outputs` has a collapse check that cannot run
  until embeddings exist.
- **`k=400` was selected at the ceiling** of the sub-cluster candidate range
  with silhouette still rising (0.065 → 0.078 → 0.085 → 0.096), so the true
  optimum is probably higher. Note 0.096 is weak in absolute terms.
- **Returns are unquantified.** `quantity` is a weekly SUM including refunds;
  non-positive values land on `log_units = 0.0`, same as "bought nothing".
  Inherited behaviour, not a decision.
- **Journeys rest on an ~8-week SQL window** (`202603-202604`), giving only a
  handful of transitions per household. Widening it is the single
  highest-value change to the journey work.

## What I'd like from you

[FILL IN YOUR ACTUAL QUESTION HERE — e.g. "review the attached files for any
other correctness or scalability issues", "help me test this end-to-end with
synthetic data at realistic scale", "explain X", etc.]
