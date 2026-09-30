# How Households Connect → Nodes → Embedding Space → Need-States

Traced from the actual code, not the design intent. Every claim below cites
the file and line that implements it.

---

## 0. The single most important thing to know first

**There are TWO different graphs in this system, and they are easy to confuse.
Households are nodes in neither of them.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GRAPH A — "inside one basket"          GraphBuilder.build_one_graph()      │
│                                                                             │
│      NODES = PRODUCTS      EDGES = product↔product co-purchase              │
│      One graph per basket. Millions of tiny graphs.                         │
│      Households do not appear. ───────────────────► fed to the GNN          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │  GNN compresses each graph to 1 vector
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GRAPH B — "between baskets"     cluster_basket_embeddings.build_basket_    │
│                                   knn_graph()                               │
│      NODES = BASKETS  (= one household × one week)                          │
│      EDGES = kNN cosine similarity in 64-dim space (one-directional)        │
│      ONE global graph over all baskets. ──────────► fed to Leiden           │
│                                                                             │
│      ◄── THIS is the only place two households ever touch each other.       │
└─────────────────────────────────────────────────────────────────────────────┘
```

A household is never a node, never has an embedding, and is never clustered.
It survives only as a **string prefix inside `basket_id`**
(`basket_store.build_baskets_table()`):

```sql
CAST(household_number AS VARCHAR) || '_' || CAST(year_week_number AS VARCHAR)
```

Consequence worth sitting with: household 4213's week-15 shop and household
4213's week-16 shop are **two unrelated nodes** in Graph B. Nothing in the code
links a household's own weeks to each other. They land in the same need-state
only if their *contents* happen to be similar — exactly as if they were two
different households.

---

## 1. Data grain — what level is the data at, at every step

```
LEVEL                                    WHERE IT'S SET                      ROW COUNT SHAPE
─────                                    ──────────────                      ───────────────

  household × tpnb × year × period × week      cltv_hh_metrics_tpnb_base        ~billions
  (the warehouse source — already pre-aggregated to week)
         │
         │  ns_household_tpnb_week_agg_train.sql:42-56
         │  SUM(quantity), SUM(orders), SUM(sales_inc_vat)
         │  WHERE year*100+period BETWEEN 202603 AND 202604   ← ~2 periods ≈ 8 weeks
         │  ⚠ NO household sampling filter is actually present in the SQL.
         │    pipeline_main warns to "check the MOD(household_number, N) = 0
         │    filter is being applied" — that filter does not exist in any file
         │    under data/. Full household population is exported as written.
         ▼
  household × tpnb × week                      the exported parquet             ~billions
         │
         │  basket_store.build_baskets_table()
         │  GROUP BY household_number, year_week_number
         │  list(tpnb) → products,  list(quantity) → units
         │  HAVING len(list(tpnb)) >= 2          ← 1-item baskets dropped
         ▼
  household × week   ("a basket")               baskets_train in DuckDB          ~tens of M
         │
         │  GraphBuilder.build_one_graph()
         ▼
  product-within-a-basket   ("a node")          PyG Data.x                       ~hundreds of M
         │
         │  global_mean_pool + proj  (BasketGNN.encode)
         ▼
  household × week, as 64 floats                basket_gnn_embeddings.parquet    ~tens of M
         │
         │  Leiden / GMM                         cluster_basket_embeddings.py
         ▼
  household × week → need_state_cluster         basket_need_state_clusters.parquet
```

### ⚠️ The grain caveat that shapes everything downstream

A "basket" is **a whole week of one household's shopping, not one shopping
trip.** `ns_household_tpnb_week_agg_train.sql:11-29` is explicit about why:

> None of the tables available in this warehouse carry a
> transaction/order/checkout identifier. […] its `orders` column is a COUNT of
> separate orders folded into that week's total, not a preserved per-order
> identity.

So a household that did a big weekly shop *plus* two top-ups gets all three
occasions merged into one node. The `orders` column is exported (SQL line 49)
but **never read by any Python file** — it's carried for future profiling only.

This is why a 22-item basket gets a low `gmm_confidence`: it genuinely spans
several need-states at once and the model is right to be unsure.

---

## 2. Where the embedding space comes from (two separate "embeddings")

Careful — the word "embedding" means two different things in this codebase:

```
 (a) PRODUCT embedding — 384-dim, TEXT-derived, built ONCE, never trained
 ────────────────────────────────────────────────────────────────────────
     build_product_embeddings.py

     TPNA (style/archetype grain, e.g. "Semi Skimmed Milk 2L")
       │  text = description | dept | class | subclass         (line 93-99)
       │  ── brand DELIBERATELY EXCLUDED (line 88-92):
       │     brand dominates similarity on short retail text
       ▼
     all-MiniLM-L6-v2  ──►  384 floats                          (line 127-133)
       │
       │  anisotropy correction: subtract top-2 PCs             (line 102-108)
       │
       │  broadcast DOWN: every tpnb under one tpna gets the
       │  IDENTICAL vector (2-pint and 4-pint milk are the same) (line 157-158)
       ▼
     product_embeddings.parquet     one row per tpnb


 (b) BASKET embedding — 64-dim, LEARNED by the GNN, the actual clustering space
 ─────────────────────────────────────────────────────────────────────────────
     GNN_Train.BasketGNN.encode()                                (line 112-121)
     This is §4 below. THIS is "the embedding space" need-states live in.
```

(a) feeds (b). Only (b) is clustered.

---

## 3. How a NODE is created  (Graph A — inside one basket)

`GraphBuilder.build_one_graph()`. Called identically
for training and scoring — there is only one graph-building function, on
purpose, so train/score can't drift.

```
   BASKET  "4213_202615"   =  household 4213, week 2026-15
   products = [MILK, BREAD, EGGS, BUTTER]      units = [2, 1, 1, 1]

        ┌──────────────────────────────────────────────────────┐
        │  dedupe: keep first occurrence, SUM units for dupes   │
        └──────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴───────────────────────────────────────────────┐
        │  ONE NODE PER DISTINCT PRODUCT.   in_dim = 384 + 3 = 387             │
        └─────────────────────────────────────────────────────────────────────┘

   NODE "MILK"  (as it exists inside THIS basket)
   ┌────────────┬────────────────────────────┬──────────────────────────────┐
   │ slot       │ what it is                  │ varies by?                   │
   ├────────────┼────────────────────────────┼──────────────────────────────┤
   │ [0:384]    │ product text embedding      │ PRODUCT only — identical in  │
   │            │ (§2a)                        │ every basket, everywhere     │
   ├────────────┼────────────────────────────┼──────────────────────────────┤
   │ [384]      │ cp_score                    │ ◄ BASKET-SPECIFIC            │
   │            │ log1p(row_sum−diag)/(n−1),  │   depends on what ELSE is in │
   │            │ then min-max within basket  │   this basket   │
   ├────────────┼────────────────────────────┼──────────────────────────────┤
   │ [385]      │ distinctiveness             │ PRODUCT only                 │
   │            │ 1 − d(own centroid)/        │   measured against the same  │
   │            │      d(farthest centroid)   │   global K-means centroids   │
   ├────────────┼────────────────────────────┼──────────────────────────────┤
   │ [386]      │ log1p(units)                │ ◄ BASKET-SPECIFIC            │
   │            │                              │   log1p(2) here              │
   └────────────┴────────────────────────────┴──────────────────────────────┘

        2 of 4 components are product constants; 2 are basket-specific.
        That mix is the design: "what kind of product is this, globally"
        + "how is it behaving in THIS basket".
```

**`sub_cluster_id` used to sit at [385] and was removed on 2026-09-30**, which
is why everything after cp_score shifted down one slot. It was the product's
global K-means cluster id stored as `label / (k − 1)`: a ~400-way nominal label
flattened onto one scalar, so cluster 200 and 201 were adjacent and meant
nothing to each other. It was also a deterministic function of the 384-dim
embedding sitting in slots [0:384] of the very same row. The global K-means
still runs — `distinctiveness` is measured against its centroids — but the
cluster id itself is gone. See `GraphBuilder.prepare_globals`' docstring.

> [!WARNING]
> **Until 2026-09-30, slots [0:384] were all zero for every product in every
> basket.** `tpnb` is an int in the basket export and a str in both product
> extracts, so the lookup in `prepare_globals()` matched nothing — silently,
> with no else branch and no counter. `in_dim` still read 388 and nothing in
> the log was out of place. Any need-state output produced before that date
> has no product information in it at all. See `CLAUDE.md` §4b.

### Edges inside the basket

```
        Source of truth: the GLOBAL co-purchase matrix
        ──────────────────────────────────────────────
        pipeline_main Stage 1a —  copurchase = Σ over chunks of Xᵀ·X
          X = basket × product incidence, so (Xᵀ X)[a,b] = # baskets
          containing BOTH a and b, population-wide.
          Built in 500k-basket chunks, checkpointed, fingerprinted.
          Exactly additive over row-disjoint chunks → chunking changes nothing.
                              │
                              │  sliced per basket — never the full matrix
                              │  GraphBuilder._basket_dense_cp_submatrix()
                              ▼
        For basket 4213_202615, a 4×4 submatrix over {MILK,BREAD,EGGS,BUTTER}
                              │
                              │  GraphBuilder._build_edges_numba()
                              ▼
        Keep each node's TOP_K = 10 strongest partners *within this basket*

              ┌─────────┐ 13.99 / 1.00  ┌─────────┐
              │  MILK   │◄─────────────►│  BREAD  │
              └────┬────┘               └────┬────┘
                   │  ╲                 ╱    │
       13.02/0.375 │   ╲ 13.30 / 0.50  ╱     │ 13.45/...
                   │    ╲             ╱      │
              ┌────▼────┐  ╲       ╱    ┌────▼────┐
              │  EGGS   │◄──────────────►│ BUTTER  │
              └─────────┘                └─────────┘

        edge_attr = 2 features:
          [0]  log1p(co-purchase count)              — absolute strength
          [1]  count / this node's OWN strongest link — relative, self-scaled
                                                        
```

Note the edge features are **purely co-purchase-derived**. No theme, category,
or hierarchy signal enters the graph anywhere — the module docstring
(GraphBuilder.py's header) is emphatic that this replaced an earlier
theme-flag design.

**Why per-basket slicing matters:** a single dense catalog-wide co-purchase
matrix would be ~200k² × 4 bytes ≈ **160 GB**. Each basket's own submatrix is
bounded by its own product count squared — independent of catalog size
(`GraphBuilder._basket_dense_cp_submatrix`).

---

## 4. Graph A → one point in the embedding space

`GNN_Train.BasketGNN`.

```
   x [n_products_in_basket, 387]
        │
        ▼  node_encoder: Linear 387 → 128, ReLU
   [n, 128]
        │
        ▼  GINEConv #1 (uses edge_attr) → ReLU → Dropout(0.1)
   [n, 128]        each product absorbs signal from its co-purchase neighbours
        │          MILK's vector shifts because BREAD/BUTTER sit next to it
        ▼  GINEConv #2 (uses edge_attr) → ReLU
   [n, 128]
        │
        ▼  global_mean_pool   ── collapses n nodes → 1 vector  
   [1, 128]
        │
        ▼  proj: Linear 128→128 → ReLU → Linear 128→64
   [1, 64]   ◄─────  THE BASKET IS NOW ONE POINT IN 64-DIM SPACE
```

### How it's trained — no labels anywhere

```
        z (64) ──► decoder: 64→128→387 ──► recon (387)
                                             │
        target = mean of this basket's own node features  (line 129-132)
                                             │
                          MSE(recon, target)  ◄── the entire loss
```

It's a **graph autoencoder**. There is no need-state label to predict; the
model only learns to compress basket structure faithfully. Need-states are
*discovered* afterwards, by clustering — never supervised.

**Read the target carefully, though.** 384 of the 387 dimensions it
reconstructs are the basket's mean product-embedding vector, so MSE is
overwhelmingly dominated by it, and the two `GINEConv` layers mix in
neighbour information the target does not contain. The co-purchase graph
therefore contributes very little to the gradient. `baseline_mean_embedding.py`
measures what that costs: it clusters the plain mean product embedding, no GNN
at all, through the identical kNN/Leiden/profiling path. If the GNN cannot beat
that control, this whole stage is an elaborate average.

Training config: 300k baskets sampled **stratified by basket size**
(buckets ≤5 / ≤15 / ≤50 / >50, the `size_bucket` column, sampled at
`sample_training_baskets()` line 188), 20 epochs, lr 1e-4, Adam, grad-clip 1.0.
Then **every** basket in the population is embedded by the same `encode()` path
in restartable 50k chunks (`GraphBuilder.run_inference`).

---

## 5. ★ How households connect to each other  (Graph B)

**This is the answer to the core question.** `build_basket_knn_graph()`,
`cluster_basket_embeddings.build_basket_knn_graph`.

```
   Every basket is now a 64-dim point. L2-normalized (line 101).

            hh 9954, wk 11  ●
                             ╲  cos 0.94
            hh 4213, wk 15  ●─────────────●  hh 5540, wk 14
                                cos 0.61
                                             ╲
            hh 9021, wk 13  ●━━━━━━━━━━━━●  hh 2207, wk 09
                              cos 0.92
                                     ╲
                                      ● hh 7788, wk 12   (cos 0.58, weak)

   STEP 1   k = 10 nearest neighbours per basket, COSINE
            pynndescent (approximate) — sklearn brute-force fallback
            warns loudly above 100k baskets because it will never finish

   STEP 2   ONE-DIRECTIONAL  (PIPELINE_USE_MUTUAL_KNN=false)
            ┌──────────────────────────────────────────────────────┐
            │  An edge A—B survives if EITHER named the other.     │
            │  Every basket therefore keeps its k edges, so every  │
            │  basket is a vertex — 100% coverage by construction. │
            └──────────────────────────────────────────────────────┘
            weight = minmax_scale( similarity )

            ⚠ The mutual filter (both must name each other) is
              available but MUST NOT be used at this scale. It left
              41% of baskets with no edge at k=15, 50% at k=30, 53%
              at k=50 — and a basket with no edge is not a vertex,
              so it silently receives no need-state at all.
              PIPELINE_MIN_GRAPH_COVERAGE now blocks that.

   STEP 3   → edges DataFrame: basket_a, basket_b, weight
            (57,115,804 baskets → 508,168,451 edges at k=10)
```

So, precisely:

> **Two households are connected iff one household's single week of shopping
> is among the other's 10 nearest neighbours in the 64-dim GNN space** — in
> either direction.

Not by demographics. Not by store. Not by category. Not by spend. Only by
*how similarly the two weeks' baskets are structured* — which, by construction
of §3–§4, folds in what was bought, what co-purchases with what, and relative
quantities.

**Scale warning that's live in the code** (line 176-182): at full population
this graph can reach 1B+ edges, and it's flagged as the next memory bottleneck
before igraph ever sees it.

---

## 6. Clustering that graph → need-states

Two independent methods run over the **same** 64-dim vectors, and **both are
kept** — `pipeline_main` deliberately does not pick one. (With `--with-gmm`;
see the note in §6.)

```
                    64-dim basket embeddings (L2-normalized)
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                              ▼
   ┌──────────────────────┐                    ┌──────────────────────────┐
   │ 6a. LEIDEN            │                    │ 6b. GMM                  │
   │ (on Graph B, §5)      │                    │ (on the raw points)      │
   ├──────────────────────┤                    ├──────────────────────────┤
   │ kNN graph, k=10       │                    │ n_components = 30        │
   │ one-directional       │                    │   ← PLACEHOLDER, flagged │
   │       ↓               │                    │     in the code as not   │
   │ NetworKit, weighted   │                    │     the team's real      │
   │       ↓               │                    │     best_k logic         │
   │ ParallelLeiden        │                    │ covariance = "diag"      │
   │ 64 threads            │                    │       ↓                  │
   │ gamma = 1.5           │                    │ fit_predict              │
   │   ← also a starting   │                    │       ↓                  │
   │     point; sweep_     │                    │ predict_proba().max()    │
   │     resolution() is   │                    │       ↓                  │
   │     provided and does │                    │ need_state_cluster_gmm   │
   │     NOT auto-pick     │                    │ + gmm_confidence         │
   │       ↓               │                    │                          │
   │ need_state_cluster    │                    │ model SAVED to           │
   │ (community id)        │                    │ gmm_basket_model.pkl     │
   └──────────┬───────────┘                    └────────────┬─────────────┘
              │                                               │
              └───────────────────┬───────────────────────────┘
                                  ▼
                   compare_leiden_gmm()  →  Adjusted Rand Index
                   1.0 = identical, ~0.0 = random agreement
                                  ▼
                   basket_need_state_clusters.parquet
                   basket_id | need_state_cluster | need_state_cluster_gmm
                             | gmm_confidence
```

> [!NOTE]
> **The GMM half of this diagram is off by default since 2026-09-30.**
> `pipeline_main.py --with-gmm` turns it on; without it the output carries the
> Leiden columns only, and `compare_leiden_gmm` does not run.
>
> `GMM_N_COMPONENTS` is still the placeholder 30 while Leiden finds ~356
> communities, and an Adjusted Rand Index between a 30-way and a 356-way
> partition is driven toward 0 by the granularity gap alone — so the
> comparison cost a full GaussianMixture fit over 57.1M x 64 float64 and
> reported a number that said nothing about whether the two methods agree.
> Pick a real K with `select_k_via_bic()` before turning it back on.
>
> None of the GMM code changed, and the row that matters below — new baskets —
> is still why it exists.

### Why both

| | Leiden | GMM |
|---|---|---|
| operates on | the kNN **graph** | the **points** directly |
| finds | arbitrary-shaped communities | ellipsoidal components |
| picks k | emerges from resolution | fixed upfront (30) |
| **new baskets** | **transductive — cannot** place a new point natively | **can** `.predict()` directly |
| confidence | neighbour-vote agreement | posterior probability |

That last row is the practical dividing line, and it's called out explicitly at
`cluster_basket_embeddings.assign_new_baskets_to_clusters` and
`cluster_basket_embeddings_gmm`.

---

## 7. How a household joins a need-state that already exists

Two different mechanisms, both in `score_new_baskets.main()`:

```
   NEW basket  (hh 8812, week 202620)
        │
        │  built + embedded through the IDENTICAL path (§3, §4)
        │  — same build_one_graph(), same model.encode()
        ▼
   64-dim point
        │
   ┌────┴─────────────────────────────────────────────────────────┐
   │                                                               │
   ▼ LEIDEN PATH                                     GMM PATH ◄────┘
   assign_new_baskets_to_clusters()                  gmm.predict(X)
   assign_new_baskets_to_clusters()                  gmm.predict()
   │                                                 │
   │ find k=15 nearest ALREADY-CLUSTERED             │ direct posterior —
   │ reference baskets (cosine)                      │ no neighbours needed
   │         ↓                                       │
   │ MAJORITY VOTE on their need_state_cluster       ▼
   │         ↓                                  need_state_cluster_gmm
   │ cluster_confidence = fraction of the 15     gmm_confidence
   │ neighbours that agreed
   │   < 0.5 ⇒ basket sits BETWEEN need-states
   │   (counted and reported)
   ▼
   need_state_cluster
```

So a new household attaches to a need-state **through its nearest existing
households** (Leiden path) or **through the fitted density model** (GMM path).

`score_new_baskets.py` runs the Leiden path always, and the GMM path only when
`gmm_basket_model.pkl` exists — which now means only when a run was done with
`pipeline_main.py --with-gmm`. Note the kNN vote excludes `UNCLUSTERED` (-1)
reference baskets: "-1" is the ABSENCE of a need-state, and letting a new
basket be voted into it would invent a need-state that every unplaced basket
appears to belong to.

---

## 8. One picture, end to end

```
 cltv_hh_metrics_tpnb_base          product.product
 (hh × tpnb × week)                 (attributes, hierarchy)
        │                                   │
        │ SQL export                        │ SQL export (TPNA grain)
        ▼                                   ▼
 ns_household_tpnb_week_agg_train    ns_item_lookup_tpna
        │                             + ns_tpnb_to_tpna_mapping
        │                                   │
        │                                   ▼  MiniLM, brand excluded
        │                            384-dim vector per tpnb
        │                                   │
        │ GROUP BY hh, week                 │
        ▼                                   │
 ┌─────────────────┐                        │
 │ BASKET          │◄───────────────────────┘
 │ hh × week       │
 │ ≥2 products     │
 └────────┬────────┘
          │  ┌──── global co-purchase matrix (XᵀX over ALL baskets)
          │  │     + global K-means sub-clusters over WHOLE catalog
          ▼  ▼
 ┌──────────────────────────────────────────┐
 │  GRAPH A  — nodes = PRODUCTS (387 feats) │   one per basket
 │             edges = top-10 co-purchase    │   households absent
 └────────────────────┬─────────────────────┘
                      │  BasketGNN: 2×GINEConv → mean-pool → proj
                      │  trained as an AUTOENCODER (no labels)
                      ▼
            ● 64-dim point per basket  ──── THE EMBEDDING SPACE
                      │
                      │  k=10 cosine, one-directional
                      ▼
 ┌──────────────────────────────────────────┐
 │  GRAPH B  — nodes = BASKETS (hh × week)  │   ONE global graph
 │             edges = cosine similarity     │   ◄ households meet HERE
 └──────┬─────────────────────────┬─────────┘
        │ Leiden                  │ GMM (on the points, not the graph)
        ▼                         ▼
 need_state_cluster       need_state_cluster_gmm + gmm_confidence
        └────────────┬────────────┘
                     │  ARI comparison, both retained
                     ▼
        basket_need_state_clusters.parquet
                     │
                     ▼
        A need-state = a set of (household, week) pairs whose
        weekly shops are structurally similar. Read it as an
        OCCASION, not a customer segment — the same household
        can and will appear in several different need-states
        across different weeks.
                     │
                     │  STAGE 2.5 — need_state_graph.py
                     ▼
 ┌───────────────────────────────┬───────────────────────────────┐
 │  ADJACENCY (undirected)        │  TRANSITIONS (directed)       │
 │  contract Graph B to           │  sequence each household's    │
 │  need-state grain              │  own weeks                    │
 │                                 │                               │
 │  "these two occasions border   │  "after A, households go to   │
 │   each other"                   │   B next week, 40% of time"   │
 │                                 │                               │
 │  need_state_adjacency.parquet   │  need_state_transitions.      │
 │                                 │  parquet                      │
 └───────────────────────────────┴───────────────────────────────┘
        Similarity is NOT movement. Only the right-hand graph
        can answer "where does this household go next".
```

---

## 10. Stage 2.5 — the need-state graphs

Stage 2 produces need-state *labels*. Stage 2.5
([need_state_graph.py](src/need_state_graph.py)) produces the *edges between*
need-states, which earlier versions of the pipeline computed and discarded:
`cluster_basket_embeddings()` built the basket edge list, passed it to Leiden,
and let it go out of scope. `pipeline_main.py` now calls
`build_basket_knn_graph()` and `run_leiden_on_basket_graph()` as two steps so
the edge list survives.

| artifact | grain | labelling | answers |
|---|---|---|---|
| `need_state_adjacency.parquet` | need-state pair | Leiden | which need-states border each other |
| `need_state_transitions.parquet` | need-state pair, **directed** | Leiden | where households actually move |
| `need_state_gmm_overlap.parquet` | GMM component pair | **GMM** | soft overlap, as a cross-check |
| `basket_knn_edges.parquet` | basket pair | — | the specific neighbouring baskets behind any adjacency |

⚠ The GMM overlap table uses **GMM component ids**, which are a different
labelling from Leiden's `need_state_cluster`. Its columns are deliberately
named `gmm_component_a/b` so a join against the other two tables fails loudly
rather than silently returning nonsense. Relate the two through `basket_id`
in `basket_need_state_clusters.parquet`, which carries both.

⚠ `basket_knn_edges.parquet` is capped at 50M edges
(`BASKET_EDGE_WRITE_CAP`). At full population the kNN graph can exceed 1B
edges — hundreds of GB — so above the cap the write is refused with a message
rather than filling the disk. The need-state-grain tables are always tiny.

**Answering "where can this household go next":**

```python
import pandas as pd, need_state_graph as nsg
clusters = pd.read_parquet("../data/output/basket_need_state_clusters.parquet")
trans    = pd.read_parquet("../data/output/need_state_transitions.parquet")

nsg.journeys_for_household(clusters, trans, household_number=4213)
# -> current_need_state, as_of_week, history, next_steps, journeys
```

`history` is the household's observed path; `journeys` is the projection.
The projection is first-order Markov — it assumes where a household goes next
depends only on where it is now. That assumption is not tested anywhere, and
with a ~8-week window it cannot be tested well. Treat multi-step paths as
plausible routes, not forecasts, and check `low_support_steps` first.

---

## 9. Footnotes on things that are easy to misread

1. **`distinctiveness` is arguably named backwards.** `build_product_subclusters`
   computes `1 − d(own centroid)/d(farthest centroid)`, so a **high** value
   means a product sits **close** to its own centroid — i.e. typical of its
   sub-cluster, not distinctive from it. Worth knowing before interpreting
   that feature.

2. **Every 2-item basket gets `cp_score = 1.0` for both nodes.** With n=2
   there's one pair, so both raw scores tie, and `_minmax()` maps
   "all-equal-and-positive" to 1.0 (`GraphBuilder._minmax`). Don't read
   `cp_score` as an absolute popularity signal — it's within-basket-relative
   only.

3. **`GMM_N_COMPONENTS = 30` is still a placeholder**, explicitly labelled as
   such in the code, and Stage 2b is off by default because of it
   (`pipeline_main.py --with-gmm`). `select_k_via_bic()` exists to inform the
   choice and deliberately does not auto-pick.

   **`LEIDEN_RESOLUTION = 1.5`** came from a measured sweep over the real
   57.1M-vertex graph, giving 356 need-states at modularity 0.4145. Below 1.0
   the graph collapses into one community; 1.0–3.0 is a stable plateau. Sweep
   upward, never downward. ⚠ That sweep ran on the pre-2026-09-30 embedding
   space (see the warning in §3), so **re-sweep before trusting 1.5 again** —
   the cliff is a property of a specific graph, not of Leiden.

4. **`orders` and `sales_inc_vat` are exported but never consumed** by any
   Python file (the export SQL). Need-states here are composition-driven
   only — no spend or frequency signal enters the model.

5. **No theme/category anywhere.** Product sub-clustering runs globally over
   the entire catalog with no pre-grouping — it is *derived* structure, not a
   supplied hierarchy label. Since 2026-09-30 the cluster **id** is not a node
   feature at all; only `distinctiveness`, measured against the same
   centroids, survives. See §3.

6. **The autoencoder target barely involves the graph.** The loss reconstructs
   the mean of a basket's node features, and 384 of 387 dimensions *are* the
   mean product-embedding vector — so MSE is dominated by it, while the two
   `GINEConv` layers mix in neighbour information the target does not contain.
   `baseline_mean_embedding.py` measures what that costs by clustering the
   plain mean product embedding with no GNN at all, through the identical
   kNN/Leiden/profiling path.

7. **Adjacency `lift` is a configuration-model null**, with within-need-state
   edge weight INCLUDED in the degree term (self-pairs are dropped from the
   output, not from `deg`). `export_need_states.py` computes the identical
   number in SQL so the parquet and the Excel workbook agree, and offers the
   cross-edges-only reading alongside it as `cross_lift`. Until 2026-09-30
   those two files reported different quantities under the name `lift`.
