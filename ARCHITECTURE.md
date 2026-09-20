# End-to-End Architecture: Input Data → Graph Nodes → Need-States

This document traces the full path from raw warehouse data to need-state
clusters, with a focus on **how each basket becomes a graph, what a "node"
actually is, and exactly what's packed inside it**.

---

## 1. Big picture

```
 WAREHOUSE TABLES                    SQL EXPORTS                  PYTHON PIPELINE
 ────────────────                    ───────────                  ───────────────

 product.product            ──┐
 (attributes, hierarchy)      │
                               ├──► ns_item_lookup_tpna.sql ──┐
 product.product              │                                ├──► build_product_embeddings.py
 (tpnb ↔ tpna mapping)   ──────┴──► ns_tpnb_to_tpna_mapping.sql ┘        │
                                                                          ▼
                                                      data/output/product_embeddings.parquet
                                                          (one 384-dim vector per tpnb)

 lab_customer_value_analytics
 .cltv_hh_metrics_tpnb_base    ──► ns_household_tpnb_week_agg_train.sql
 (household × tpnb × week)                    │
                                               ▼
                                    pipeline_main.py Stage 0/1
                                    (see §2 and §3 below)
                                               │
                                               ▼
                          ┌────────────────────────────────────┐
                          │  ONE GRAPH PER BASKET (§3, §4)      │
                          │  nodes = products, edges = co-buy   │
                          └────────────────────────────────────┘
                                               │
                                               ▼
                              GNN (BasketGNN, §5) → 64-dim basket embedding
                                               │
                                               ▼
                         Leiden (mutual-kNN + community detection)
                         AND
                         GMM (Gaussian Mixture)              (§6)
                                               │
                                               ▼
                     data/output/basket_need_state_clusters.parquet
                     (need_state_cluster, need_state_cluster_gmm per basket)
                                               │
                                               ▼
                              Stage 2.5 — need_state_graph.py  (§8)
                     adjacency (which need-states border each other)
                     transitions (where households actually go next)
```

Every parameter mentioned in this document is set in `.env` via
`src/config.py` — see PIPELINE_GUIDE.md §2b. The values quoted here are the
defaults.

---

## 2. Input data → product embeddings (one vector per product)

`build_product_embeddings.py` runs once, before anything else:

```
tpna="TPNA123"                                                    all-MiniLM-L6-v2
  description: "Semi Skimmed Milk 2L"        ┐                    sentence-transformer
  department:  "Dairy"                        ├─► text string ──► embeds ──► [384 floats]
  class:       "Milk"                         │   "Semi Skimmed
  subclass:    "Fresh Milk"                    ┘    Milk 2L | Dairy | Milk | Fresh Milk"

  (brand deliberately excluded — found to dominate similarity for short,
   generic product text, causing clustering by brand instead of by what
   the product actually is)
```

Embeddings are built once at **TPNA** grain (style/archetype — one entry per
"Semi Skimmed Milk 2L" regardless of pack count or promo variant), then
**broadcast down** to every sibling `tpnb`: a 2-pint and 4-pint version of
the same milk get an *identical* 384-dim vector. Result:
`data/output/product_embeddings.parquet` — one row per `tpnb`, column
`embedding` (384 floats).

**More examples** (illustrative — made up to show the pattern, not real
warehouse values):

| tpna | description embedded | department / class / subclass | resulting embedding |
|---|---|---|---|
| `TPNA123` | "Semi Skimmed Milk 2L \| Dairy \| Milk \| Fresh Milk" | Dairy / Milk / Fresh Milk | `[0.021, -0.114, ..., 0.077]` (384 floats) |
| `TPNA456` | "Baked Beans 415g \| Grocery \| Tinned Vegetables \| Beans" | Grocery / Tinned Vegetables / Beans | `[0.183, 0.042, ..., -0.055]` (384 floats, far from `TPNA123`'s) |
| `TPNA789` | "White Sliced Bread 800g \| Bakery \| Bread \| Sliced Bread" | Bakery / Bread / Sliced Bread | `[0.099, -0.201, ..., 0.014]` (384 floats) |

**TPNA → TPNB broadcast** — one style, many barcodes, one shared vector:

| tpnb | tpna | embedding |
|---|---|---|
| `100234` (2-pint carton) | `TPNA123` | *same 384 floats as below* |
| `100567` (2-pint bottle, different packaging) | `TPNA123` | *same 384 floats as above* |
| `100812` (4-pint bottle) | `TPNA123` | *same 384 floats as above* |

**Why excluding brand matters** — two products with almost identical
generic text but different brands land in nearly the same place, which is
the point (brand differences shouldn't drive product similarity here):

| tpnb | text actually embedded (brand stripped) | embedding |
|---|---|---|
| Tesco Baked Beans 415g | "Baked Beans 415g \| Grocery \| Tinned Vegetables \| Beans" | near-identical to the row below |
| Heinz Baked Beans 415g | "Baked Beans 415g \| Grocery \| Tinned Vegetables \| Beans" | near-identical to the row above |

If brand text were included instead, these two would likely separate into
different regions of embedding space (and different global sub-clusters,
§3) purely because "Tesco" and "Heinz" are different tokens — even though
they're functionally the same product to a shopper.

---

## 3. Input data → baskets (one row per household × week)

`pipeline_main.py` Stage 1, reading `ns_household_tpnb_week_agg_train`:

```
household_number=4213, week 202615:
  tpnb=MILK,  quantity=2
  tpnb=BREAD, quantity=1
  tpnb=EGGS,  quantity=1
  tpnb=BUTTER,quantity=1
        │
        │  .groupby(["household_number", "year_week_number"])
        ▼
basket_id = "4213_202615"
  products = [MILK, BREAD, EGGS, BUTTER]
  units    = [2, 1, 1, 1]
```

Baskets with fewer than 2 products are dropped. This is basket grain as it
exists in this pipeline — see `data/TABLE_REFERENCE.md` for why WEEK is the
finest grain available (no true single-visit/transaction ID exists in any
source table).

**More examples** (illustrative — showing the range of basket sizes that
week grain actually produces):

```
household 7788, week 202612  →  basket "7788_202612"          (small — a top-up shop)
  products = [MILK, WINE]
  units    = [1, 1]

household 9021, week 202613  →  basket "9021_202613"          (medium — "pasta night")
  products = [PASTA, TOMATO_SAUCE, PARMESAN, GARLIC_BREAD, RED_WINE]
  units    = [1, 2, 1, 1, 1]

household 5540, week 202614  →  basket "5540_202614"          (large — a family's big weekly shop)
  products = [MILK, BREAD, EGGS, BUTTER, CHEESE, NAPPIES, WIPES, KIDS_SNACKS,
              CEREAL, PASTA, MINCE, CHICKEN, FROZEN_PEAS, YOGHURT, BANANAS,
              APPLES, TOILET_ROLL, DISH_SOAP, LAUNDRY_POD, DOG_FOOD,
              KETCHUP, ORANGE_JUICE]                            (22 items)
```

Alongside this, `pipeline_main.py` also builds the **global co-purchase
matrix** — a product×product table of "how many baskets, across the WHOLE
population, contained both product A and product B" — and
`GraphBuilder.prepare_globals()` runs one **global K-means clustering pass**
over every product's embedding (whole catalog, no category pre-grouping —
picks `k` by silhouette score from `SUBCL_K_CANDIDATES`). Both of these feed
node construction below.

**Example co-purchase counts** (illustrative — a tiny slice of the full
`n_products × n_products` matrix, restricted to the products in the first
basket above):

| product pair | co-purchase count (across ALL baskets, population-wide) |
|---|---|
| MILK ↔ BREAD | 1,200,000 |
| MILK ↔ EGGS | 450,000 |
| MILK ↔ BUTTER | 600,000 |
| BREAD ↔ EGGS | 280,000 |
| BREAD ↔ BUTTER | 700,000 |
| EGGS ↔ BUTTER | 200,000 |

These raw counts are what §4's per-basket co-purchase score and edges are
computed from — always restricted to just the products actually present in
one basket at a time, never the full catalog-wide matrix at once.

---

## 4. THE KEY PART: how one basket becomes a graph

This is `GraphBuilder.build_one_graph()` — called once per basket, identically
during training and scoring. Take the basket above: `[MILK, BREAD, EGGS, BUTTER]`.

```
                         GRAPH FOR BASKET "4213_202615"

                    ┌─────────┐              ┌─────────┐
                    │  MILK   │◄────────────►│  BREAD  │
                    └────┬────┘              └────┬────┘
                         │      ╲            ╱     │
                         │       ╲          ╱      │
                         │        ╲        ╱       │
                    ┌────▼────┐    ╲      ╱   ┌────▼────┐
                    │  EGGS   │◄────╳────╳───►│ BUTTER  │
                    └─────────┘    ╱      ╲    └─────────┘

   NODES  = the 4 distinct products in this basket
   EDGES  = each product's top-10 strongest co-purchase partners
            *within this basket* (fewer than 10 here, since the
            basket itself only has 3 other products)
```

**One NODE = one product, as it appears in this specific basket.** Its
feature vector (`in_dim = 384 + 4 = 388` numbers) is:

```
NODE "MILK" (inside basket 4213_202615)
┌──────────────────────────────────────────────────────────────────────┐
│ [0:384]  product embedding            → MILK's 384-dim text vector    │
│                                          (same for every MILK anywhere,│
│                                           from §2 — identical across   │
│                                           all baskets)                 │
│ [384]    co-purchase score            → how strongly MILK co-occurs   │
│                                          with EGGS/BREAD/BUTTER        │
│                                          *specifically*, taken from    │
│                                          the global co-purchase matrix,│
│                                          normalised 0–1 within THIS    │
│                                          basket                        │
│ [385]    sub-cluster id (0–1)         → which of the K global product │
│                                          clusters MILK belongs to      │
│                                          (same for every MILK anywhere)│
│ [386]    distinctiveness (0–1)        → how far MILK sits from its    │
│                                          cluster centroid vs. the      │
│                                          farthest centroid — same for  │
│                                          every MILK anywhere           │
│ [387]    log(units)                   → log1p(2) — MILK-SPECIFIC to   │
│                                          THIS basket (bought 2 here)   │
└──────────────────────────────────────────────────────────────────────┘
```

Three of these five components (embedding, sub-cluster id, distinctiveness)
are **product-level constants** — the same for every MILK node in every
basket anywhere. Two (co-purchase score, log-units) are **basket-specific**
— they depend on what else is in *this* basket and how much was bought
*here*. That mix is deliberate: the GNN needs both "what kind of product is
this, globally" and "how is it behaving in this particular basket."

**Worked example — all 4 nodes, using §3's illustrative co-purchase counts**
(K=200 global sub-clusters assumed; distinctiveness values are illustrative
since they depend on the real embedding geometry):

| step | MILK | BREAD | EGGS | BUTTER |
|---|---|---|---|---|
| co-purchase partners (this basket) | BREAD 1.2M, EGGS 450K, BUTTER 600K | MILK 1.2M, EGGS 280K, BUTTER 700K | MILK 450K, BREAD 280K, BUTTER 200K | MILK 600K, BREAD 700K, EGGS 200K |
| row sum ÷ (n−1=3) | 750,000 | 726,667 | 310,000 | 500,000 |
| log1p(·) | 13.53 | 13.50 | 12.64 | 13.12 |
| **[384] cp_score** (min-max over these 4) | **1.000** | **0.966** | **0.000** | **0.539** |
| **[385] sub-cluster id** (cluster ÷ 199) | 0.070 (cluster 14 — dairy) | 0.236 (cluster 47 — bakery) | 0.106 (cluster 21) | 0.070 (cluster 14 — dairy, same as MILK) |
| **[386] distinctiveness** | 0.71 | 0.58 | 0.83 | 0.65 |
| **[387] log(units)** | log1p(2) = 1.099 | log1p(1) = 0.693 | log1p(1) = 0.693 | log1p(1) = 0.693 |

Note MILK and BUTTER land in the *same* sub-cluster (both dairy, by
embedding similarity) even though they were never told that label —
that's the global K-means from §3 doing its job.

**EDGES** carry 2 features each, built purely from co-purchase counts (no
theme/category signal anywhere):
```
edge (MILK → BREAD):
  [0] log(1 + co-purchase count between MILK and BREAD, across ALL baskets)
  [1] that count, as a fraction of MILK's single strongest co-purchase
      partner's count (so every node's edges are self-relative — a
      "how important is this link to THIS product" signal)
```
Only the top `TOP_K=10` strongest partners per node are kept as edges (a
basket with ≤10 other products, like this one, just keeps all of them).

**Worked example — MILK's 3 outgoing edges** (MILK's strongest partner is
BREAD at 1.2M, so that's the denominator for MILK's relative-strength column):

| edge | log1p(count) | relative strength (÷ MILK's max = 1.2M) |
|---|---|---|
| MILK → BREAD | log1p(1,200,000) = 13.99 | 1,200,000 / 1,200,000 = **1.00** |
| MILK → BUTTER | log1p(600,000) = 13.30 | 600,000 / 1,200,000 = **0.50** |
| MILK → EGGS | log1p(450,000) = 13.02 | 450,000 / 1,200,000 = **0.375** |

**Same product, different basket — what actually changes.** Take the small
basket from §3, household 7788: `[MILK, WINE]`, units `[1, 1]`, with
MILK↔WINE co-purchased in only 60,000 baskets population-wide (much rarer
than MILK↔BREAD):

| MILK's feature | in basket `4213_202615` (with BREAD/EGGS/BUTTER) | in basket `7788_202612` (with WINE only) |
|---|---|---|
| [0:384] embedding | *identical* 384 floats | *identical* 384 floats |
| [384] cp_score | 1.000 | **1.000** (see note below) |
| [385] sub-cluster id | 0.070 | *identical* — 0.070 |
| [386] distinctiveness | 0.71 | *identical* — 0.71 |
| [387] log(units) | log1p(2) = 1.099 | log1p(1) = 0.693 |

*Note on the tie*: with only 2 products in a basket, there's exactly one
co-purchase pair to compute from, so MILK and WINE's raw scores are
necessarily equal before normalizing — and `_minmax()` maps "everything
equal and positive" to `1.0` for both, rather than dividing by a zero range.
So every 2-item basket's two nodes get `cp_score = 1.0` regardless of how
common or rare that specific pair actually is population-wide — a real
edge-case behavior of the current normalization, not a bug, but worth
knowing about if a downstream analysis ever reads `cp_score` as an absolute
popularity signal rather than a within-basket-relative one.

The resulting object per basket is a PyTorch Geometric `Data`:
`x` (shape `[n_products_in_basket, 388]`), `edge_index` (shape `[2, n_edges]`),
`edge_attr` (shape `[n_edges, 2]`). Building this small, basket-scoped
structure — rather than one giant matrix over the whole catalog — is exactly
what keeps this step's memory bounded regardless of catalog size (see the
memory-fix history in `GPT_CONTEXT_PROMPT.md`).

---

## 5. Graph → basket embedding (the GNN itself)

`GNN_Train.BasketGNN.encode()` runs the SAME path for every basket, whether
it's a training basket or a brand-new one being scored later:

```
 x [n, 388] ──► node_encoder (Linear 388→128) ──► ReLU
              │
              ▼
      GINEConv #1 (128→128, uses edge_attr) ──► ReLU ──► Dropout
              │
              ▼
      GINEConv #2 (128→128, uses edge_attr) ──► ReLU
              │
              ▼
      global_mean_pool  (average ALL node vectors in this basket → ONE vector)
              │
              ▼
      proj (Linear 128→128 → ReLU → Linear 128→64)
              │
              ▼
        ONE 64-DIM VECTOR representing the whole basket
```

The two `GINEConv` layers let each product's representation absorb signal
from its co-purchase neighbors (MILK's vector shifts based on BREAD/EGGS/
BUTTER being nearby); `global_mean_pool` then collapses all of a basket's
(now neighbor-aware) node vectors into one fixed-size basket vector. This
64-dim vector — not the raw products list — is what gets clustered next.

**Worked example — the pooling step, in miniature.** Real vectors are
128-dim at this point, which isn't something to eyeball — but the mechanic
`global_mean_pool` performs is just an elementwise average. Shrinking to a
toy 4-dim vector per node to make that concrete, for basket `4213_202615`'s
4 nodes *after* both GINEConv layers have run (illustrative numbers only —
actual values depend on trained weights):

| node | (toy) post-conv vector |
|---|---|
| MILK | `[0.42, -0.18, 0.91, 0.03]` |
| BREAD | `[0.38, -0.22, 0.85, 0.11]` |
| EGGS | `[-0.05, 0.61, 0.14, -0.29]` |
| BUTTER | `[0.30, -0.09, 0.77, 0.08]` |
| **mean** (→ this basket's pooled vector) | `[0.263, 0.030, 0.668, -0.018]` |

Notice MILK, BREAD, and BUTTER (the mutually strongly co-purchased trio,
per §4's edge weights) ended up with *similar* post-conv vectors, while
EGGS — MILK's weakest link in this basket — sits further away. That's the
message-passing step doing its job: nodes that are strongly connected pull
each other's representations closer together before pooling. The mean of
all 4 is what `proj` then maps down to the final 64-dim basket embedding —
in a real basket, one lone dissimilar item doesn't get to dominate the
average the way it would in a 2-item basket.

During training, a decoder (`64→128→388`) tries to reconstruct the basket's
own mean node features from this 64-dim vector, and the reconstruction error
is the training loss (a graph autoencoder) — there's no need-state label to
predict; the model just learns to compress basket structure faithfully. A
basket whose products are all wildly dissimilar (nothing to compress
losslessly) will tend to produce a higher reconstruction error than one
where message-passing already made the products look alike — which is
exactly the pressure that shapes the 64-dim embedding space Leiden and GMM
cluster over next.

---

## 6. Basket embeddings → need-states

Every basket now has one 64-dim vector. Two independent clustering passes
run over these vectors (not over products, not over raw baskets):

```
                    64-dim basket embeddings
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
   mutual-kNN graph (k=15,             GaussianMixture
   cosine similarity)                  (n_components=30,
              │                        diagonal covariance)
              ▼                            │
   Leiden community detection              ▼
   (leidenalg, resolution=1.0)        need_state_cluster_gmm
              │                        + gmm_confidence
              ▼                            │
   need_state_cluster                      │
              └─────────────┬──────────────┘
                             ▼
              basket_need_state_clusters.parquet
        (both label sets kept side by side — compared via
         Adjusted Rand Index, neither discarded)
```

A basket's `need_state_cluster` is therefore a group of baskets whose
64-dim GNN embeddings sit close together — which, by construction, means
their products, co-purchase patterns, and relative quantities were similar,
without any product category ever being consulted.

**Worked example — 6 baskets through both methods** (illustrative basket
embeddings/labels, to show the mechanics — not real cluster IDs):

| basket_id | rough contents | nearest neighbor (cosine) |
|---|---|---|
| `4213_202615` | milk, bread, eggs, butter | `9954_202611` (sim 0.94) — another dairy/breakfast basket |
| `9954_202611` | milk, bread, eggs, cheese | `4213_202615` (sim 0.94) |
| `9021_202613` | pasta, tomato sauce, parmesan, garlic bread, red wine | `2207_202609` (sim 0.92) — another "pasta night" |
| `2207_202609` | pasta, pesto, parmesan, white wine | `9021_202613` (sim 0.92) |
| `7788_202612` | milk, wine | `9021_202613` (sim 0.58) — weak link, via wine |
| `5540_202614` | 22-item big weekly shop | `4213_202615` (sim 0.61) — overlaps on milk/bread/eggs, but pulled in many directions |

```
Leiden communities (mutual-kNN + community detection):
  community 5  = { 4213_202615, 9954_202611, 5540_202614 }   "dairy / breakfast basics"
  community 12 = { 9021_202613, 2207_202609, 7788_202612 }   "pasta night + wine-adjacent"
```

| basket_id | GMM: top components (illustrative, out of 30) | assigned | gmm_confidence |
|---|---|---|---|
| `4213_202615` | comp 5: 0.82, comp 12: 0.11, rest: 0.07 | 5 | 0.82 |
| `9021_202613` | comp 12: 0.91, comp 5: 0.05, rest: 0.04 | 12 | 0.91 |
| `7788_202612` | comp 12: 0.55, comp 5: 0.30, rest: 0.15 | 12 | **0.55** (low — sits between two need-states) |
| `5540_202614` | comp 5: 0.44, comp 9: 0.28, rest: 0.28 | 5 | **0.44** (low — a 22-item basket spans too many occasions to fit one cleanly) |

`compare_leiden_gmm()` would report something like **Adjusted Rand Index ≈
0.71** for this scenario — Leiden and GMM mostly agree (both group the
pasta/wine baskets together and the dairy baskets together), but aren't
identical, which is exactly why both are kept side by side rather than
picking one.

Two things worth noticing in this example, both realistic behaviors of the
actual code: (1) `7788_202612`'s low `gmm_confidence` (0.55) is the model
correctly flagging that a 2-item milk+wine basket is genuinely ambiguous
between "quick top-up" and "date night" — that's `cluster_confidence` /
`gmm_confidence` doing exactly what they're for; (2) the 22-item big-shop
basket (`5540_202614`) also gets a low, spread-out confidence, because a
basket covering many occasions at once is inherently harder for either
method to place cleanly into ONE need-state — a direct, visible consequence
of the basket-grain discussion in `data/TABLE_REFERENCE.md`.

---

## 7. Nuances that are easy to misread

Each of these is real behaviour of the current code, verified by a test in
`test_pipeline.py`. Several were bugs found the hard way.

**1. `distinctiveness` is arguably named backwards.** It is computed as
`1 − d(own centroid) / d(farthest centroid)`, so a **high** value means the
product sits **close** to its own sub-cluster centroid — i.e. *typical* of it,
not distinctive from it. Worth knowing before interpreting node feature
`[386]`.

**2. Every 2-item basket gets `cp_score = 1.0` for both nodes.** With n=2
there is exactly one co-purchase pair, so both raw scores tie, and `_minmax()`
maps "all equal and positive" to 1.0. Do not read `cp_score` as absolute
popularity — it is within-basket-relative only.

**3. Edge relative-strength was wrong for small baskets until
`GRAPH_BUILDER_VERSION 3`.** The denominator was taken as the first neighbour
in column order rather than the strongest link, which is only the same thing
when a node has more than `TOP_K` neighbours. Every basket of ≤11 products
was affected — with counts 30/90/60 it produced 1.0 / 3.0 / 2.0 instead of
0.33 / 1.0 / 0.67, violating the documented invariant that every value lies in
(0, 1] with exactly one equal to 1.0. The §4 worked example above describes
the *fixed* behaviour. Embeddings produced before this fix are not comparable
with those produced after.

**4. Zero and negative quantities are real, and become `log_units = 0.0`.**
The warehouse `quantity` column is a weekly SUM that folds in returns, so a
product bought and returned in the same week nets to 0 and a net refund goes
negative. `log1p` is undefined at or below −1, and those cases land on `0.0` —
the same value as "bought nothing". That is inherited behaviour, not a
deliberate modelling decision. If returns are a meaningful share of your data,
decide in SQL what should happen to them.

**5. `GMM_N_COMPONENTS = 30` and `LEIDEN_RESOLUTION = 1.0` are placeholders**,
explicitly labelled as such in the code. `sweep_resolution()` and
`select_k_via_bic()` exist to inform the choice and deliberately do not
auto-pick.

**6. `orders` and `sales_inc_vat` are exported but never consumed** by any
Python file. Need-states here are composition-driven only — no spend or
frequency signal enters the model.

**7. No theme or category anywhere.** Product sub-clustering (node feature
`[385]`) runs globally over the whole catalog with no pre-grouping. It is
*derived* structure, not a supplied hierarchy label.

**8. A low training loss does not mean the embeddings are useful.** The model
reconstructs the *mean* of each basket's node features. If those means vary
little across baskets, a near-constant output scores well — and every basket
lands in nearly the same place, making clustering meaningless while everything
appears to succeed. `test_pipeline.py --prod-outputs` checks for exactly this.

---

## 8. Stage 2.5 — how need-states relate to each other

Stages 1-6 above produce need-state *labels*. `need_state_graph.py` produces
the **edges between** need-states — which earlier versions of this pipeline
computed and discarded.

Two edge sets, for two different questions, and **they are not
interchangeable**:

```
ADJACENCY (undirected)          TRANSITIONS (directed)
contract the basket kNN graph   sequence each household's own weeks
        │                                │
        ▼                                ▼
"these two occasions border     "after A, households go to B
 each other; boundary baskets    next week, 40% of the time"
 are genuinely ambiguous"
        │                                │
        ▼                                ▼
use for: substitutable states   use for: JOURNEYS, next-best-action
```

Similarity is not movement. Households do not travel along adjacency edges.

A third table, `need_state_gmm_overlap.parquet`, recovers the full GMM
posterior (the code previously kept only its maximum, as `gmm_confidence`) as
a probabilistic cross-check on adjacency. Its columns are named
`gmm_component_a/b` rather than `need_state_a/b` on purpose: **GMM component
ids are a different labelling from Leiden community ids**, and sharing a
column name would invite a join that silently returns nonsense.

Deliberately not provided: centroid-to-centroid distance between need-states.
Leiden communities are arbitrary-shaped, so an elongated community's centroid
can land where no basket exists. The quotient graph measures the same
intuition using only points that really exist.

See PIPELINE_GUIDE.md §6b for the API and `HOUSEHOLD_GRAPH_FLOW.md` for how
households, baskets, products and need-states relate across both graphs.

---

## 9. Dimension cheat-sheet

| Quantity | Value | Where |
|---|---|---|
| Product embedding dim | 384 | `all-MiniLM-L6-v2`, `build_product_embeddings.py` |
| Extra node features | 4 (co-purchase score, sub-cluster id, distinctiveness, log-units) | `GraphBuilder.build_one_graph()` |
| Node feature width (`in_dim`) | 388 | `emb_dim + 4` |
| Edge feature width | 2 (log co-purchase count, relative strength) | `GraphBuilder._build_edges_numba()` |
| Edges kept per node | ≤ `TOP_K = 10` | `PIPELINE_TOP_K` in `.env` |
| GNN hidden width | 128 | `PIPELINE_HIDDEN_DIM` in `.env` |
| Basket embedding dim (`out_dim`) | 64 | `PIPELINE_OUT_DIM` in `.env` |
| Global product sub-clusters (K) | chosen from `[50, 100, 200, 400]` by silhouette | `PIPELINE_SUBCL_K_CANDIDATES` in `.env`. On the real catalog k=400 was selected — the ceiling of the range, with silhouette still rising, so the true optimum is likely higher |
| Leiden kNN neighbors | 15 | `PIPELINE_BASKET_KNN_K` in `.env` |
| GMM components | 30 (placeholder) | `PIPELINE_GMM_N_COMPONENTS` in `.env` |
