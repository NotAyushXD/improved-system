# Basket-size banding: clustering need-states separately, traversing them together

*Design proposal, 2026-09-27. Every number below is measured on the real
57,115,804-basket population, not estimated.*

---

## 1. The problem this solves

A **basket is a household's entire week**, not a shopping trip. No source table
carries a transaction or visit identifier — `cltv_hh_metrics_tpnb_base` has
`week_number` as its finest time grain — so a week is the best the data allows.

That matters because a need-state is meant to be an **occasion**. A week is not
an occasion; it is a mixture of them. A household that shopped three times in a
week produces one 30-item basket blending a big shop, a top-up and a treat run.
Every such blend resembles every other blend, and no clustering algorithm can
separate them.

Clustering the whole population at once confirmed this:

| | all 57.1M baskets |
|---|---|
| Need-states found | 356 |
| Median max product lift | **1.95** |
| Need-states with lift > 3 | **44 of 284 (15%)** |
| Products/basket in the 15 largest | 29–34 |

Lift measures how much more often a product appears in a need-state than in
shopping generally. **Lift ≈ 1 means a cluster holds a representative sample of
products — a region of space, not a shopping occasion.** The 15 largest
need-states topped out at lift 1.7, and the same products (courgettes, bottled
water) were "most distinctive" for many different clusters at once.

The clusters were geometrically tidy and semantically empty.

### The measurement that pointed the way

Distinctiveness turned out to be a direct function of basket size —
**Spearman −0.829**, monotonic across every band:

| avg products/basket | need-states | median max lift | baskets |
|---|---|---|---|
| ≤10 | 65 | **3.22** | 8.5M |
| 10–15 | 86 | 2.15 | 9.8M |
| 15–20 | 34 | 1.82 | 6.1M |
| 20–25 | 37 | 1.68 | 9.0M |
| 25–30 | 26 | 1.62 | 8.0M |
| 30+ | 36 | 1.50 | 15.7M |

Small weeks are single occasions and cluster sharply. Large weeks are blends
and cannot.

### Confirmation

Re-clustering only the 23,341,615 baskets with ≤10 products:

| | all baskets | ≤10 items |
|---|---|---|
| Median max lift | 1.95 | **5.03** |
| Characterised (lift > 3) | 15% | **97%** |
| Need-states above lift 10 | ~0 | **58** |
| Near-duplicate clusters | — | **0** |

The occasions that emerged are unmistakable: a **meal deal** (sandwich +
smoothie + crisps) at lift 15–21, a **beer run** (Heineken / cider / Peroni), a
**flower purchase** (tulips, roses), **lunch-on-the-go** variants. These are the
*largest* clusters in that band, not obscure ones.

---

## 2. How this idea emerged

Recorded because the design was not reasoned from theory. It came out of a
negative result that was very nearly accepted as final.

**The pipeline finished and looked healthy.** 356 need-states, modularity
0.4145, no embedding collapse, every basket labelled, all checks passing.
Modularity was read at the time as evidence of real community structure.

**But nobody knew what any need-state *was*.** They were integers. Product-lift
profiling was built to answer that — for each cluster, which products are
over-represented relative to the population.

**The first profile was a shock.** All fifteen of the largest need-states topped
out at lift 1.3–1.7, and the same handful of products — loose courgettes,
bottled water, a "pendrive" — appeared as the "most distinctive" item for many
different clusters at once. The partition was tidy and meaningless.

That also invalidated the modularity reading. **A kNN graph is locally connected
by construction**, so Leiden returns neat modular partitions over structureless
data too. Modularity had measured geometry, not meaning. Only lift could tell
those apart, and it did.

**The near-miss.** The obvious conclusion was that clustering had failed, that
the week grain made need-states impossible, and that the problem was in the
data rather than the method. That conclusion was wrong, and a single extra
query caught it.

The fifteen largest need-states are only about 1% of baskets each. Ranking
*all 284* by their maximum lift told a completely different story: **44 exceeded
lift 3**, the best reaching 9 — meal deals, beer runs, flower purchases. They
had been invisible because they were small, and the first view had been sorted
by size.

Had the investigation stopped at the largest-fifteen view, the work would have
been written off as a failure.

**The pattern hiding in those 44.** Every distinctive need-state averaged
6.4–11.3 products per basket. Every undistinctive one averaged 29–34. Quantified
across all need-states: **Spearman −0.829**, monotonic through every size band
(the table in §1).

**The test.** Cluster only the 23.3M baskets with ≤10 products. Median max lift
went 1.95 → 5.03; characterised need-states went 15% → 97%.

**The objection that produced this design.** Clustering bands separately means
their label spaces never connect — so a household moving between a top-up week
and a big-shop week would have no representable transition, and 59% of baskets
would drop out of the journey graph. The stratified-clustering-with-shared-
traversal design in §3 is the answer to that objection.

### What to take from this

Three things worth carrying into the next iteration:

1. **A clustering algorithm always returns clusters.** Ask Leiden for groups and
   it gives you groups, on noise as readily as on structure. Internal metrics
   like modularity do not test whether a grouping *means* anything.
2. **Sort by the thing you are measuring, not by size.** The evidence that
   rescued this was one query away the whole time, behind a default ordering.
3. **A negative result is a finding, not an endpoint** — but only if you
   interrogate it before acting on it.

---

## 3. The proposal

Split baskets into 3–4 populations by size. **Cluster each separately.**
Traverse between them using household week sequences, which exist independently
of any clustering.

```
Household 12345
  week 202608    4 items  → band S  → need-state S34   (meal deal)
  week 202609   38 items  → band XL → need-state L12   (full weekly shop)
  week 202610    7 items  → band S  → need-state S91   (beer run)
```

Three bands, three label spaces, one chain.

### Why the traversal works

`basket_id` is `<household_number>_<year_week>`. The household-week sequence is
recoverable from the id alone and does not depend on which band a week fell
into. `need_state_graph.build_need_state_transitions()` consumes
`(basket_id, need_state)` pairs — union the band label spaces into a single
column with distinct id ranges and it runs unchanged.

This is **stratified clustering with a shared temporal layer**. Each band gets a
model suited to its structure; the sequence stitches them together.

### What is and isn't shared

| | shared across bands? |
|---|---|
| **Embedding space** | **Yes.** All 57.1M baskets were embedded in one pass by the same GNN. A 5-item and a 40-item basket occupy the same 64-dim space and their distance is meaningful. |
| **kNN graph edges** | **No.** Restricting to a band before building the graph means a basket can only find neighbours inside its own band. |
| **Leiden communities** | **No.** Leiden reads only the graph, so a community can never span bands. |

The disconnection is *deliberate*. Letting a meal-deal basket link to 30-item
weeks is exactly what dragged it into the generic mass and produced lift 1.5.

---

## 4. Proposed bands

Three bands were run on 2026-09-28. **All results below are measured, not
predicted.**

| band | items | baskets | share | communities | profiled | **median max lift** | **lift > 3** | thin profiles |
|---|---|---|---|---|---|---|---|---|
| *baseline* | *all* | *57,115,804* | *100%* | *356* | *284* | *1.949* | *15%* | *—* |
| **S** | ≤10 | 23,341,615 | 40.9% | 2,090 | 225 | **5.032** | **97%** | **90%** |
| **M** | 11–20 | 12,138,111 | 21.3% | 98 | 92 | **3.009** | **53%** | 8% |
| **L** | 21+ | 21,636,078 | 37.9% | 78 | 78 | **1.991** | **4%** | 0% |

The gradient is monotonic and steep. Predictions made *before* the runs — M at
2.0–3.0 and L at 1.5–1.8 — both held.

### Band S — the occasion vocabulary

Sharpest by a wide margin: 97% of its need-states are characterised, 58 exceed
lift 10, and the top clusters reach lift 21. These are the meal deals, beer
runs, flower purchases and lunch-on-the-go trips.

**Caveat: 90% of S's 2,090 communities fall below the profiling support
threshold** (median size 45 baskets). Only ~201 carry a full profile. **Report
~201, never 2,090.**

### Band M — two families, almost no noise

53% characterised with only an 8% thin tail — the cleanest band. It splits
into two clearly distinguishable groups:

- **Fresh food shops** (need-states 7, 13, 20, 28, 30, 35, 36, 51, 59, 74, 21)
  — cucumber, blueberries, tomatoes, apples, carrots, onions at lift 2.7–3.3
- **Snack and drinks shops** (52, 23, 39) — Fridge Raiders, Peperami,
  Powerade, energy drinks, Kopparberg

### Band L — not occasions, and not weekly-shop archetypes either

**1.991 against a 1.949 baseline.** Isolating 21.6M large baskets and
clustering them alone bought essentially nothing; only 3 of 78 need-states
exceed lift 3. This is the blend hypothesis confirmed, not a failure.

But L *is* finding something — just not what was expected. Its need-states group
**Cookstown** sausages, **Denny** pork, **Coleraine** cheddar, **Wilson's
Country** potatoes, **Connolly's** gammon, **Keelings** grapes and **Isle of Man
Creamery** milk: Northern Irish and Irish brands clustering together.

**Band L is capturing region and store assortment, not shopping occasion.** A
full weekly shop reflects where someone lives and what their store stocks. That
is a legitimate segmentation — label it as *regional / store weekly-shop
archetypes*. Calling it a need-state vocabulary would be wrong.

### Should band L be included at all?

Both ways have costs. Excluding it leaves 37.9% of baskets unlabelled, which
drops them out of the transition graph entirely. Including it means a third of
the "need-states" are not occasions.

**Recommendation: include it, explicitly marked as a different kind of state.**
A journey reading *"regional weekly shop → meal deal → beer run"* is honest and
useful. One implying all three are comparable occasion types is not.

### Underlying size distribution

≤5 22.4% · 6–10 18.4% · 11–15 12.2% · 16–20 9.0% · 21–30 12.9% · 31–50 14.9% ·
51+ 10.0%. Maximum observed: 2,037 distinct products in one household-week —
that record is not a household, and the 51+ range deserves inspection before
anyone draws conclusions from it.

---

## 5. The transition layer

### Split within-band from across-band

These mean different things and must not be averaged:

- **Within-band** (`S34 → S91`) — a genuine occasion-to-occasion move. *"After
  a meal deal, households buy beer."*
- **Across-band** (`S34 → L12`) — mostly encodes a change in shopping *mode*.
  The band change is the signal, not the need-state pair. *"After a top-up,
  households do a big shop."*

Both are interesting. Reported as one table, the obvious one buries the
interesting one.

### Sparsity

Four bands at ~200 need-states each gives ~800 states and ~640,000 directed
pairs, against roughly 35.2M household-week steps. The current 356-state run
observes 86,097 pairs; this will be several times thinner per cell and
`MIN_TRANSITION_SUPPORT=30` will flag considerably more. Workable, but the long
tail becomes noise — filter before reading.

### Band boundaries are discontinuous

A 15-item week and a 16-item week land in different label spaces entirely. Some
households will flip bands over a single extra item. Nothing to fix; just don't
over-read a transition that is really a boundary artifact.

---

## 6. Cross-band linking (the refinement)

The same occasion will appear in several bands — a meal deal in S, and a
meal-deal-flavoured corner of M — as unrelated need-states with different ids.

After profiling each band, compute **similarity between need-state product
profiles across bands** (Jaccard over top-N high-lift products, the same
measure used to check within-band redundancy). Where an S need-state and an M
need-state share their markers, link or merge them.

That turns four disconnected vocabularies into one vocabulary with size
variants, and makes "the same occasion at different scales" visible rather than
duplicated. Cost is trivial — a few hundred × few hundred comparison on data
already produced.

---

## 7. What this design does and does not license

| Claim | Valid |
|---|---|
| "These are the occasions found in ≤5-item weeks" | ✓ |
| "S34 and L12 are different need-states" | ✓ — different label spaces by construction |
| "This household moved from a meal deal to a big shop" | ✓ — the sequence is band-independent |
| "This 30-item week is 30% meal-deal by product content" | ✓ via product-profile scoring |
| "This 30-item week **is** the meal-deal need-state" | ✗ — it is a mixture; forcing one label is the error that produced lift 1.5 |
| "We found 800 need-states" | ✗ — most are small. Report only well-supported ones |

---

## 8. Known limitations

**The grain is the root cause and this is a workaround.** Basket size is a
proxy for trip count, not a measurement of it. If anyone can obtain a
transaction or visit identifier from the warehouse, that single change would do
more than everything described here. Worth asking whoever owns
`product.product` and `cltv_hh_metrics_tpnb_base` before investing further.

**A longer aggregation window makes this worse, not better.** Biweekly or
monthly baskets blend *more* occasions per row. The direction that helps is
finer, and finer is unavailable.

**The observation window is thin.** 21.9M households across 57.1M baskets is
**2.6 observed weeks per household**. Most contribute one or two transitions.
`PIPELINE_TRANSITION_MAX_WEEK_GAP` is currently `none`, which counts a 1-week
and a 6-week step identically — check the `avg_week_gap` column on every
transition row before quoting a journey number. Widening the ~8-week SQL window
is the proper fix.

**Small bands lean toward lighter shoppers.** With only 2.6 weeks observed per
household, "small weeks" does not cleanly mean "the light weeks of every
household". State this as a selection effect rather than hiding it.

**Product descriptions are not fully trustworthy.** `tpnb 54739758` appears in
1,767,393 baskets (3.1%) described as "PENDRIVE FLASH DRIVE 12GB SLIQ" with
department "FRESH FRUIT/VEG/SALAD". It maps to exactly one tpna, so the
contradiction is in the source extract. Descriptions also feed the product
embeddings, so corruption affects the clustering and not just the labels. Run
`src/audit_product_data.py` before trusting any label.

---

## 9. Implementation

1. **`--min-products` alongside `--max-products`** in
   `cluster_basket_embeddings.py --build-edges`, so any band is one command.
   Cache filenames already encode the subset, so bands cannot overwrite each
   other.
2. **Cluster each band** with `cluster_leiden_networkit.py --edges <band> --min-coverage 0`.
   Coverage is measured against the full population, so a band reports its own
   share — that is the check working on the wrong question, not a failure.
3. **Profile each band** with `profile_need_states.py`. Outputs are named after
   the label file, so bands accumulate rather than overwrite.
4. **Score each band** with `evaluate_run.py`, which appends to
   `experiment_log.csv`.
5. **Union the label spaces** into one `need_state_cluster` column with
   distinct id ranges plus a `band` column.
6. **Cross-band profile similarity** to link equivalent occasions.
7. **Split transitions** into within-band and across-band tables.

Steps 1, 5, 6 and 7 are not yet built. Steps 2–4 exist.

---

## 10. How to judge whether a band worked

Use **`median_max_lift`** and **`pct_lift_over_3`** from `experiment_log.csv`.

**Do not use modularity.** A kNN graph is locally connected by construction, so
Leiden returns tidy modular partitions over structureless data too. Modularity
0.4145 was read as evidence of real structure in the full-population run; the
lift analysis later showed those clusters were meaningless. Modularity cannot
distinguish a real grouping from a geometric one.

Use **`median_twin_jaccard`** to catch the failure lift cannot see: one
occasion split many ways. Lift compares a cluster to the *population*; only
this compares clusters to *each other*.

Reference points from the two completed runs:

| run | median max lift | lift > 3 | twin Jaccard |
|---|---|---|---|
| all 57.1M baskets | 1.95 | 15% | not measured |
| ≤10 items | **5.03** | **97%** | 0.429, 0 identical |

A new band that lands near 1.95 has not worked. Near or above 5.03 has.
