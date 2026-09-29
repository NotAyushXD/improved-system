# Basket-size banding: an investigation, and why it was not adopted

*Investigation record. Proposed 2026-09-27, tested 2026-09-28, concluded
2026-09-29. Every number is measured on the real 57,115,804-basket population.*

> [!IMPORTANT]
> **Verdict: banding does not improve need-states and is not recommended as a
> production segmentation.** The decisive test — what share of ≤10-item baskets
> land in a distinctive need-state — gives **19.4%** for the ordinary
> full-population clustering and **~20%** for a dedicated small-basket
> clustering. Identical within rounding.
>
> The idea was accepted for a day on the strength of a measurement that turned
> out to be wrong (§4). It is recorded in full because the reasoning that
> produced it was sound, the bug it exposed was serious, and the method built
> to settle it — permutation nulls (§5) — is now the project's standard
> validation and is worth more than the hypothesis was.

---

## 1. The question

A **basket is a household's entire week**, not a shopping trip. No source table
carries a transaction or visit identifier — `cltv_hh_metrics_tpnb_base` has
`week_number` as its finest grain — so a week is the best the data allows.

That matters because a need-state is meant to be an **occasion**. A week is not
an occasion; it is a mixture of them. A household that shopped three times
produces one 30-item basket blending a big shop, a top-up and a treat run.
Every such blend resembles every other blend.

**Hypothesis:** if large baskets are blends and small baskets are single
occasions, then clustering small baskets *separately* — so they cannot be
pulled toward the generic mass by kNN edges to 40-item weeks — should produce
sharper need-states.

This document records how that was tested and why it failed.

---

## 2. How the idea emerged

Kept because the design was not reasoned from theory. It came out of a negative
result that was very nearly accepted as final, and the near-miss is the most
transferable part of this whole exercise.

**The pipeline finished and looked healthy.** 356 need-states, modularity
0.4145, no embedding collapse, every basket labelled, all checks passing.
Modularity was read at the time as evidence of real community structure.

**But nobody knew what any need-state *was*.** They were integers. Product-lift
profiling was built to answer that.

**The first profile was a shock.** All fifteen of the largest need-states topped
out at lift 1.3–1.7, and the same handful of products — loose courgettes,
bottled water, a "pendrive" — appeared as the "most distinctive" item for many
different clusters at once. The partition was tidy and meaningless.

That also invalidated the modularity reading. **A kNN graph is locally connected
by construction**, so Leiden returns neat modular partitions over structureless
data too. Modularity measured geometry, not meaning.

**The near-miss.** The obvious conclusion was that clustering had failed and the
week grain made need-states impossible. That conclusion was wrong, and one
extra query caught it. The fifteen largest need-states are only ~1% of baskets
each. Ranking *all 284* by maximum lift told a different story: **44 exceeded
lift 3**. They had been invisible because they were small, and the first view
was sorted by size.

**The pattern hiding in those 44.** Every distinctive need-state averaged
6.4–11.3 products per basket; every undistinctive one averaged 29–34. Across all
need-states: **Spearman −0.829**.

That correlation is real and has been reproduced on the corrected data (§6). It
is the observation that motivated banding. **The observation survived; the
intervention built on it did not.**

### What to carry forward

1. **A clustering algorithm always returns clusters.** Ask Leiden for groups and
   it gives you groups, on noise as readily as on structure. Internal metrics
   like modularity do not test whether a grouping *means* anything.
2. **Sort by the thing you are measuring, not by size.** The evidence that
   rescued this was one query away the whole time, behind a default ordering.
3. **A negative result is a finding, not an endpoint** — but only if you
   interrogate it before acting on it.
4. **And the converse, learned the hard way in §4: a *positive* result is not an
   endpoint either.** The same interrogation is owed to results you like.

---

## 3. What was built and run

Three bands, clustered independently at the working configuration (k=10,
one-directional kNN, gamma 1.5), then profiled and scored:

| band | items | baskets | share |
|---|---|---|---|
| **S** | ≤10 | 23,341,615 | 40.9% |
| **M** | 11–20 | 12,138,111 | 21.3% |
| **L** | 21+ | 21,636,078 | 37.9% |

The three partition the population exactly
(23,341,615 + 12,138,111 + 21,636,078 = 57,115,804), which makes their coverage
figures summable — used in §6.

`--min-products` / `--max-products` on `cluster_basket_embeddings.py
--build-edges` restrict the graph to a band. Cache filenames encode the subset,
so bands cannot overwrite each other.

---

## 4. The bug that produced the wrong answer

The first round of results looked decisive:

| band | median max lift (as first measured) |
|---|---|
| baseline | 1.949 |
| S ≤10 | **5.032** |
| M 11–20 | 3.009 |
| L 21+ | 1.991 |

That was wrong. `profile_need_states.build_counts()` computed `product_baseline`
over the **whole** basket table with no cluster filter, while the numerator
covered only labelled baskets. **A run that labels a subset was scored against
the baskets it had deliberately excluded.**

Banding is precisely the case this breaks — every band labels a subset, so every
band was measured against a population it was not drawn from.

The correct property, now enforced by `check_lift_formula` in
`test_pipeline.py`: **a clustering that does nothing must score lift 1.0
everywhere.** Put all labelled baskets in one cluster and the fixed code returns
exactly 1.0. The old code did not — band S's "do nothing" baseline was already
above 1, because small baskets' product mix differs from the full population's.

### The correction, and why its size varies

Per product, the two measurements differ by exactly the product's *band-level*
lift:

```
lift_vs_all_shopping = lift_within_band × band_lift
```

So the inflation is proportional to how small a share of all **item-rows** a
band holds. Measured:

| run | item-rows | share of total | correction factor |
|---|---|---|---|
| baseline | ~1.06B | 100% | **1.00×** |
| L 21+ | ~750M | ~71% | 1.15× |
| M 11–20 | ~182M | ~17% | 1.66× |
| S ≤10 | 126,319,468 | ~12% | **3.07×** |

**The baseline came back at 1.949 to three decimals after the fix** — it
labelled every basket, so old and new baselines are the same set. That is the
control proving the fix corrects rather than merely deflates.

Band S's headline was inflated threefold. That single number was the entire
quantitative case for banding.

---

## 5. Method: the permutation null

Once the lift figures moved, the obvious question was how much of what remained
was chance. `median_max_lift ≈ 1.6` has no meaning without knowing what a
*structureless* clustering scores on the same data.

**The null:** permute the `need_state_cluster` column, preserving the exact
cluster-size distribution, then re-profile and re-score. One DuckDB pass, ~3
minutes, no new code.

```powershell
# Full population (no UNCLUSTERED rows)
.\.venv\Scripts\python.exe -c "import pandas as pd, numpy as np; d=pd.read_parquet(r'..\data\output\basket_need_state_clusters_k10_onedir_r1p5.parquet', columns=['basket_id','need_state_cluster']); d['need_state_cluster']=np.random.default_rng(42).permutation(d['need_state_cluster'].values); d.to_parquet(r'..\data\output\basket_need_state_clusters_k10_onedir_SHUFFLED_r1p5.parquet', index=False)"
```

> [!WARNING]
> **For a band, permute ONLY the non-`UNCLUSTERED` rows.** Shuffling the whole
> column scatters the `-1`s, the "band" becomes a random subset of all 57.1M
> baskets, and the null is meaningless.
>
> ```python
> m = (d["need_state_cluster"] != -1).values
> v = d.loc[m, "need_state_cluster"].values.copy()
> rng.shuffle(v)
> d.loc[m, "need_state_cluster"] = v
> ```

### Two reusable calibrations came out of this

**All four nulls produced ZERO need-states above lift 3 and zero above lift 5**,
across cluster counts from 78 to 2,090 and populations from 12.1M to 57.1M
baskets. `lift > 3` was a convention in this project; it is now an empirically
validated threshold. Chance never reaches it on this dataset.

**All four nulls sit in 1.199–1.261.** "Median max lift around 1.2 means
nothing here" is a reusable constant.

### The null rises with cluster size — do not reason the other way

| run | item-rows per cluster | null |
|---|---|---|
| L 21+ | ~9.7M | **1.261** |
| baseline | ~3.0M | 1.255 |
| M 11–20 | ~1.9M | 1.234 |
| S ≤10 | ~60k | **1.199** |

Counter-intuitive but mechanical: more item-rows means more products clear the
`MIN_PRODUCT_BASKETS=200` support floor, so the maximum is taken over a larger
pool of candidates, and the maximum of more draws is larger. **More data per
cluster gives a higher noise floor, not a lower one.** This was predicted wrong
once during the investigation; don't repeat it.

It also means **each run needs its own null**. Borrowing another run's floor
will mislead by up to 0.06 lift, which is ~13% of the signal being measured.

---

## 6. Results

All four runs, corrected baseline, each against its own permutation null.
`excess` = observed − own null, and is the number to compare.

| run | comms | obs | null | **excess** | ratio | >3 of all comms | >5 | >10 | twin / null |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** all 57.1M | 356 | 1.949 | 1.255 | **0.694** | 1.55× | 44/356 = **12.4%** | 13 | 0 | 6.3× |
| **M** 11–20 | 98 | 1.811 | 1.234 | 0.577 | 1.47× | 7/98 = 7.1% | 1 | 0 | **8.1×** |
| **L** 21+ | 78 | 1.732 | 1.261 | 0.471 | 1.37× | 7/78 = 9.0% | 1 | 0 | 6.8× |
| **S** ≤10 | 2,090 | 1.639 | 1.199 | 0.440 | 1.37× | 62/2,090 = 3.0% | 34 | **19** | 2.25× |

### Population coverage — and why it disagrees with the table above

`median_max_lift` takes a median **over communities**. Band S's communities
range from 45 baskets to 195,000; 2,028 of its 2,090 are junk, and they count
as much as the 62 real ones. The basket-weighted view inverts the ranking:

| run | % of *its* baskets in a need-state with lift > 3 | baskets | % of all 57.1M |
|---|---|---|---|
| baseline | 10% | 5.71M | **10.0%** |
| **S ≤10** | **20%** | 4.67M | 8.2% |
| M 11–20 | 6% | 0.73M | 1.3% |
| L 21+ | 4% | 0.87M | 1.5% |

Because the bands partition the population exactly, the banded approach as a
whole covers **~11.0%** of all baskets against the baseline's **10.0%** — a tie
once the whole-number rounding on those percentages is allowed for (union
10.5–11.5%, baseline 9.5–10.5%).

**Take from this: `median_max_lift` is the wrong summary when community sizes
span four orders of magnitude.** It is fine for the baseline (median community
135,094, fairly uniform) and misleading for band S.

### The decisive test

Neither view above is a clean comparison: band S's 20% is measured over small
baskets, the baseline's 10% over all baskets, most of which are large. The
direct question is *what share of the baseline's own ≤10-item baskets are in a
distinctive need-state?*

```
 small_baskets  in_distinctive  pct
      23341615       4534414.0 19.4
```

**19.4% for the baseline, ~20% for band S, over an identical set of 23,341,615
baskets.** The dedicated small-basket clustering — a separate graph, 2,090
communities, its own pipeline path — buys between 0.1 and 1.1 percentage points.

One caveat, and it runs against banding: the baseline's `max_lift` is computed
over need-states holding both small and large baskets, so a small basket can be
counted "distinctive" on the strength of a large-basket product. **19.4% is
therefore slightly generous to the baseline.** Not enough to change the verdict
at this margin, but it is the honest direction of the error.

---

## 7. Verdict

| question | answer |
|---|---|
| Does banding cover more shopping? | **No.** 19.4% vs ~20% on identical baskets. |
| Does banding find more need-states worth naming? | **No.** 12.4% of communities vs 3.0%. |
| Is the typical banded need-state more distinctive? | **No.** Excess 0.694 vs 0.440. |
| Does banding produce *sharper* occasions? | **Yes.** 34 need-states above lift 5 and 19 above lift 10, against the baseline's 13 and **zero**. Both nulls give zero, so these are real. |

**Banding is a discovery tool for sharp occasions, not a population
segmentation.** It describes the same ~4.6M small baskets the baseline already
describes, but carves them into purer groups — the meal deal at lift 15–21
rather than a mixed cluster at lift 4. That is worth something for *naming and
explaining* occasions. It is not worth a separate production pipeline.

Treat 19 as an upper bound: max lift rewards over-splitting, and band S's twin
Jaccard is 2.25× its null with 11 near-duplicate pairs among 225 profiled.

### What each band actually found

**Band S — the sharpest occasions, and a 90% junk tail.** The recognisable
occasions are here: a meal deal (sandwich + smoothie + crisps) at lift 15–21,
lunch-on-the-go variants, beer runs, flower purchases. But 1,889 of 2,090
communities fall below the profiling support threshold (median size 45
baskets). **Report ~201, never 2,090.**

**Band M — the worst over-splitter.** Twin Jaccard 0.429 against a 0.053 null —
8.1×, the highest of any run, with 8 near-duplicate pairs among 92 profiled —
despite having only 98 communities. It splits into two recognisable families:
fresh-food shops (cucumber, blueberries, tomatoes, apples at lift 2.7–3.3) and
snack/drinks shops (Fridge Raiders, Peperami, Powerade, energy drinks).

**Band L — capturing region, not occasion.** Its need-states group **Cookstown**
sausages, **Denny** pork, **Coleraine** cheddar, **Wilson's Country** potatoes,
**Connolly's** gammon, **Keelings** grapes and **Isle of Man Creamery** milk:
Northern Irish and Irish brands clustering together. A full weekly shop reflects
where someone lives and what their store stocks. That is a legitimate
segmentation — *regional / store weekly-shop archetypes* — but calling it a
need-state vocabulary would be wrong.

### The size effect is real; banding just isn't how to exploit it

Spearman(avg basket size, max lift), measured per run on the corrected data:

| run | size range | Spearman |
|---|---|---|
| baseline | 1 – 2,037 | **−0.83** |
| L 21+ | 21 – 2,037 | **−0.93** |
| S ≤10 | 1 – 10 | −0.55 |
| M 11–20 | 11 – 20 | **0.00** |

The correlation tracks how much size range each band contains — strong where
there is range to see it, exactly zero in the narrow band. The baseline's −0.83
reproduces the original −0.829 that started this investigation.

**Small weeks genuinely are more distinctive than large ones. The
full-population clustering already exploits that** — its distinctive
need-states are the small-basket ones. Isolating them first adds nothing.

---

## 8. What was proposed but not built

Kept for the record. None of it should be built unless the verdict above
changes.

**Stratified clustering with a shared temporal layer.** Cluster each band
separately; traverse between them using household week sequences, which exist
independently of any clustering. `basket_id` is `<household>_<year_week>`, so
the sequence is recoverable from the id alone regardless of band.
`need_state_graph.build_need_state_transitions()` consumes
`(basket_id, need_state)` pairs — union the band label spaces into one column
with distinct id ranges and it runs unchanged.

This was the answer to the objection that banding disconnects label spaces, so a
household moving between a top-up week and a big-shop week would have no
representable transition.

**Within-band vs across-band transitions.** `S34 → S91` is a genuine
occasion-to-occasion move; `S34 → L12` mostly encodes a change in shopping
*mode*, where the band change is the signal rather than the need-state pair.
Reported as one table the obvious one buries the interesting one.

**Cross-band profile linking.** The same occasion appears in several bands — a
meal deal in S, a meal-deal-flavoured corner of M — as unrelated ids. Jaccard
over top-N high-lift products would link them.

**Band boundaries are discontinuous.** A 15-item and a 16-item week land in
different label spaces. Some households flip bands over a single extra item.

**Sparsity.** ~800 states across four bands gives ~640,000 directed pairs
against 35.2M household-week steps; the current 356-state run observes 86,097
pairs. Several times thinner per cell, and `MIN_TRANSITION_SUPPORT=30` would
flag considerably more.

---

## 9. Limitations that still stand

**The grain is the root cause, and banding was a workaround for it.** Basket
size is a proxy for trip count, not a measurement of it. If anyone can obtain a
transaction or visit identifier from the warehouse, that single change would do
more than everything described here. Worth asking whoever owns
`product.product` and `cltv_hh_metrics_tpnb_base` before investing further.

**A longer aggregation window makes this worse.** Biweekly or monthly baskets
blend *more* occasions per row. The direction that helps is finer, and finer is
unavailable.

**The observation window is thin.** 21.9M households across 57.1M baskets is
**2.6 observed weeks per household**. `PIPELINE_TRANSITION_MAX_WEEK_GAP` is
`none`, which counts a 1-week and a 6-week step identically — check
`avg_week_gap` on every transition row before quoting a journey number.
Widening the ~8-week SQL window is the proper fix.

**Small bands lean toward lighter shoppers.** With 2.6 weeks observed per
household, "small weeks" does not cleanly mean "the light weeks of every
household". A selection effect, not a property of the occasions.

**Product descriptions are not fully trustworthy.** `tpnb 54739758` appears in
1,767,393 baskets (3.1%) described as "PENDRIVE FLASH DRIVE 12GB SLIQ" with
department "FRESH FRUIT/VEG/SALAD". It maps to exactly one tpna, so the
contradiction is in the source extract. Descriptions feed the product
embeddings, so corruption affects the clustering and not just the labels. Run
`src/audit_product_data.py` before trusting any label.

**Basket-size distribution, for reference.** ≤5 22.4% · 6–10 18.4% · 11–15
12.2% · 16–20 9.0% · 21–30 12.9% · 31–50 14.9% · 51+ 10.0%. Maximum observed:
2,037 distinct products in one household-week — that record is not a household.

---

## 10. How to judge any future clustering run

This is the part of the investigation worth keeping.

**1. Always run a permutation null (§5).** Every metric is uninterpretable
without one, and the null is per-run. This costs one profiling pass.

**2. Judge on `excess` = observed − own null**, not on the raw metric.

**3. Use the basket-weighted number for anything a stakeholder sees.** The
profile log's `BASKETS in a need-state with max lift > 3` line. Community-
weighted medians mislead when community sizes are heavy-tailed.

**4. Read `median_twin_jaccard` against its null** to catch the failure lift
cannot see: one occasion split many ways. Lift compares a cluster to the
population; only this compares clusters to each other.

**5. Do not use modularity.** A kNN graph is locally connected by construction,
so Leiden returns tidy modular partitions over structureless data. Modularity
0.4145 was read as evidence of structure in the full-population run; lift later
showed those clusters were meaningless.

**6. `pct_lift_over_3` in `experiment_log.csv` is computed over *profiled*
need-states and is misleading across runs with different thin-profile rates.**
Band S reads 27.6% against the baseline's 15.5% — apparently 1.8× better. Over
all communities it is 3.0% vs 12.4%, i.e. 4× worse. Divide by `n_communities`.

### Current reference points — all on `lift_basis="clustered"`, all null-corrected

| run | obs | null | excess | % of baskets, lift>3 |
|---|---|---|---|---|
| baseline all 57.1M | 1.949 | 1.255 | **0.694** | **10%** |
| M 11–20 | 1.811 | 1.234 | 0.577 | 6% |
| L 21+ | 1.732 | 1.261 | 0.471 | 4% |
| S ≤10 | 1.639 | 1.199 | 0.440 | 20% |

A new run should beat the baseline's **0.694 excess** and **10% basket
coverage**. Anything scored before 2026-09-29 carries `lift_basis="population"`
or a blank, and is not comparable to these.

---

## 11. Open questions this investigation did not answer

**How many need-states should there be?** Excess-over-null cannot answer it —
max lift rewards over-splitting, and the null falls as clusters shrink, so both
terms move the wrong way together. The clean answer is **held-out product
prediction**: leave one product out, score it under the need-state's
distribution versus the population's, average the log ratio in bits. It peaks
at the true cluster count, has an absolute zero, and needs no null. It is one
SQL query over the `ns_product_counts` and `product_baseline` tables
`profile_need_states.py` already builds. Not yet implemented.

**Is 10% basket coverage good?** It is unambiguously above chance — the null is
zero — but it means 90% of shopping weeks sit in a need-state with no signature.
Whether that is a floor imposed by the week grain (§9) or something the method
can improve is not established.

**Does `top_product_coverage` measure the right product?** It takes the rank-1
product *by lift*, which is systematically the rarest one clearing the 200-basket
floor — so it is pinned near that floor by construction and reads ~0.3% almost
everywhere. The "marker vs description" question it was built to answer needs
the best-covered product among the enriched ones, not the highest-lift one.
