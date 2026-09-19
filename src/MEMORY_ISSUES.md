# Memory and speed issues in this pipeline — what broke, and how it was fixed

This document walks through every stage of the pipeline where we hit (or
could have hit) an out-of-memory problem, in the order those stages
actually run — and then, at the end, the one place where the pipeline ran
out of *time* rather than memory. The short version of the memory story:
**almost every issue there came from the same root cause** — holding the
entire population of baskets (tens of millions of rows at real scale) as
one object in Python's memory for the whole script. Each fix either shrank
what's held at a given moment, moved the data somewhere that isn't Python's
memory (disk, DuckDB, LMDB), or removed a redundant copy of something
already large.

No fix ever reduced the amount of data used or the quality of the model —
every fix is an engineering/storage change, not a data-reduction one.

**Read this part too:** the speed section near the end
([Stage 1 — Inference was going to take nine days](#stage-1--inference-was-going-to-take-nine-days))
is not a separate topic. That slowdown was *caused by* one of the memory
fixes below (#4). Solving a memory problem by doing a small piece of work
per basket is correct — but it moves the cost from "how much RAM do we
need" to "how fast is that small piece of work", and at 57 million baskets
a millisecond in the wrong place is a fortnight of wall clock. The two
concerns are the same story told twice.

---

## Stage 0 — Reading the raw warehouse export

**Problem:** The household × product × week export is a huge flat table —
one row per household, per product, per week. Loading it in one shot with
`pd.read_parquet()` tries to build one giant table in memory in a single
step. At real scale this failed outright with `pyarrow.lib.ArrowMemoryError`
— pyarrow needs one big contiguous block of memory to do the conversion,
and there wasn't one available even though the machine had free RAM overall
(free RAM being fragmented/scattered doesn't help if one operation needs it
all in one contiguous piece).

**Fix:** Never load the whole file at once. Read it in bounded batches (a
few million rows at a time) and only ever hold one batch in memory. This
alone bought some headroom, but the *real* fix (below) was to stop
building anything basket-shaped in Python at all — the batches now get
piped straight into a local database instead.

---

## Stage 0/1 — Turning raw rows into "baskets"

**Problem:** Even reading the file in batches, the code still had to turn
millions of individual rows into "baskets" (all the products one household
bought in one week). The way this was done — building up Python
dictionaries that grow with every batch — meant that by the time every
batch had been read, those dictionaries held the **entire** basket
population anyway, just spread across many small steps instead of one big
one. Same problem, just hidden one layer deeper.

**Fix:** This grouping step now happens inside a small, local, embedded
database (DuckDB) instead of in Python. DuckDB knows how to group huge
tables into smaller ones without needing everything in memory at once (it
spills to disk on its own when needed), and can read the raw export
directly from its files — this is exactly what a database is built for, so
instead of re-inventing that logic in Python, we just use it. The result is
one basket-level table that lives in the database, not in a Python
variable. (An earlier version of this fix used a self-contained Postgres
instance instead of DuckDB — switched after Postgres's setup step turned
out to be blocked by a permission restriction on the deployment machine;
the underlying idea — do the grouping in a database, not Python — is the
same either way.)

---

## Stage 1 — Building the co-purchase matrix (which products get bought together)

**Problem:** This step originally tried to compute one giant matrix (every
product × every product) in a single mathematical operation across every
basket at once. At real data volumes this silently crashed with no error
message — the process just died.

**Fix:** Build the matrix in chunks — process, say, 500,000 baskets at a
time, add each chunk's contribution to a running total, and save progress
to disk after every chunk. If the process crashes or is stopped, restarting
it picks up from the last completed chunk instead of starting over. This
also protects against a subtler bug: if the leftover progress file is from
a *different* dataset (a different data pull, or the basket definition
changed), the code detects the mismatch and rebuilds from scratch instead
of silently mixing old and new results together.

There was also a **200GB threshold check** originally meant to catch this
matrix getting too big and fall back to something smaller — but at real
catalog sizes the matrix could exceed 150GB+ on its own, meaning the
"safety net" wouldn't even trigger before running out of memory. The real
fix wasn't a bigger threshold — it was realizing the full matrix never
needed to exist at all (see the next section).

---

## Stage 1 — Building the graph for one basket at a time

**Problem:** For the model to learn from a basket, each product in it needs
to know how often it's bought together with the *other* products in that
same basket. An earlier version of the code built one enormous table
covering every product that showed up anywhere across the whole training
sample — and because a large enough sample of baskets touches nearly the
entire product catalog, this table ended up close to catalog-size ×
catalog-size, i.e. well over 100GB, no matter how the sample was chosen.

**Fix:** A basket only ever needs the relationship between the handful of
products *inside that one basket* — never the whole catalog. So instead of
one shared giant table, each basket gets its own tiny slice, computed on
the fly and thrown away right after. The size of this slice depends only
on how many products are in that one basket (usually a few dozen), never
on how big the whole catalog is.

---

## Stage 1 — Picking which baskets to train on

**Problem:** Before training, a sample of ~300,000 baskets gets picked out
of the full population. The original code copied the *entire* basket table
just to pick a random subset out of it — a wasteful full duplication just
to extract a small piece.

**Fix (first pass):** Sample using lightweight index lists instead of
copying the whole table.

**Fix (final):** Now that baskets live in the database, the sampling
itself happens there too — one query asks the database for a random,
representative subset, and only that small subset (not the full
population) ever comes back into Python.

---

## Stage 1 — Holding all 300,000 training graphs in memory at once

**Problem:** After picking the training sample, the code built a full graph
object (nodes, edges, features) for every one of those 300,000 baskets and
kept *all of them* in one Python list at the same time — because the
training step needs to shuffle and re-visit them every epoch. At real
basket sizes, this list alone was roughly 20-25GB.

**Fix:** Build each graph once, and instead of keeping it in a Python list,
save it into a small on-disk database built exactly for this
("LMDB") — think of it as a lookup table on disk: "give me training
example #4,213" and it hands it back instantly, without needing all
300,000 examples loaded into memory at the same time. Training reads
whichever examples it needs, when it needs them.

---

## Stage 1 — Embedding *every* basket after training (not just the sample)

**Problem:** Once the model is trained, it needs to process **every**
basket in the whole population (potentially tens of millions), not just
the training sample — building a graph for each one and running it through
the model. Doing this for the entire population all at once would need
building millions of graph objects simultaneously — on the order of
terabytes.

**Fix:** Process baskets in bounded batches (e.g. 50,000 at a time): build
graphs for one batch, run them through the model, save the results, throw
the graphs away, move to the next batch. On top of that, this step is now
**restartable** — the database keeps track of which batches are done,
in-progress, or not-started-yet, so if the machine crashes 8 hours into a
long run, restarting it picks up only the unfinished batches instead of
starting the whole thing over.

---

## Stage 1 — A duplicate copy of the co-purchase matrix

**Problem:** One line of code converted the entire co-purchase matrix
(multiple billions of numbers) to a different numeric type, which required
making a **second full copy** of it — right at the exact moment a real
crash was observed. This copy served no real purpose, since the actual
place that number gets used already converts a much smaller, per-basket
slice to the right type anyway.

**Fix:** Removed the unnecessary whole-matrix copy entirely — one line
change, no downside, freed up roughly 17-34GB depending on data size.

---

## Stage 2 — Clustering all the basket embeddings into need-states

**Problem:** The last step groups baskets into need-states by first finding
each basket's nearest neighbors (in embedding space), then running a
community-detection algorithm on that neighbor graph. Finding nearest
neighbors the "obvious" way (compare every basket to every other basket)
is computationally impossible at tens of millions of baskets — it doesn't
just get slow, it effectively never finishes.

**Fix:** Use an approximate-nearest-neighbor method (`pynndescent`) that's
built for exactly this scale — confirmed installed and confirmed (via a
printed log line) that it's actually the method being used, not a much
slower fallback. Also added a print of how big the resulting neighbor
graph is (how many connections, roughly how much memory) *before* handing
it to the clustering step, so if this ever becomes the next bottleneck at
truly enormous scale, it'll show up as a visible number in the logs rather
than a silent crash.

This is the one area flagged as "**watch, don't assume solved**" — it's
handled for realistic scale today, but hasn't been stress-tested at the
very largest end.

---

## Scoring new baskets later — checking what's already been scored

**Problem:** When scoring a new batch of baskets, the code needs to skip
any basket that's already been processed before. The original approach
loaded *every previously-scored basket ID* into one big Python collection
just to check for overlaps — at real scale, that's a collection with tens
of millions of entries.

**Fix:** Do the comparison inside the database instead — stream just the
basket ID column (not the actual data) into a small temporary table, and
let the database do the "which of these are duplicates" check directly.
Nothing resembling the full list of IDs ever gets built as a Python object.

---

---

# Speed — when the pipeline ran out of time instead of memory

Everything above is about not running out of RAM. This section is about the
one place where the pipeline fit in memory perfectly well and simply would
not finish.

---

## Stage 1 — Inference was going to take nine days

**Problem:** Embedding every basket (the restartable chunked step, #7 above)
was measured on real data at roughly **17.4 minutes per 50,000-basket
chunk**. That works out to about **21 milliseconds per basket**. Across a
population of 57.1 million baskets, that's ~1,143 chunks and roughly **9 to
10 days of continuous running** on a CPU machine.

Nothing was crashing. Memory was fine. It was just never going to finish in
a sensible amount of time.

21 milliseconds is a suspicious number, and that's what made this findable.
The actual work per basket is tiny — build a graph of a few dozen products
and push it through a small model. The model part takes microseconds. So the
time had to be going somewhere that had nothing to do with the model.

**Cause — and it came from one of the memory fixes above.** Fix #4 solved a
100GB+ memory problem by giving each basket its own small slice of the
co-purchase table instead of building one giant shared one. That was the
right call, and it stays. But *the way* that slice was being taken was
quietly doing enormous amounts of unnecessary work:

> To get the co-purchase numbers for the ~30 products in one basket, the
> code asked for those 30 **rows** of the big co-purchase table, and only
> then narrowed down to the 30 **columns** it wanted.
>
> The catch is what a "row" contains. The row for a popular product — milk,
> bread, bananas — lists every other product it has *ever* been bought
> alongside, which at a 200,000-product catalog is a very large fraction of
> the catalog. So the code was copying out hundreds of thousands of numbers
> per product, for every product in the basket, in order to keep 900 of them
> (30 × 30) and throw the rest away.
>
> Then it did that again for the next basket. 57 million times.

**Fix:** Stop reading whole rows. Each row's entries are stored in sorted
order, so instead of copying the row and filtering it, the code now
**looks up** just the handful of products it actually needs — the same way
you'd find a word in a dictionary by opening it near the right letter
rather than reading every page from the start.

The cost of this no longer depends on how popular the products in the basket
are. A basket containing milk now costs the same as one containing an
obscure item, which was emphatically not true before.

**Why you can trust it on a half-finished run.** This is the important part.
A long inference run was already 370 chunks in — several days of compute
that nobody wanted to redo. A "faster" version that produced even slightly
different numbers would have made those 370 chunks inconsistent with
everything after them, and the only safe response would have been to start
over.

So the replacement was checked to be **exactly identical**, not
approximately identical:

- compared against the original implementation across **3,000 randomly
  generated co-purchase matrices** of varying size and density — every
  single value matched
- plus the awkward edge cases on purpose: products that co-occur with
  nothing at all, baskets of one product, the very first and very last
  product in the catalog, and baskets containing the entire catalog
- the same comparison is now a permanent test
  (`test_pipeline.py`, graph-primitives group), so a future change to this
  function can't silently alter results
- and `benchmark_inference.py` re-runs the comparison against **your real
  co-purchase matrix and your real baskets** before you rely on it

Because the numbers are identical, a half-finished run just **resumes**. The
chunk queue skips the chunks already marked complete; the one that was
interrupted mid-flight goes back in the queue and is redone.

**One supporting change.** The new lookup relies on each row's entries being
sorted and free of duplicates. That's normally already true of how the
co-purchase matrix is built, but "normally" isn't good enough when the
failure mode is silently reading *wrong* co-purchase numbers into every node
and edge in the pipeline. So `prepare_globals()` now explicitly guarantees
it. Both calls are no-ops when the matrix is already in that form, and
neither copies the matrix — so this costs nothing in the normal case and
removes an entire class of silent-corruption risk.

**How to check what this is actually worth on your data.** The speedup
depends on how dense your co-purchase matrix is — specifically, the average
number of entries per product row, which is exactly what the old code was
copying. That number is a property of your data, not something that can be
predicted from here. So rather than quoting a figure:

```
cd src
python benchmark_inference.py --remaining-baskets <however many you have left>
```

It runs in about two minutes and reports three things: whether old and new
agree element-for-element on your matrix, a per-basket cost breakdown of
each stage (slice extraction, graph building, model), and a projected wall
clock for the baskets you have left — single-process and with parallel
workers.

---

## Stage 2 — Still open: clustering at 57 million baskets

This one is **not fixed**, and is flagged here so it isn't discovered the
hard way after a long inference run finally completes.

**Problem:** The memory fixes above all concern getting *to* the basket
embeddings. Stage 2 then has to cluster them, and at 57 million baskets the
arithmetic is uncomfortable on a memory-constrained CPU machine:

| what | rough size at 57M baskets |
|---|---|
| the embeddings themselves | ~15 GB |
| ...while being normalised (makes a copy) | ~29 GB at peak |
| the nearest-neighbour graph | a few hundred million connections, several GB as a table, more once handed to the clustering library |
| GMM's internal working array | ~14 GB **per iteration**, and it runs many iterations, three times over |

The nearest-neighbour search itself is already handled (#9 — an approximate
method built for this scale), and the code prints the graph size before
clustering so it shows up as a number rather than a silent crash. But the
normalisation copy and the GMM working array are new at this volume.

**The likely fix, not yet implemented:** cluster a sample rather than the
whole population, then assign everything else. Fit Leiden and GMM on a few
million baskets, then label the remaining tens of millions by asking which
already-found group each one falls nearest to. Both halves of this already
exist in the codebase — it's exactly what `score_new_baskets.py` does for
newly arriving baskets (`assign_new_baskets_to_clusters` for the Leiden
side, the saved GMM model's own prediction for the other). What's missing is
wiring that path into the main run so the full population never has to be
clustered in one go.

This turns an intractable 57-million-point clustering into a tractable
few-million-point one plus a cheap streaming assignment pass, and it is the
next thing to build.

---

## The one-sentence summary

Every memory fix here follows the same idea: **don't build the whole thing
in Python memory just because you eventually need to look at all of it.**
Stream it in pieces, store it somewhere disk-backed (DuckDB, LMDB, plain
files) that's built for handling more data than fits in RAM, and only ever
pull a small, bounded piece into Python at any one time.

And the speed fix adds the natural follow-on: **once you're doing a small
piece of work per basket, make sure that small piece is actually small.**
Doing per-item work is what keeps memory flat — but it also means the cost
of that one item gets multiplied by tens of millions, so anything wasteful
hiding inside it stops being a rounding error and becomes the whole runtime.

---

## Summary table — memory

| # | Stage | Where the issue was | Main solution | File(s) changed |
|---|---|---|---|---|
| 1 | Stage 0 | Loading the raw warehouse export in one shot (`pd.read_parquet()`) | Read in bounded batches instead of one giant load; now DuckDB reads the parquet files directly, no Python-side batching needed at all | `parquet_loader.py` (original streaming fix, now retired) → `basket_store.py` (`build_baskets_table`, via DuckDB's `read_parquet()`) |
| 2 | Stage 0/1 | Grouping raw rows into baskets (Python dict accumulators) | Do the grouping inside a database (DuckDB), not Python | `parquet_loader.py` (`stream_build_baskets_and_units_avg`, now retired) → `basket_store.py` (`build_baskets_table`) |
| 3 | Stage 1 | Building the co-purchase matrix in one shot | Build it in checkpointed chunks, resumable on crash | `pipeline_main.py` (co-purchase chunk loop + checkpoint files) |
| 4 | Stage 1 | Per-basket co-purchase lookup table (200GB threshold) | Slice a tiny per-basket table on the fly instead of one shared giant one | `GraphBuilder.py` (`_basket_dense_cp_submatrix`) |
| 5 | Stage 1 | Picking the training sample (copied the whole basket table) | Sample via lightweight indexes, then via a database query | `GraphBuilder.py` (`sample_baskets`, now retired) → `basket_store.py` (`sample_training_baskets`) |
| 6 | Stage 1 | Holding all 300k training graphs in memory at once | Cache built graphs to disk (LMDB), read by index during training | `GraphBuilder.py` (`build_training_graphs`/`save_training_graphs`, now retired) → `lmdb_graph_cache.py`, `GNN_Train.py` |
| 7 | Stage 1 | Embedding every basket after training | Process in bounded, restartable chunks; save and discard per chunk | `GraphBuilder.py` (`embed_all_baskets_fast` → `_embed_basket_chunk` + `run_inference`), `basket_store.py` (chunk-queue table) |
| 8 | Stage 1 | Duplicate full-size copy of the co-purchase matrix | Removed the redundant whole-matrix copy | `GraphBuilder.py` (`prepare_globals`) |
| 9 | Stage 2 | Nearest-neighbor search across all basket embeddings | Use an approximate method (`pynndescent`), confirmed active; log graph size before clustering | `cluster_basket_embeddings.py` (`build_basket_knn_graph`) |
| 10 | Scoring | Checking new baskets against every previously-scored basket ID | Compare inside the database (anti-join), never build a Python set of all IDs | `score_new_baskets.py`, `basket_store.py` (`exclude_existing_basket_ids`) |

---

## Summary table — speed

| # | Stage | Where the issue was | Main solution | File(s) changed |
|---|---|---|---|---|
| 11 | Stage 1 | Embedding every basket: ~21 ms per basket → **~9-10 days** for 57.1M baskets. Taking a basket's slice of the co-purchase table read *entire rows* first (hundreds of thousands of entries for popular products) to keep 900 of them. A direct consequence of memory fix #4. | Look up only the needed entries by binary search over each row's sorted indices, instead of copying the row and filtering. Cost no longer scales with product popularity. Verified **bit-identical** over 3,000 random matrices + edge cases, so a part-finished run resumes instead of restarting. | `GraphBuilder.py` (`_basket_dense_cp_submatrix`) |
| 12 | Stage 1 | The new lookup needs each row's indices sorted and de-duplicated — usually already true, but a silent wrong-answer risk if ever not | `prepare_globals()` now guarantees canonical form explicitly. No-op when already canonical; no matrix copy either way | `GraphBuilder.py` (`prepare_globals`) |
| — | Stage 1 | No way to tell where per-basket time was going, or to confirm a "faster" version returns the same numbers on real data | Added a benchmark that reports old-vs-new equivalence, a per-stage cost breakdown, and a projected wall clock for the baskets remaining | `benchmark_inference.py` (new) |

---

## Known and still open

| Stage | Issue | Status |
|---|---|---|
| Stage 2 | Clustering 57M embeddings: ~29 GB peak during normalisation, ~14 GB per GMM iteration, several GB for the neighbour graph | **Not fixed.** Likely approach — cluster a few-million-basket sample, then assign the rest using the already-existing `assign_new_baskets_to_clusters` / saved-GMM prediction path. Next thing to build. |
| Stage 1 | Multi-process inference would cut wall clock further, but each worker needs its own copy of the embedding matrix and co-purchase matrix | **Not built.** Viable on Linux/macOS, where `fork` shares those pages copy-on-write. On Windows each worker gets a full copy, which may not fit — check the platform before launching workers. |
