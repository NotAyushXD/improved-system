# Need-State Pipeline

Turns raw warehouse transaction data into **need-states** — data-driven
groupings of shopping baskets representing *why* a household shopped,
discovered without ever being told what product categories exist.

Raw rows → baskets → one small graph per basket → a GNN compresses each to 64
numbers → cluster those → need-states → graphs showing how need-states relate.

---

## Which document do I want?

| If you want to… | Read |
|---|---|
| **Run the pipeline** | [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) — setup, configuration, run order, production operations, troubleshooting |
| **Understand how it works** | [ARCHITECTURE.md](ARCHITECTURE.md) — data → nodes → embeddings → need-states, with worked examples |
| **Understand how households, baskets and need-states connect** | [HOUSEHOLD_GRAPH_FLOW.md](HOUSEHOLD_GRAPH_FLOW.md) — the two-graph structure, and journeys |
| **Know what every parameter does** | [.env.example](.env.example) — all 50, documented inline |
| **Know why the code looks like it does** | [src/MEMORY_ISSUES.md](src/MEMORY_ISSUES.md) — every memory and speed problem hit at scale, and the fix |
| **Know what the source tables contain** | [data/TABLE_REFERENCE.md](data/TABLE_REFERENCE.md) |
| **Brief someone (or an AI) on this codebase** | [GPT_CONTEXT_PROMPT.md](GPT_CONTEXT_PROMPT.md) — current state, what's verified, what isn't |
| **Work on this codebase** | [CLAUDE.md](CLAUDE.md) — the machine, the testing rules, the working configuration, and every trap that has already cost time |
| **Know how to tell whether a clustering run is any good** | [BASKET_BANDING_DESIGN.md](BASKET_BANDING_DESIGN.md) — a rejected hypothesis (clustering per basket-size band), and the permutation-null method built to reject it, which is now the standard way to judge any run. Then `profile_need_states.py` → `evaluate_run.py` → `export_need_states.py` |

---

## Quick start

```powershell
cd src
copy ..\.env.example ..\.env     # first time only, then edit
python .\config.py               # confirm what will actually be used
python .\test_pipeline.py        # verify before spending hours
python .\build_product_embeddings.py

# Stage 0/1 + the Stage 2a edge build. STOPS at Stage 2a and names the
# clustering command — by design, see the note below.
python -u .\pipeline_main.py *> run.log

# Stage 2a clustering, on all cores. Sweep first: the resolution cliff is a
# property of THIS graph, so a value carried over from another run is a guess.
python -u .\cluster_leiden_networkit.py --sweep 1.0 1.5 2.0 3.0 *> sweep.log
python -u .\cluster_leiden_networkit.py --resolution <from the sweep> *> labels.log

# Rerun: every cache now matches, so this reaches Stage 2.5 in minutes.
python -u .\pipeline_main.py *> run2.log

python .\test_pipeline.py --fast --prod-outputs   # confirm the output means something

# Find out what the need-states ARE. Without this they are just integers.
python -u .\profile_need_states.py
# Build this run's permutation null, profile it, then score both together.
# `excess` (observed minus its OWN null) is the only lift number comparable
# across runs — raw lift has a cluster-size-dependent floor.
python -u .\evaluate_run.py --clusters <labels> --null-clusters <shuffled labels> --note "what was different"
```

`cluster_basket_embeddings.py --build-edges` rebuilds *only* the kNN graph,
without running the rest of the pipeline — useful once embeddings exist, but it
cannot be the first step because it reads them.

The SQL in `data/*.sql` must be run against your warehouse first, with each
result downloaded as parquet into the matching `data/` subfolder. There is no
live warehouse connection anywhere in the Python code.

**Why clustering is separate from `pipeline_main.py`:** `leidenalg` is
single-threaded and does not finish at production scale — it ran overnight on a
28.7M-vertex graph without completing one optimiser iteration. NetworKit's
`ParallelLeiden` clusters the larger 57.1M-vertex graph in ~36 minutes across
64 threads. `pipeline_main.py` loads the labels that step produces, and stops
with the exact command to run if they are missing or don't match the graph.

---

## Five things to know before reading anything else

**1. A basket is a household's whole WEEK, not a shopping trip.** No source
table carries a transaction identifier, so week is the finest grain the data
supports. A big shop plus two top-ups merge into one basket.

**2. Households are never nodes in any graph.** The GNN's graphs have
*product* nodes. Households meet each other only in the basket-similarity
graph, and only as (household, week) pairs. A household exists downstream
purely as a prefix inside `basket_id`.

**3. A need-state is an occasion, not a customer segment.** The same household
appears in different need-states in different weeks.

**4. Similarity is not movement.** Two need-states being adjacent does not
mean households travel between them. Journeys require the directed transition
graph, not the adjacency graph.

**5. Nothing is configured in source files.** Every parameter lives in `.env`.
Some invalidate cached artifacts when changed — those are fingerprinted and
rebuild automatically.

---

## Current state

*Last updated 2026-09-30.*

Run at production scale on a **64-core, 512 GB** Windows box: 57,115,804
baskets, 154,597 products, 1.47B co-purchase non-zeros.

> [!IMPORTANT]
> **A rebuild is pending, and until it lands no lift figure in this repository
> describes the current model.** On 2026-09-30 the product embedding was found
> never to have reached the graph: `tpnb` is an int on the basket side and a
> str on both product sides, the lookup in `prepare_globals()` matched nothing,
> and every node carried an all-zero 384-dim vector — silently, with `in_dim`
> still reading 388 and every test passing. See **[CLAUDE.md](CLAUDE.md) §4b**
> for the fix, what it invalidates, and the exact run order.
>
> Encouragingly, 44 of 356 need-states cleared product lift 3 **with no product
> information in the model at all**, so the corrected run should be strictly
> better.

**Working end to end:** Stage 0, Stage 1 (co-purchase matrix, sub-clustering,
GNN training, and the full inference pass over all 57.1M baskets), Stage 2a
(edge build + NetworKit clustering, 100% basket coverage), and Stages 2.5 /
profiling / scoring.

**Parked:** Stage 2b (GMM) is opt-in behind `pipeline_main.py --with-gmm`. At
the placeholder `GMM_N_COMPONENTS=30` against ~356 Leiden communities the
Adjusted Rand Index is driven to ~0 by the granularity gap alone, so it cost a
full fit and reported nothing. GMM's real value here is scoring new baskets
directly, which `score_new_baskets.py` still uses.

**Memory is no longer the constraint.** Earlier versions of this document and
of [src/MEMORY_ISSUES.md](src/MEMORY_ISSUES.md) described a memory-constrained
box and put Stage 2 at ~29 GB peak, expected not to fit. The machine has 512 GB
and a full Stage 2 run peaks around 78 GB. The real constraints turned out to
be single-threaded libraries and a mutual-kNN filter that was silently dropping
half the population. Both are fixed; see [CLAUDE.md](CLAUDE.md) for the
measured detail.
