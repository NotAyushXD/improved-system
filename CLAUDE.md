# CLAUDE.md — working agreement for this repository

Read this before touching anything. It covers how work actually happens here,
which is unusual enough that the normal defaults will waste hours.

This file does **not** explain the architecture — `ARCHITECTURE.md` and
`HOUSEHOLD_GRAPH_FLOW.md` do that well. This is operational: the machine, the
testing rules, the current working configuration, and the specific traps that
have already cost real time.

---

## 1. THE THREE RULES

### Rule 1 — You cannot run anything here

This working directory is a **OneDrive-synced copy with no Python
environment**. There is no numpy, no pandas, no igraph, no venv. Do not try to
install them. Do not write a throwaway script and try to run it.

What you *can* do locally:

- `compile()` a file to syntax-check it — always do this before handing work over
- `ast.parse()` to verify names, imports, call signatures, argument counts
- Run **pure-stdlib** logic to validate an algorithm before writing the numpy
  version. This has caught real bugs; see §7.
- `grep`/`find` to read the codebase

Anything touching numpy, pandas, igraph, leidenalg, sklearn, torch, duckdb,
networkit or lmdb **cannot be executed here**. State clearly which parts of
your change remain unverified.

### Rule 2 — Tests go in `src/test_pipeline.py`, and the user runs them

Do not write standalone test scripts. Add a `check_*()` function to
`test_pipeline.py`, register it in `main()`'s `checks` list, and hand it over.
The user runs it on the production box and pastes the output back.

```python
def check_thing():
    print(); print("=" * 70)
    print("THING — one line on what this pins and why it exists")
    print("=" * 70)
    ok = True
    if not condition:
        ok = _fail("what went wrong, specifically")
    else:
        print("  what was verified — OK")
    print("PASSED" if ok else "FAILED")
    return ok
```

Registered checks, in order:

`static` · `config` · `graph primitives` · `co-purchase additivity` ·
`need-state graph` · `clustering progress / graph build / Leiden parity` ·
`graph coverage / no silent label loss` · `label cache / no stale Leiden labels`
· `functional / parity / basket store` (skipped with `--fast`) ·
`prod outputs` (only with `--prod-outputs`)

### Rule 3 — Never trust silence, and never trust a "stale" doc

Two failure modes have dominated this project:

- **A healthy process that looks hung.** Stage 2 steps run for tens of minutes
  with no output. Diagnose before concluding (§8).
- **A filter that discards data without saying so.** A run once clustered 28.7M
  of 57.1M baskets and wrote a file with the correct row count and half the
  column empty. Nothing in the log mentioned it.

When you add a step that can drop, skip or cap anything: **count it and print
the count.** That principle is why `MIN_GRAPH_COVERAGE` and the label-cache
fingerprint exist.

---

## 2. THE MACHINE

| | |
|---|---|
| Path | `E:\ayp\improved-system` (Windows) |
| Shell | PowerShell |
| Python | **3.13**, venv at `src\.venv` |
| Cores | **64 logical** |
| RAM | **512 GB** |
| GPU | none — CPU only |
| Interpreter | `.\.venv\Scripts\python.exe` |

**Memory is not a constraint.** A full Stage 2 run peaks around 78 GB — 15% of
the box. Older docs (`MEMORY_ISSUES.md`, `PIPELINE_GUIDE.md`, `README.md`)
describe this as a "memory-constrained box" and quote ~29 GB as a ceiling.
That is historical. Do not optimise for memory without measuring first.

**Cores are the constraint that bites**, because most libraries here are
single-threaded by default.

**Python 3.13 is deliberate.** It was moved off 3.14 on 2026-09-23 because
neither `networkit` nor `graspologic-native` publishes cp314 wheels. Do not
"upgrade" it.

Always redirect long runs to a file:

```powershell
.\.venv\Scripts\python.exe -u .\script.py *> E:\ayp\run.log
```

`-u` for unbuffered output, `*>` to capture stderr too.

---

## 3. DATA SCALE

Every number below is measured on the real dataset, not estimated.

| Quantity | Value |
|---|---|
| Baskets (household × week) | **57,115,804** |
| Distinct products in baskets | 154,597 |
| Products with embeddings | 198,652 |
| Co-purchase matrix non-zeros | 1,474,348,846 |
| Training baskets sampled | 299,999 |
| Inference chunks | 1,143 × 50,000 |
| Node feature width | 388 (384 text + 4) |
| Basket embedding width | 64 |
| kNN edges (k=10, one-directional) | **508,168,451** |
| Need-states at gamma=1.5 | **356** |

A basket is **one household's entire week**, not a shopping trip — no source
table carries a transaction id. `basket_id` is `<household>_<year_week>`.

---

## 4. CURRENT WORKING CONFIGURATION

Established 2026-09-25 after a long investigation. These values are load-bearing
and were each chosen for a measured reason.

```
PIPELINE_BASKET_KNN_K=10
PIPELINE_USE_MUTUAL_KNN=false
PIPELINE_LEIDEN_RESOLUTION=1.5
PIPELINE_LEIDEN_N_ITERATIONS=2
PIPELINE_MIN_GRAPH_COVERAGE=0.95
```

Result: 508,168,451 edges, **100% basket coverage**, 356 communities,
modularity 0.4145, largest community 1.5% of baskets, ~36 min on 64 threads.

**If you change `k`, `mutual` or `resolution`, `.env` and the on-disk artifacts
must agree** or the pipeline rebuilds an hour of work and then refuses the
labels. See §6.

---

## 5. HOW TO RUN EACH PIECE

All commands from `E:\ayp\improved-system\src`.

```powershell
# What will actually be used — run before any long job
.\.venv\Scripts\python.exe .\config.py

# Tests
.\.venv\Scripts\python.exe .\test_pipeline.py
.\.venv\Scripts\python.exe .\test_pipeline.py --fast            # skip DuckDB/LMDB/torch
.\.venv\Scripts\python.exe .\test_pipeline.py --prod-outputs    # also check real artifacts

# Stage 2a, step 1 — build the kNN edge list only (no clustering)
.\.venv\Scripts\python.exe -u .\cluster_basket_embeddings.py --build-edges *> E:\ayp\edges.log

# Measure mutual-kNN coverage per k (one search, no Leiden)
.\.venv\Scripts\python.exe -u .\cluster_basket_embeddings.py --k 15 30 50 *> E:\ayp\coverage.log

# Stage 2a, step 2 — cluster. THIS is Leiden, not pipeline_main.
.\.venv\Scripts\python.exe -u .\cluster_leiden_networkit.py --sweep 1.0 1.5 2.0 *> E:\ayp\sweep.log
.\.venv\Scripts\python.exe -u .\cluster_leiden_networkit.py --resolution 1.5 *> E:\ayp\labels.log

# Everything else (Stage 0, 1, 2b GMM, 2c, 2.5)
.\.venv\Scripts\python.exe -u .\pipeline_main.py *> E:\ayp\pipeline.log
```

### Stage 2a runs OUTSIDE pipeline_main — this is deliberate

`pipeline_main.py` does **not** cluster. It loads labels produced by
`cluster_leiden_networkit.py`, and raises with the exact command to run if the
fingerprint doesn't match.

Why: `leidenalg` is single-threaded and does not finish at this scale — it ran
overnight on a 28.7M-vertex graph without completing one optimiser iteration.
NetworKit's `ParallelLeiden` did the larger 57.1M-vertex / 508M-edge graph in
36 minutes across 64 threads. `run_leiden_on_basket_graph()` still exists in
`cluster_basket_embeddings.py` and is still tested, but it is for small graphs
and parity checks only. **Do not wire it back into the main run.**

---

## 6. CACHES AND FINGERPRINTS

Every expensive artifact is fingerprinted. A mismatch means rebuild, never
silent reuse. Understand this before changing config.

| Artifact | Fingerprinted on | Rebuild cost |
|---|---|---|
| `basket_knn_edges_k10_onedir.parquet` | k, mutual, seed, basket count, embedding digest, backend | ~35 min |
| `basket_need_state_clusters_k10_onedir_r1p5.parquet` | the whole edge manifest + resolution + iterations + backend | ~36 min |
| `training_graphs.lmdb` | TOP_K, sub-cluster settings, seed | hours |
| `basket_gnn_model.pt` | architecture, settings, seed, graph fingerprint | 20 epochs |
| `copurchase_sparse.npz` | basket count + product count only | hours |
| `product_subclusters.pkl` | **fixed filename — not fingerprinted** | — |
| `basket_gnn_embeddings.parquet` | **not fingerprinted** | full inference |

**The two unfingerprinted artifacts are the dangerous ones.** If you change
`TOP_K`, the LMDB cache and model correctly rebuild — but
`basket_gnn_embeddings.parquet` sits there looking valid and you will cluster
vectors from the *previous* model. Delete it by hand.

**Cache filenames encode their settings** (`_edge_cache_path`,
`labels_path_for`), so a run at different settings builds a *different* file
rather than overwriting yours. Before that, a `pipeline_main` run picking up a
reverted `.env` destroyed a 35-minute graph and only then reported the
mismatch it had just caused — twice.

Note `basket_knn_edges.parquet` (no suffix) is a **different artifact**: Stage
2.5's drill-down edge list from `need_state_graph.save_basket_edges()`, capped
at `PIPELINE_BASKET_EDGE_WRITE_CAP`. Don't confuse it with the Stage 2a cache.

### Checking that artifacts agree

```powershell
.\.venv\Scripts\python.exe -c "import cluster_basket_embeddings as c, cluster_leiden_networkit as n, config, json; e=c._read_manifest(c._cache_manifest_path(c.BASKET_EDGES_PATH)); lp=n.labels_path_for(c.BASKET_EDGES_PATH, config.LEIDEN_RESOLUTION); l=c._read_manifest(c._cache_manifest_path(lp)); print('edges :', c.BASKET_EDGES_PATH); print('labels:', lp); print('MATCH :', bool(l) and l['edges']==e)"
```

`MATCH: True` plus a matching `PIPELINE_LEIDEN_RESOLUTION` means
`pipeline_main` will reuse both caches and reach Stage 2b in ~30 seconds.

---

## 7. LANDMINES

Each of these cost hours. None is obvious from reading the code.

### Two different `k` parameters

- `PIPELINE_TOP_K` (default 10) — **products** within one basket, Stage 1
- `PIPELINE_BASKET_KNN_K` (default 15, currently 10) — **baskets** across the
  whole population, Stage 2a

Changing the first is hours (retrain + re-embed). Changing the second is ~35
minutes. They are unrelated.

### Mutual-kNN silently deletes half the population — settled, do not revisit

Vertices are derived from edge endpoints, so a basket that loses every edge
stops existing. Measured coverage under `USE_MUTUAL_KNN=true`:

| k | coverage |
|---|---|
| 15 | 41.0% |
| 30 | 50.2% |
| 50 | 53.3% |

The curve flattens in the low 50s. It never reaches the 95% floor. The dropped
baskets are **not random** — they are the ones in sparse regions, i.e. the most
distinctive shopping missions. Use `USE_MUTUAL_KNN=false`, which gives 100%
coverage by construction; `k` then controls density only, never coverage.

### The resolution cliff

On the k=10 one-directional graph:

| gamma | communities | modularity | largest share |
|---|---|---|---|
| 0.05 | 1 | 0.0000 | 100% |
| 0.20 | 6,779 | 0.0036 | 99.8% |
| 0.50 | 12,943 | 0.0087 | 99.4% |
| **1.00** | 247 / 256 | 0.4209 | 3.1% |
| **1.50** | 350 / 356 | 0.4149 | 1.1% |
| **2.00** | 563 | 0.4114 | 1.0% |
| **3.00** | 1,045 | 0.4075 | 0.5% |

**Below 1.0 the graph collapses into one blob.** Above it there is a wide,
well-behaved plateau — modularity varies only 3% from gamma 1.0 to 3.0. Sweep
upward, never downward. You cannot get fewer than ~250 need-states by lowering
gamma; that requires hierarchical rollup (§9).

Modularity **favours the fine-grained end** and will happily rank a useless
partition above a usable one — the mutual graph scored 0.7101 with 94,845
communities. Always read it alongside `largest_share`.

### pynndescent pads failed rows with out-of-range sentinels

The warning `Failed to correctly find n_neighbors for some samples` fires on
every full-scale run and is **not cosmetic**. Unfilled slots get an index of
`n` (not `-1`) and an infinite distance. Following one as an array index raises
`IndexError`. Worse, `src * n + dst` with `dst == n` aliases exactly onto
`(src+1) * n + 0` — a real pair's key. Always filter `(dst >= 0) & (dst < n)`
before any key arithmetic.

### GMM spends 40+ minutes before EM iteration 1

`GaussianMixture(init_params='kmeans')` — sklearn's default — fits a **complete
k-means over all 57.1M points** before the first EM step, and `n_init` pays it
again per restart. sklearn prints `Initialization 0`, then nothing at all until
`Iteration 1`. Set `PIPELINE_GMM_INIT_PARAMS=k-means++` at full scale.

`PIPELINE_GMM_N_COMPONENTS=30` is a **placeholder**, flagged as such in the
code and in `.env`. The Leiden-vs-GMM ARI compares 356 communities against 30
components and will read low on granularity mismatch alone — that is not the
methods disagreeing.

### `$p.CPU` in PowerShell always reports zero

`Get-Process` returns a live object; `.CPU` re-reads `TotalProcessorTime` at
**access** time, not capture time. Sampling `$p1.CPU` / `$p2.CPU` around a
`Start-Sleep` measures two reads microseconds apart and reports ≈0, which looks
exactly like a hung process. It once led to a full misdiagnosis. Force scalars:

```powershell
$a = (Get-Process -Id <PID>).TotalProcessorTime.TotalSeconds
Start-Sleep -Seconds 60
$b = (Get-Process -Id <PID>).TotalProcessorTime.TotalSeconds
"CPU seconds in 60s wall: {0:N1}" -f ($b - $a)
```

~60 means one core saturated. ~0 means genuinely blocked.

### Windows console QuickEdit can freeze the process

Clicking inside a console window puts it in selection mode and blocks the
writing process at its next stdout write. With `python -u` that is immediate.
Symptoms: alive, 0% CPU, memory flat, log frozen mid-line. Press `Esc`. Prefer
redirecting to a file so it cannot happen.

---

## 8. DIAGNOSTIC PLAYBOOK

When something looks stuck, in this order:

1. **Press `Esc`** in the console — free, rules out QuickEdit.
2. **Is it alive?**
   ```powershell
   Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like '*pipeline_main*' } | Select-Object ProcessId, WorkingSetSize
   ```
3. **Is it working?** The scalar CPU form above.
4. **Where is it?** — this is the one that actually answers the question:
   ```powershell
   .\.venv\Scripts\py-spy.exe dump --pid <PID>
   ```
   `py-spy` is installed. It prints the live Python stack for every thread.
   `active+gil` on a thread means it is running right now.
5. **If it died:** `Get-Content E:\ayp\pipeline.log -Tail 40` — `*>` captures
   stderr, so a traceback will be there.

Capture the py-spy dump **before** killing anything.

---

## 9. STATE OF PLAY

### Working end to end
Stage 0, Stage 1 (co-purchase, sub-clusters, GNN training, full inference over
all 57.1M baskets), Stage 2a (edge build + NetworKit clustering, 100% coverage).

### In progress
Stage 2b (GMM) — slow, see §7. Stages 2c and 2.5 unreached at production scale.

### Designed but not built
- **Hierarchical rollup.** 356 need-states is too many for a business audience,
  and gamma cannot go lower. The route is: one centroid per community (356
  points × 64 dims — trivial), cluster those into ~25 super-need-states, keep
  the 356 as sub-states.
- **SNN (shared nearest neighbours).** Would weight edges by neighbour-list
  overlap instead of raw distance, suppressing hub bridges without deleting
  edges wholesale. Feasible here (~30–45 min on top of the existing search).
  Only worth building if `largest_share` shows hub collapse — at gamma 1.5 it
  is 1.1%, so there is currently no evidence it is needed.
- **Similarity threshold on kNN edges.** There is no knob for "only connect
  baskets more than X% similar" — you control *how many* neighbours, never *how
  close*. Distance survives only as the edge weight.

### Known data limitation
`PIPELINE_TRANSITION_MAX_WEEK_GAP=none` is a deliberate deviation. The training
SQL covers ~8 weeks and households do not shop weekly, so the strict setting
(`1`) leaves most households contributing no transitions at all. `none` treats
a 1-week and a 6-week step identically — check the `avg_week_gap` column on
every row of `need_state_transitions.parquet` before quoting any journey
number. The proper fix is widening the SQL window.

---

## 10. CONVENTIONS WHEN CHANGING CODE HERE

- **Every new parameter goes in `config.py` AND `.env.example`.** A test
  enforces that `.env.example` documents every key `config.py` reads.
- **Comments explain WHY, not what.** This codebase's comments carry the
  reasoning behind non-obvious choices — read them before changing the code
  they sit above. Several encode expensive lessons.
- **Fail before the expensive step, not after.** The coverage check runs before
  Leiden, not after, so a bad graph costs a kNN search rather than a day.
- **Refuse rather than silently recompute.** When a fingerprint mismatches,
  raise with the exact command the user should run. Don't quietly start an
  hour of work nobody asked for.
- **Make skipped work visible.** Print counts for anything dropped, capped or
  padded.
- **Don't add memory optimisations without measuring.** 512 GB. This has
  already misled one round of work.

### Doc map, and which docs are stale

| Question | Document |
|---|---|
| How does the architecture work | `ARCHITECTURE.md` |
| How do households/baskets/need-states connect | `HOUSEHOLD_GRAPH_FLOW.md` |
| How do I run it, what does each parameter do | `PIPELINE_GUIDE.md`, `.env.example` |
| Why does the code look like this | `src/MEMORY_ISSUES.md` |
| What do the source tables contain | `data/TABLE_REFERENCE.md` |
| **Current state, machine, Stage 2 config** | **this file** |

⚠ `README.md`'s "Current state" section and `GPT_CONTEXT_PROMPT.md` are **out
of date** as of 2026-09-26. They describe a memory-constrained box, incomplete
inference, and Stage 2 as unsolved. All three have been superseded — this file
is the authority on current state. `MEMORY_ISSUES.md` remains accurate as
*history*; its conclusions about what will and won't fit no longer hold on a
512 GB machine.
