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

---

## Quick start

```powershell
cd src
copy ..\.env.example ..\.env     # first time only, then edit
python .\config.py               # confirm what will actually be used
python .\test_pipeline.py        # verify before spending hours
python .\build_product_embeddings.py
python .\pipeline_main.py 2>&1 | Tee-Object -FilePath run.log
python .\test_pipeline.py --fast --prod-outputs   # confirm the output means something
```

The SQL in `data/*.sql` must be run against your warehouse first, with each
result downloaded as parquet into the matching `data/` subfolder. There is no
live warehouse connection anywhere in the Python code.

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

Verified at production scale: 57.1M baskets, 154,597 products, 1.47B
co-purchase non-zeros. Stage 0, Stage 1a, sub-clustering and GNN training all
complete; `test_pipeline.py` passes all six groups on the target machine.

**Not yet complete:** the full inference pass over 57M baskets, and Stage 2
clustering — which at this scale needs ~29GB peak and is expected not to fit
on a memory-constrained box. The sample-then-assign fix is designed but not
wired in. See [GPT_CONTEXT_PROMPT.md](GPT_CONTEXT_PROMPT.md) for the full
verified/unverified split.
