"""
evaluate_run.py

Scores one clustering run and appends it to a comparable log, so "did this
iteration do better?" is a table lookup rather than a memory exercise.

WHY THIS EXISTS
───────────────
The <=10-item experiment beat the full-population run decisively, and proving
that took a dozen ad-hoc one-liners whose results lived only in a chat window.
Worse, running the second profiling overwrote the first, so the comparison
could not be reproduced afterwards at all. This makes the scorecard a build
artifact: run it after each clustering, and `experiment_log.csv` accumulates.

WHAT IT MEASURES, AND WHY THESE
───────────────────────────────
Ordered by how much they actually tell you:

  excess            THE headline, where a null has been measured:
                    median_max_lift minus the SAME run's permutation null.
                    Raw lift is not interpretable on its own — see
                    "WHY EXCESS, NOT LIFT" below.

  median_max_lift   For each need-state, the lift of its most over-represented
                    product; median across need-states. Lift ~1 means a
                    cluster holds a representative sample of products — a
                    region of space, not a shopping occasion. Read against
                    `null_median_max_lift`, never alone.

  pct_lift_over_3   Share of need-states that are characterised at all. This
                    is the number to show a stakeholder. The >3 threshold is
                    empirically validated: across four permutation nulls,
                    chance never once reached it on this dataset.

  lift_basis        Which basket population the lift denominator covered. Rows
                    with different values here are NOT comparable on ANY lift
                    column. See the warning below.

WHY EXCESS, NOT LIFT
────────────────────
`median_max_lift` has a floor that depends on CLUSTER SIZE, so comparing it
across runs partly compares how big their clusters are. The mechanism is
mechanical: more item-rows per cluster means more products clear
`MIN_PRODUCT_BASKETS`, so the per-need-state maximum is taken over a larger
pool of candidates, and the maximum of more draws is larger. Measured across
four runs, the null ranged 1.199 (smallest clusters) to 1.261 (largest) — a
0.062 spread against a smallest real signal of 0.440, i.e. ~14% of it.

So: shuffle the label column preserving the size distribution, re-profile,
re-score, and subtract. `--null-clusters` does the subtraction and records
the provenance. Full method in BASKET_BANDING_DESIGN.md §5. **Each run needs
its OWN null** — borrowing another's misleads by up to that 0.062.

THE LIFT DENOMINATOR: SETTLED. THE NUMBERS: PROVISIONAL.
────────────────────────────────────────────────────────
Two separate questions, and only one of them is closed.

**Settled.** `profile_need_states.build_counts()` once scored a run against
baskets it had excluded — numerator over labelled baskets, denominator over
the whole table. That is fixed, and the fix is *validated*, because the
correction it implies was predicted and then observed:

    lift_vs_all_shopping = lift_within_band × band_lift

so the inflation is proportional to how small a share of all ITEM-rows a run
covers. The control is the full-population run, which labelled everything and
therefore has a correction factor of exactly 1.00× — it came back unmoved to
three decimals, proving the fix corrects rather than merely deflates. Every
band landed on its predicted value. See BASKET_BANDING_DESIGN.md §4 for the
table. `check_lift_formula` in test_pipeline.py pins the property that makes
it work: a clustering that does nothing must score lift 1.0 everywhere.

**Provisional.** Every lift figure measured before 2026-09-30 — including all
of the above — came from a model in which the product embedding never reached
a single graph node (CLAUDE.md §4b: `tpnb` was int on the basket side and str
on the product side, so the lookup matched nothing and every node carried an
all-zero 384-dim vector). The *methods* stand. The *numbers* do not describe
the current model, and are superseded by the first corrected run. Do not
quote an absolute lift from this log without checking its `when` against that
date.

The banding verdict is the exception and survives: it rested on a relative
comparison over an identical set of baskets, where both sides carried the same
defect.

  median_twin_jaccard
                    Each need-state vs its most similar neighbour, over top-10
                    product sets. Lift compares a cluster to the POPULATION;
                    this compares clusters to EACH OTHER, which lift
                    structurally cannot do. High values mean one occasion has
                    been split many ways.

  modularity        Reported, deliberately NOT trusted. A kNN graph is locally
                    connected by construction, so Leiden returns tidy modular
                    partitions over structureless data too. Modularity 0.4145
                    was once read here as evidence of real structure; product
                    lift later showed those clusters were meaningless. Keep it
                    for comparison between runs, never as proof.

  largest_share     One community holding most baskets is hub collapse, not a
                    need-state.

USAGE
    # score a run
    python -u evaluate_run.py --clusters ../data/output/basket_need_state_clusters_k10_onedir_r1p5.parquet --note "post embedding fix"

    # score it against its own permutation null, in one call — preferred
    python -u evaluate_run.py \
        --clusters      ../data/output/basket_need_state_clusters_k10_onedir_r1p5.parquet \
        --null-clusters ../data/output/basket_need_state_clusters_k10_onedir_SHUFFLED_r1p5.parquet \
        --note "post embedding fix"

    python -u evaluate_run.py --show          # just print the log

Building the null (shuffle the labels, keep the size distribution, then
profile it so --null-clusters can score it):

    python -c "import pandas as pd, numpy as np; p=r'../data/output/basket_need_state_clusters_k10_onedir_r1p5.parquet'; d=pd.read_parquet(p, columns=['basket_id','need_state_cluster']); m=(d['need_state_cluster']!=-1).values; v=d.loc[m,'need_state_cluster'].values.copy(); np.random.default_rng(42).shuffle(v); d.loc[m,'need_state_cluster']=v; d.to_parquet(p.replace('_k10_onedir_','_k10_onedir_SHUFFLED_'), index=False)"
    python -u profile_need_states.py --clusters ../data/output/basket_need_state_clusters_k10_onedir_SHUFFLED_r1p5.parquet

Permute ONLY the non-UNCLUSTERED rows, as above. Shuffling the whole column
scatters the -1s, the "clustered subset" becomes a random subset of the whole
table, and the null is meaningless.
"""

import argparse
import itertools
import os
from datetime import datetime, timezone

import pandas as pd

import config
from profile_need_states import outputs_for

OUTPUT_DIR = config.OUTPUT_DIR
LOG_PATH = os.path.join(OUTPUT_DIR, "experiment_log.csv")

COLUMNS = [
    "run", "note", "when",
    "n_baskets_labelled", "n_unclustered", "n_communities",
    "largest_share", "median_community_size",
    "n_profiled", "lift_basis",
    # excess first of the lift group: it is the comparable one. median_max_lift
    # carries a cluster-size-dependent floor (see the module docstring), so a
    # log sorted on it is partly sorted on cluster size.
    "excess", "null_median_max_lift", "null_run",
    "median_max_lift", "pct_lift_over_3",
    "n_lift_over_5", "n_lift_over_10",
    "median_twin_jaccard", "n_twin_over_0p7", "pct_thin_profiles",
]

# The column the log is ordered by, and the one to read first.
SORT_COLUMN = "excess"

# Which basket population the lift denominator covered, recorded per row rather
# than assumed, because the log outlives the code that wrote it.
#
#   "clustered"  — baseline restricted to labelled baskets. Current, correct.
#   "population" — baseline scanned every basket in the table regardless of
#                  label. Harmless for a run that labels everything; wrong for
#                  one that does not, and the log contains both kinds.
#
# Lift columns are meaningless across a change of basis. Anything scored before
# build_counts() started filtering the baseline is "population".
LIFT_BASIS = "clustered"


def score(clusters_path: str, column: str = "need_state_cluster") -> dict:
    """Everything worth comparing between two clustering runs, in one row."""
    profiles_path, summary_path = outputs_for(clusters_path)
    for p in (clusters_path, profiles_path, summary_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nRun profile_need_states.py against this label file "
                f"first — the scorecard is mostly built from its output."
            )

    # Named error rather than a bare pyarrow one. profile_need_states.py guards
    # this the same way; evaluate_run did not, and it became reachable on
    # 2026-09-30 when Stage 2b went opt-in — a default run writes no
    # need_state_cluster_gmm column at all.
    try:
        clusters = pd.read_parquet(clusters_path, columns=["basket_id", column])
    except Exception as e:
        available = list(pd.read_parquet(clusters_path).columns)
        raise SystemExit(
            f"{clusters_path} has no column {column!r} ({type(e).__name__}).\n"
            f"  available: {available}\n"
            f"  A run without `pipeline_main.py --with-gmm` writes Leiden labels "
            f"only, so need_state_cluster_gmm will be absent. Score with the "
            f"default --column need_state_cluster."
        ) from None

    profiles = pd.read_parquet(profiles_path)
    summary = pd.read_parquet(summary_path)

    labelled = clusters[clusters[column] != -1]
    sizes = labelled[column].value_counts()

    per_ns_max_lift = profiles.groupby("need_state")["lift"].max()

    # Cluster-vs-cluster similarity, restricted to need-states with a FULL
    # top-10. A cluster whose profile was truncated by the support threshold
    # shares its top-10 with every other truncated cluster for reasons that
    # have nothing to do with the clustering — including them reported
    # identical pairs that did not exist.
    counts = profiles.groupby("need_state").size()
    full = counts[counts >= counts.max()].index
    sets = profiles[profiles["need_state"].isin(full)].groupby("need_state")["tpnb"].apply(set)
    ids = list(sets.index)
    if len(ids) > 1:
        pairs = pd.DataFrame(
            [(a, b, len(sets[a] & sets[b]) / len(sets[a] | sets[b]))
             for a, b in itertools.combinations(ids, 2)],
            columns=["a", "b", "jac"],
        )
        best = pd.concat([pairs.groupby("a").jac.max(),
                          pairs.groupby("b").jac.max()]).groupby(level=0).max()
    else:
        best = pd.Series(dtype=float)

    thin = (summary["n_products_profiled"] < counts.max()).mean() \
        if "n_products_profiled" in summary.columns else float("nan")

    return {
        "run": os.path.splitext(os.path.basename(clusters_path))[0],
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "n_baskets_labelled": int(len(labelled)),
        "n_unclustered": int((clusters[column] == -1).sum()),
        "n_communities": int(labelled[column].nunique()),
        "largest_share": round(sizes.iloc[0] / len(labelled), 4) if len(sizes) else 0.0,
        "median_community_size": int(sizes.median()) if len(sizes) else 0,
        "n_profiled": int(len(per_ns_max_lift)),
        "lift_basis": LIFT_BASIS,
        "median_max_lift": round(float(per_ns_max_lift.median()), 3),
        "pct_lift_over_3": round(float((per_ns_max_lift > 3).mean()), 3),
        "n_lift_over_5": int((per_ns_max_lift > 5).sum()),
        "n_lift_over_10": int((per_ns_max_lift > 10).sum()),
        "median_twin_jaccard": round(float(best.median()), 3) if len(best) else float("nan"),
        "n_twin_over_0p7": int((best >= 0.7).sum()) if len(best) else 0,
        "pct_thin_profiles": round(float(thin), 3),
    }


def attach_null(row: dict, null_row: dict) -> dict:
    """
    Fold a null run's score into a real run's row as `excess`.

    Deliberately takes a SCORED null rather than a number: the null arrives
    via --null-clusters, is scored by the same score() function over the same
    profile artifacts, and brings its own run name with it. A plain
    `--null 1.255` would put a figure in the log with no provenance, which is
    exactly the failure mode `lift_basis` exists to prevent.
    """
    out = dict(row)
    out["null_run"] = null_row["run"]
    out["null_median_max_lift"] = null_row["median_max_lift"]
    out["excess"] = round(row["median_max_lift"] - null_row["median_max_lift"], 3)
    return out


def append(row: dict, path: str = LOG_PATH):
    """Append, keeping one row per `run` — a rerun replaces its own entry."""
    log = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=COLUMNS)
    log = log[log["run"] != row["run"]] if "run" in log.columns else log
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    for c in COLUMNS:
        if c not in log.columns:
            log[c] = None
    # Sorted on excess, not on median_max_lift — see the module docstring. Rows
    # with no measured null sort LAST rather than first or silently mid-table:
    # na_position="last" says "unranked", which is the honest reading, where
    # pandas' default would interleave them among real scores.
    log = log[COLUMNS].sort_values(
        SORT_COLUMN, ascending=False, na_position="last")
    log.to_csv(path, index=False)
    return log


def show(path: str = LOG_PATH):
    if not os.path.exists(path):
        print(f"no log yet at {path}")
        return
    log = pd.read_csv(path)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("=" * 110)
    print(f"EXPERIMENT LOG — best first by {SORT_COLUMN}; rows with no null sort last")
    print("=" * 110)
    print(log.to_string(index=False))
    print()
    print("excess           : median_max_lift MINUS this run's own permutation null.")
    print("                   THE comparable number. Raw lift has a cluster-size-")
    print("                   dependent floor (nulls measured 1.199-1.261), so comparing")
    print("                   median_max_lift across runs partly compares cluster size.")
    print("median_max_lift  : typical need-state's most over-represented product.")
    print("                   ~1 = holds a representative sample of products, i.e. not")
    print("                   an occasion. Read it against null_median_max_lift.")
    print("pct_lift_over_3  : share of need-states that mean anything at all. The >3")
    print("                   threshold is validated — four nulls, none ever reached it.")
    print("median_twin_jaccard: cluster-vs-CLUSTER overlap. High = one occasion split")
    print("                   many ways. Lift cannot detect this; only this can.")
    print("modularity is deliberately absent — a kNN graph produces tidy partitions")
    print("over structureless data, so it cannot tell a real grouping from a geometric")
    print("one. Judge runs on excess.")

    # ── Rows with no measured null ───────────────────────────────────────
    if "excess" in log.columns:
        n_no_null = int(log["excess"].isna().sum())
        if n_no_null:
            print()
            print(f"NOTE: {n_no_null} of {len(log)} row(s) have no permutation null, so no "
                  f"`excess`.")
            print("      They are sorted last, not ranked. median_max_lift alone cannot say")
            print("      whether they beat chance — build each one a null and re-score with")
            print("      --null-clusters. See BASKET_BANDING_DESIGN.md section 5.")

    # ── Lift basis ───────────────────────────────────────────────────────
    # A sorted table invites reading down the lift column. Say plainly when the
    # rows in it were not measured the same way, rather than letting the sort
    # imply a ranking that does not exist.
    #
    # A MISSING basis means "scored before the column existed", i.e. pre-fix —
    # it is not a row to be skipped. This used to read
    # `set(log["lift_basis"].dropna())`, which threw away exactly the evidence
    # the warning exists to detect: a log of 4 legacy rows plus 1 current row
    # collapsed to a single basis and printed nothing. Two of the four possible
    # log shapes were silent misses, including both of the ones a part-way
    # migration actually passes through.
    # The rule is "warn unless EVERY row is on the current basis". Checking for
    # more than one distinct basis is not enough: a log whose rows are ALL on a
    # stale basis is the worst case, not the safe one, because the inflation
    # differs per run (1.00x where a run labelled everything, 3.07x where it
    # labelled ~12% of the item-rows). Such a log looks internally rankable and
    # is not — and that is exactly the 1.949-vs-5.03 pair the seed history holds.
    UNRECORDED = "(pre-fix, unrecorded)"
    if "lift_basis" in log.columns:
        bases = sorted({UNRECORDED if pd.isna(v) else str(v) for v in log["lift_basis"]})
    else:
        # No column at all: every row in this log predates the basis fix.
        bases = [UNRECORDED] if len(log) else []

    stale = [b for b in bases if b != LIFT_BASIS]
    has_current = LIFT_BASIS in bases

    if stale and has_current:
        print()
        print("!" * 110)
        print(f"MIXED LIFT BASES IN THIS LOG: {', '.join(bases)}")
        print("Every lift column above is meaningless ACROSS those groups, and so is the")
        print(f"sort. Rows not marked `{LIFT_BASIS}` were scored before the lift baseline")
        print("was restricted to labelled baskets — a run that labelled only part of the")
        print("table was measured against baskets it had excluded, which inflates its lift")
        print("in proportion to how little of the table it covered (1.00x to 3.07x on the")
        print("runs on record). Compare within one basis only, and prefer `excess`.")
        print("!" * 110)
    elif stale:
        print()
        print("!" * 110)
        print(f"NO ROW IN THIS LOG IS ON THE CURRENT LIFT BASIS (found: {', '.join(bases)})")
        print("These rows were all scored against the WHOLE basket table regardless of how")
        print("much of it they labelled, so each one is inflated by a DIFFERENT factor —")
        print("1.00x for a run that labelled everything, 3.07x for one that labelled ~12%")
        print("of the item-rows. The log therefore looks internally comparable and is not;")
        print("ranking these against each other is the specific mistake that once made a")
        print("rejected hypothesis look decisive. Re-profile and re-score each label file")
        print("before reading anything off this table.")
        print("!" * 110)


# Measured results from the two runs done before this script existed, recorded
# so the comparison that motivated the size split is reproducible rather than
# living in a chat transcript. Both sets of numbers came from real runs on the
# full 57,115,804-basket population; the full-population profiles were
# overwritten by the <=10-item run before outputs_for() made filenames
# run-specific, which is exactly the gap this file closes.
#
# Blank fields were genuinely not measured at the time — not zero, not lost.
#
# ⚠ THESE TWO ROWS ARE PROVENANCE, NOT FINDINGS. Both carry
#   lift_basis="population", so neither is comparable to anything scored since.
#   What each turned out to BE is now known, and is recorded in the notes below
#   so nobody re-derives it:
#
#     * The BASELINE row labelled every basket, so its correction factor is
#       exactly 1.00x and it came back UNMOVED at 1.949 after the denominator
#       fix. That makes it the control that validated the fix — it proved the
#       fix corrects rather than merely deflates. Its own number is unchanged;
#       its STATUS changed from "suspect" to "the control".
#     * The SIZE SPLIT row's 5.03 was inflated 3.07x, because band S holds only
#       ~12% of all item-rows. Corrected, it is 1.639 — BELOW the baseline.
#       That single number was the entire quantitative case for banding, and
#       banding was rejected on the corrected figures.
#
#   Do not replace these rows with the corrected values. They are kept at their
#   AS-MEASURED numbers precisely so the correction is auditable; the corrected
#   run belongs in the log as its own row, from its own re-score.
#   BASKET_BANDING_DESIGN.md section 4 has the full table.
#
#   Both also predate the 2026-09-30 embedding fix (CLAUDE.md section 4b), so
#   like every pre-fix figure they describe a model with no product semantics
#   in it. Neither has a permutation null recorded, so neither gets an
#   `excess` — they sort last, which is correct: they are history, not results.
SEED_HISTORY = [
    {
        "run": "basket_need_state_clusters",
        "note": "BASELINE: all 57.1M baskets, k=10 one-directional, gamma 1.5 "
                "| THE CONTROL that validated the lift-denominator fix: labelled "
                "everything, correction factor 1.00x, unmoved at 1.949 after it "
                "| pre-embedding-fix, see CLAUDE.md 4b",
        "when": "2026-09-27 (recorded)",
        "n_baskets_labelled": 57115804, "n_unclustered": 0, "n_communities": 356,
        "largest_share": 0.015, "median_community_size": None,
        "n_profiled": 284, "lift_basis": "population",
        "median_max_lift": 1.949, "pct_lift_over_3": 0.155,
        "n_lift_over_5": None, "n_lift_over_10": None,
        "median_twin_jaccard": None, "n_twin_over_0p7": None,
        "pct_thin_profiles": None,
    },
    {
        "run": "basket_need_state_clusters_k10_onedir_max10_r1p5",
        "note": "SIZE SPLIT: only the 23.3M baskets with <=10 products, same k/gamma "
                "| AS-MEASURED 5.03 was inflated 3.07x (band holds ~12% of item-rows); "
                "corrected it is 1.639, BELOW the baseline, and banding was rejected "
                "on that | pre-embedding-fix, see CLAUDE.md 4b",
        "when": "2026-09-27 (recorded)",
        "n_baskets_labelled": 23341615, "n_unclustered": 33774189, "n_communities": 2090,
        "largest_share": 0.008, "median_community_size": 45,
        "n_profiled": 225, "lift_basis": "population",
        "median_max_lift": 5.03, "pct_lift_over_3": 0.973,
        "n_lift_over_5": 114, "n_lift_over_10": 58,
        "median_twin_jaccard": 0.429, "n_twin_over_0p7": 11,
        "pct_thin_profiles": 0.107,
    },
]


def seed_history(path: str = LOG_PATH):
    """Write the two pre-existing runs into the log, without clobbering newer rows."""
    for row in SEED_HISTORY:
        append(row, path)
    print(f"Seeded {len(SEED_HISTORY)} historical runs into {path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Score a clustering run and append it to experiment_log.csv"
    )
    parser.add_argument("--seed-history", action="store_true",
                        help="record the two runs done before this script existed")
    parser.add_argument("--clusters", default=None)
    parser.add_argument("--column", default="need_state_cluster")
    parser.add_argument("--note", default="", help="what was different about this run")
    parser.add_argument("--show", action="store_true", help="print the log and exit")
    parser.add_argument(
        "--null-clusters", default=None, metavar="PATH",
        help="label file of this run's PERMUTATION NULL — the same labels "
             "shuffled, preserving the cluster-size distribution. It is scored "
             "the same way, and the difference is recorded as `excess`, which is "
             "the only lift number comparable across runs. Profile it first "
             "(profile_need_states.py --clusters <that file>). Omit it and the "
             "row gets no excess and sorts last. See the module docstring.")
    args = parser.parse_args()

    if args.seed_history:
        seed_history()

    if args.show or not args.clusters:
        show()
        return

    row = score(args.clusters, args.column)
    row["note"] = args.note

    if args.null_clusters:
        if os.path.abspath(args.null_clusters) == os.path.abspath(args.clusters):
            raise SystemExit(
                "--null-clusters is the same file as --clusters. The null must be a "
                "SHUFFLED copy of the labels; scoring a run against itself gives "
                "excess 0 and means nothing. See the module docstring for the "
                "shuffle command."
            )
        print(f"Scoring the permutation null from {args.null_clusters} ...")
        null_row = score(args.null_clusters, args.column)
        row = attach_null(row, null_row)
        print(f"  null median_max_lift = {null_row['median_max_lift']}  "
              f"-> excess = {row['excess']}")
        if row["excess"] <= 0:
            print("  WARNING: excess <= 0. This clustering does no better than a "
                  "size-preserving shuffle of its own labels — it has found no "
                  "product structure at all.")
    else:
        print("NOTE: no --null-clusters given, so this row gets no `excess` and will "
              "sort last.\n      median_max_lift alone cannot say whether this run "
              "beats chance: the null\n      has a cluster-size-dependent floor "
              "(measured 1.199-1.261 across four runs).")

    append(row)
    print("Scored this run:")
    for k, v in row.items():
        print(f"  {k:24} {v}")
    print()
    show()


if __name__ == "__main__":
    main()
