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

  median_max_lift   THE headline. For each need-state, the lift of its most
                    over-represented product; median across need-states. Lift
                    ~1 means a cluster holds a representative sample of
                    products — a region of space, not a shopping occasion.

  pct_lift_over_3   Share of need-states that are characterised at all. This
                    is the number to show a stakeholder.

  lift_basis        Which basket population the lift denominator covered. Rows
                    with different values here are NOT comparable on ANY lift
                    column. See the warning below.

⚠ THE 1.95-vs-5.03 COMPARISON IS SUSPENDED — DO NOT QUOTE IT
────────────────────────────────────────────────────────────
Both seeded runs were scored before profile_need_states.py restricted the lift
baseline to labelled baskets. The <=10-item run clustered 23.3M of 57.1M
baskets, and its 5.03 was measured against a baseline that still included the
33.8M big baskets it had deliberately excluded — so an unknown part of the gap
to the full population's 1.95 is that mismatch rather than the size split
itself. Those rows carry lift_basis="population" and are kept for provenance
only.

To get a comparable pair, re-run profile_need_states.py and then evaluate_run.py
against BOTH label files. Until then the size split is unjustified by this log.

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
    python -u evaluate_run.py --clusters ../data/output/basket_need_state_clusters_k10_onedir_max10_r1p5.parquet --note "<=10 item baskets"
    python -u evaluate_run.py --show          # just print the log
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
    "n_profiled", "lift_basis", "median_max_lift", "pct_lift_over_3",
    "n_lift_over_5", "n_lift_over_10",
    "median_twin_jaccard", "n_twin_over_0p7", "pct_thin_profiles",
]

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


def append(row: dict, path: str = LOG_PATH):
    """Append, keeping one row per `run` — a rerun replaces its own entry."""
    log = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=COLUMNS)
    log = log[log["run"] != row["run"]] if "run" in log.columns else log
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    for c in COLUMNS:
        if c not in log.columns:
            log[c] = None
    log = log[COLUMNS].sort_values("median_max_lift", ascending=False)
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
    print("EXPERIMENT LOG — best first by median_max_lift")
    print("=" * 110)
    print(log.to_string(index=False))
    print()
    print("median_max_lift  : typical need-state's most over-represented product.")
    print("                   ~1 = holds a representative sample of products, i.e. not")
    print("                   an occasion. >3 = characterised. >5 = strongly so.")
    print("pct_lift_over_3  : share of need-states that mean anything at all.")
    print("median_twin_jaccard: cluster-vs-CLUSTER overlap. High = one occasion split")
    print("                   many ways. Lift cannot detect this; only this can.")
    print("modularity is deliberately absent — a kNN graph produces tidy partitions")
    print("over structureless data, so it cannot tell a real grouping from a geometric")
    print("one. Judge runs on lift.")

    # A sorted table invites reading down the lift column. Say plainly when the
    # rows in it were not measured the same way, rather than letting the sort
    # imply a ranking that does not exist.
    bases = sorted(set(log["lift_basis"].dropna())) if "lift_basis" in log.columns else []
    if len(bases) > 1:
        print()
        print("!" * 110)
        print(f"MIXED LIFT BASES IN THIS LOG: {', '.join(bases)}")
        print("Every lift column above is meaningless ACROSS those groups, and so is the")
        print("sort. Rows marked `population` were scored before the lift baseline was")
        print("restricted to labelled baskets — a run that labelled only part of the table")
        print("was measured against baskets it had excluded, which moves its lift by an")
        print("unknown amount. In particular the 1.95-vs-5.03 comparison that motivated the")
        print("size split is NOT valid evidence until both runs are re-profiled and")
        print("re-scored. Compare within one basis only.")
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
# ⚠ BOTH ROWS ARE lift_basis="population" AND THEIR LIFT COLUMNS ARE NOT
#   COMPARABLE — not to each other, and not to anything scored since.
#   profile_need_states.py now restricts the lift baseline to labelled baskets.
#   These were scored before that: the <=10-item run's 23.3M labelled baskets
#   were measured against a baseline containing all 57.1M, the 33.8M it had
#   excluded included. Its 5.03 is therefore inflated by an unknown amount
#   relative to the baseline run's 1.949, and the gap cannot be attributed to
#   the size split. They are kept for provenance; re-profile and re-score both
#   label files to replace them.
SEED_HISTORY = [
    {
        "run": "basket_need_state_clusters",
        "note": "BASELINE: all 57.1M baskets, k=10 one-directional, gamma 1.5 "
                "| LIFT NOT COMPARABLE: scored on the pre-fix population baseline",
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
                "| LIFT NOT COMPARABLE: baseline included the 33.8M excluded baskets",
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
    args = parser.parse_args()

    if args.seed_history:
        seed_history()

    if args.show or not args.clusters:
        show()
        return

    row = score(args.clusters, args.column)
    row["note"] = args.note
    append(row)
    print("Scored this run:")
    for k, v in row.items():
        print(f"  {k:24} {v}")
    print()
    show()


if __name__ == "__main__":
    main()
