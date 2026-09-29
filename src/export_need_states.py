"""
export_need_states.py

One workbook containing every need-state run, for people who do not read parquet.

Discovers each clustering run from its filename, computes the relationships
between need-states, and writes five sheets:

  run_scorecard        one row per run — how good was this clustering
  need_state_summary   one row per (run, need_state) — size, households,
                       avg products, label, max/avg/median lift
  need_state_products  one row per (run, need_state, tpnb) — the detail table:
                       description, department, baskets, lift, share
  need_state_adjacency one row per (run, need_state_a, need_state_b) — which
                       need-states BORDER each other, and how strongly
  need_state_transitions  one row per (run, from, to) — where households
                       actually MOVE, with avg_week_gap

ADJACENCY AND TRANSITIONS ARE DIFFERENT QUESTIONS
─────────────────────────────────────────────────
Adjacency contracts the basket kNN graph: an edge whose two baskets landed in
different need-states is a boundary between them. It is undirected, static, and
says *these two occasions look alike*.

Transitions walk each household's own weeks in order. Directed, temporal, and
says *households actually go from one to the other*. Similarity is not movement
— only the transition table supports a journey claim.

WHY ADJACENCY IS COMPUTED HERE IN SQL
─────────────────────────────────────
need_state_graph.build_need_state_adjacency() does this in pandas, which is
fine at need-state grain but has to join 192M basket-level edges twice to get
there. DuckDB streams that. The formula matches: n_edges, weight_sum, and lift
as observed-over-expected under independence.

A NOTE ON avg_week_gap
──────────────────────
PIPELINE_TRANSITION_MAX_WEEK_GAP is currently `none`, which counts a 1-week and
a 6-week step identically. `avg_week_gap` is therefore the column that tells you
whether a transition is really "next week" or "eventually". Check it before
quoting any journey number.

USAGE
    python -u export_need_states.py
    python -u export_need_states.py --out E:\\ayp\\need_states.xlsx --top-adjacency 2000
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

# Windows defaults stdout to cp1252, and PowerShell's `*>` redirection makes it
# strict — so a single character outside that codepage raises UnicodeEncodeError
# and kills the run. A box-drawing dash did exactly that here, after the
# discovery phase had already completed.
#
# errors="replace" downgrades that from a crash to a '?'. The encoding is left
# alone deliberately: forcing UTF-8 would fix the log file but make PowerShell
# render every em dash as mojibake unless the reader passes -Encoding UTF8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

import config
import duckdb_manager
import need_state_graph
from cluster_basket_embeddings import (
    UNCLUSTERED, _cache_manifest_path, _read_manifest, progress_step,
)
from profile_need_states import outputs_for

OUTPUT_DIR = config.OUTPUT_DIR
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "need_states_export.xlsx")


def discover_runs(column: str = "need_state_cluster"):
    """
    Every label file paired with a profile and an edge list that PROVABLY
    belong to it.

    Filename convention alone is not enough, and trusting it produced a wrong
    export. Two legacy artifacts predate the run-specific naming scheme:

      need_state_profiles.parquet   written by whichever run profiled last —
                                    the baseline resolved to it and was
                                    exported carrying band S's 2,090
                                    need-states instead of its own 356.
      basket_knn_edges.parquet      the old k=15 MUTUAL graph from an aborted
                                    run, which the baseline's adjacency would
                                    have been computed from despite its labels
                                    coming from the k=10 one-directional graph.

    Both are now checked rather than assumed:
      * the profile must describe the same number of need-states the label
        file actually contains;
      * the edge list must carry the exact manifest the labels were
        fingerprinted against. Unverifiable pairings get no adjacency rather
        than a plausible-looking wrong one.
    """
    runs = []
    for clusters in sorted(glob.glob(os.path.join(OUTPUT_DIR,
                                                  "basket_need_state_clusters*.parquet"))):
        name = os.path.basename(clusters)
        profiles, summary = outputs_for(clusters)
        if not (os.path.exists(profiles) and os.path.exists(summary)):
            print(f"  SKIP {name}: no profile beside it")
            continue

        present = pd.read_parquet(clusters, columns=[column])[column].unique()
        n_labels = int(len(present) - (1 if UNCLUSTERED in present else 0))
        n_summary = len(pd.read_parquet(summary, columns=["need_state"]))
        if n_labels != n_summary:
            print(f"  SKIP {name}: its label file holds {n_labels:,} need-states but "
                  f"{os.path.basename(summary)} describes {n_summary:,}. They are not "
                  f"the same run — re-profile this label file to export it.")
            continue

        stem = os.path.splitext(name)[0]
        edges_stem = re.sub(r"_r[0-9p]+$", "", stem).replace(
            "basket_need_state_clusters", "basket_knn_edges")
        edges = os.path.join(OUTPUT_DIR, edges_stem + ".parquet")

        # Provenance, not just existence: the label manifest embeds the edge
        # manifest it was built from, so an exact match is verifiable.
        verified = None
        if os.path.exists(edges):
            want = _read_manifest(_cache_manifest_path(clusters))
            have = _read_manifest(_cache_manifest_path(edges))
            if want and have and want.get("edges") == have:
                verified = edges
            elif want is None:
                print(f"  {name}: no manifest, so its edge list cannot be verified — "
                      f"adjacency skipped rather than computed from a possibly "
                      f"unrelated graph ({os.path.basename(edges)})")
            else:
                print(f"  {name}: {os.path.basename(edges)} is not the graph these "
                      f"labels came from — adjacency skipped")

        runs.append({
            "run": stem, "clusters": clusters, "profiles": profiles,
            "summary": summary, "edges": verified,
        })
    return runs


def adjacency(con, edges_path: str, clusters: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Which need-states border each other, from the basket-level kNN edges.

    `lift` here is observed edges over what independence would predict, so a
    lift of 1 means two need-states touch exactly as often as their sizes
    imply — i.e. no meaningful adjacency. At full scale the adjacency table is
    ~79% of all possible pairs, so it is ONLY readable filtered by lift.
    """
    con.register("ns_labels", clusters.rename(columns={clusters.columns[1]: "ns"}))
    return con.execute(f"""
        WITH e AS (
            SELECT a.ns AS ns_a, b.ns AS ns_b, x.weight
            FROM read_parquet('{edges_path}') x
            JOIN ns_labels a ON a.basket_id = x.basket_a
            JOIN ns_labels b ON b.basket_id = x.basket_b
            WHERE a.ns <> {UNCLUSTERED} AND b.ns <> {UNCLUSTERED} AND a.ns <> b.ns
        ),
        pairs AS (
            SELECT LEAST(ns_a, ns_b) AS need_state_a,
                   GREATEST(ns_a, ns_b) AS need_state_b,
                   COUNT(*) AS n_edges,
                   SUM(weight) AS weight_sum
            FROM e GROUP BY 1, 2
        ),
        deg AS (
            SELECT need_state_a AS ns, SUM(n_edges) AS d FROM pairs GROUP BY 1
            UNION ALL
            SELECT need_state_b, SUM(n_edges) FROM pairs GROUP BY 1
        ),
        degree AS (SELECT ns, SUM(d) AS degree FROM deg GROUP BY 1),
        total AS (SELECT SUM(n_edges) AS m FROM pairs)
        SELECT p.need_state_a, p.need_state_b, p.n_edges,
               ROUND(p.weight_sum, 2) AS weight_sum,
               ROUND(p.n_edges * 1.0 / (SELECT m FROM total), 6) AS share_of_all_edges,
               ROUND((p.n_edges * 1.0 / (SELECT m FROM total))
                     / NULLIF((da.degree * 1.0 / (2 * (SELECT m FROM total)))
                            * (db.degree * 1.0 / (2 * (SELECT m FROM total))) * 2, 0), 3)
                   AS lift
        FROM pairs p
        JOIN degree da ON da.ns = p.need_state_a
        JOIN degree db ON db.ns = p.need_state_b
        ORDER BY p.n_edges DESC
        LIMIT {top_n}
    """).df()


def transitions(clusters: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Where households actually move, week to week. Reuses need_state_graph's
    own implementation rather than reimplementing it — it handles the year
    boundary (202552 -> 202601 is a gap of 1, not 49) and carries avg_week_gap.
    """
    labelled = clusters[clusters[column] != UNCLUSTERED]
    return need_state_graph.build_need_state_transitions(
        labelled.rename(columns={column: "need_state_cluster"}),
        cluster_col="need_state_cluster",
        max_week_gap=config.TRANSITION_MAX_WEEK_GAP,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export every need-state run to one Excel workbook."
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--column", default="need_state_cluster")
    parser.add_argument("--top-adjacency", type=int, default=5000,
                        help="adjacency rows kept per run, strongest first. The full "
                             "table can be ~79%% of all possible pairs and is unreadable "
                             "unfiltered (default 5000)")
    parser.add_argument("--skip-transitions", action="store_true",
                        help="transitions walk 57M rows per run; skip for a fast export")
    args = parser.parse_args()

    con = duckdb_manager.get_connection()
    runs = discover_runs(args.column)
    if not runs:
        raise SystemExit(
            f"no runs found in {OUTPUT_DIR}. Expected "
            f"basket_need_state_clusters*.parquet with a profile beside each — "
            f"run profile_need_states.py first."
        )
    print(f"Found {len(runs)} run(s): {[r['run'] for r in runs]}\n")

    summaries, products, adjacencies, transitions_all = [], [], [], []

    for r in runs:
        print(f"-- {r['run']}")
        summary = pd.read_parquet(r["summary"])
        profile = pd.read_parquet(r["profiles"])

        # All three, because each alone misleads. max_lift is the only one
        # comparable BETWEEN need-states — it is always the rank-1 product. But
        # a single high-lift outlier can make a diffuse cluster look sharply
        # defined, which is what avg_lift and median_lift are here to expose:
        # max >> avg is one spike, max ~ avg is a broadly characterised
        # need-state.
        #
        # The catch is that avg/median are taken over the top-N products
        # build_profiles kept, and N varies. A need-state with only 3 products
        # above the support threshold averages its best 3 while a full one
        # averages 10 including the weaker ranks — so they read HIGH exactly
        # where the profile is thinnest. Read them against
        # n_products_profiled, never on their own. (Same trap as the Jaccard
        # 1.0 incident; evaluate_run.py guards it by restricting to full
        # top-10 sets.)
        lift_stats = profile.groupby("need_state")["lift"].agg(
            max_lift="max", avg_lift="mean", median_lift="median")
        # profile_need_states.py now writes these into the summary parquet too.
        # Drop before merging rather than letting pandas suffix them _x/_y —
        # recomputing from the profile keeps old summaries working and keeps
        # both artifacts agreeing by construction.
        summary = summary.drop(columns=[c for c in lift_stats.columns if c in summary.columns])
        summary = summary.merge(lift_stats, left_on="need_state", right_index=True, how="left")
        summary.insert(0, "run", r["run"])
        profile.insert(0, "run", r["run"])
        summaries.append(summary)
        products.append(profile)
        print(f"   {len(summary):,} need-states, {len(profile):,} product rows")

        clusters = pd.read_parquet(r["clusters"], columns=["basket_id", args.column])

        if r["edges"]:
            with progress_step(f"adjacency for {r['run']}"):
                adj = adjacency(con, r["edges"], clusters, args.top_adjacency)
            adj.insert(0, "run", r["run"])
            adjacencies.append(adj)
            print(f"   {len(adj):,} adjacency pairs (top {args.top_adjacency:,})")
        else:
            print(f"   no edge file — adjacency skipped")

        if not args.skip_transitions:
            with progress_step(f"transitions for {r['run']}"):
                tr = transitions(clusters, args.column)
            tr.insert(0, "run", r["run"])
            transitions_all.append(tr)
            print(f"   {len(tr):,} directed transitions")

    scorecard_path = os.path.join(OUTPUT_DIR, "experiment_log.csv")
    scorecard = pd.read_csv(scorecard_path) if os.path.exists(scorecard_path) else pd.DataFrame()

    sheets = {
        "run_scorecard": scorecard,
        "need_state_summary": pd.concat(summaries, ignore_index=True),
        "need_state_products": pd.concat(products, ignore_index=True),
        "need_state_adjacency": pd.concat(adjacencies, ignore_index=True) if adjacencies else pd.DataFrame(),
        "need_state_transitions": pd.concat(transitions_all, ignore_index=True) if transitions_all else pd.DataFrame(),
    }

    try:
        with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
            for name, df in sheets.items():
                # Excel tops out at 1,048,576 rows; truncate loudly rather than
                # letting the writer fail after minutes of work.
                if len(df) > 1_000_000:
                    print(f"  TRUNCATING {name}: {len(df):,} rows exceeds Excel's limit")
                    df = df.head(1_000_000)
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"\nSaved {args.out}")
    except ImportError:
        print("\nopenpyxl is not installed, so no .xlsx was written.")
        print("  pip install openpyxl   — or use the CSVs below, which are always written.")

    # CSVs regardless: they survive the row limit, diff cleanly, and do not
    # depend on an Excel engine being present.
    base = os.path.splitext(args.out)[0]
    for name, df in sheets.items():
        path = f"{base}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path} ({len(df):,} rows)")

    print("""
READING THESE
  need_state_summary     max_lift is the only lift stat comparable BETWEEN need-states.
                         avg_lift and median_lift are taken over the top-N products
                         kept per need-state, so they read high wherever
                         n_products_profiled is small — read them together with that
                         column. max >> avg is a single spike; max ~ avg is broadly
                         characterised.
  need_state_products    `lift` distinguishes a need-state: the product's share of that
                         need-state's ITEMS over its share of all items, so trip size
                         is divided out rather than rewarded. `share_within_need_state`
                         is that first share alone and sums to 1.0 per need-state. High
                         lift on a tiny share is a marker, not a description.
                         Lift is computed against the LABELLED baskets only, so runs
                         that labelled different shares of the population are not
                         directly comparable — see lift_basis in run_scorecard.
  need_state_adjacency   similarity, NOT movement. Filter by `lift` — unfiltered it
                         is most of the possible pairs and says nothing.
  need_state_transitions movement. `avg_week_gap` matters: with
                         PIPELINE_TRANSITION_MAX_WEEK_GAP=none a 1-week and a
                         6-week step count the same, so check it before calling
                         anything a journey.""")


if __name__ == "__main__":
    main()
