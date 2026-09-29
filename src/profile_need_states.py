"""
profile_need_states.py

Turns anonymous need-state integers into something readable.

Stage 2 produces `need_state_cluster` values 0..N. Nothing in the pipeline
says what any of them ARE, which makes every downstream artifact — the
adjacency graph, the transition matrix, the journeys — a set of relationships
between unnamed things. This closes that gap: for each need-state, which
products and which departments are OVER-REPRESENTED relative to the whole
population.

WHY LIFT AND NOT FREQUENCY
──────────────────────────
The most frequent product in almost every need-state is whatever is most
frequent overall — bananas, milk, bread. Ranking by raw count gives 356
near-identical lists and tells you nothing.

    lift = (items of product P in need-state / all items in need-state)
         / (items of product P overall       / all items overall)

A lift of 8 means "product P takes up eight times more of this need-state's
shopping than it takes up of shopping in general". Both the lift and the raw
basket count are reported, because lift alone can crown a product that appears
in forty baskets — hence MIN_PRODUCT_BASKETS.

THE DENOMINATOR IS ITEMS, NOT BASKETS — DO NOT "FIX" THIS
─────────────────────────────────────────────────────────
The obvious-looking alternative is basket incidence:

    P(product in basket | need-state) / P(product in basket)

It reads more naturally, it has been proposed twice, and it is wrong here,
because need-states differ enormously in basket size — a top-up shop holds ~4
products, a big shop ~40. A 40-item basket has ten times more chances to
contain ANY given product, so basket incidence rises for every product at once
with trip size. Simulated on a pure-size null — identical product mix in every
need-state, only basket size differing, so the true lift is 1.0 everywhere:

    need-state          mean size   item share   basket incidence
    top-up                    1.4        1.013              0.311
    normal                    3.6        1.002              0.775
    big shop                  9.0        1.000              1.919

Item share divides trip size out. Basket incidence manufactures a 6x spread
from nothing but trip size, and would rank need-states by how much households
bought rather than by what they bought. Basket size is already reported
separately as `avg_products_per_basket`; keeping it out of lift is what lets
you tell "this need-state is about barbecues" from "this need-state is large".

`check_lift_formula` in test_pipeline.py pins that null, so a change back to
basket counts fails a test instead of silently reordering every profile.

Two consequences worth knowing:
  * `share_within_need_state` is a share of ITEMS, and sums to exactly 1.0
    across all of a need-state's products. Basket incidence would sum to the
    average basket size instead.
  * lift has no ceiling. Basket incidence caps at 1/P(product), which would
    quietly bound `pct_lift_over_3` — the headline in evaluate_run.py.

WHERE THE DATA COMES FROM
─────────────────────────
Everything is read from files already on disk. The .sql files in data/ are
documentation of how the warehouse extracts were produced; they are not run
here and there is no warehouse connection.

    basket -> products      the `baskets_<tag>` DuckDB table, which already
                            holds `products` as a LIST per basket and, more
                            importantly, already holds the `basket_id` built
                            the same way Stage 2's labels key on. Rebuilding
                            that key from the raw parquet would risk a join
                            that silently matches nothing.
    basket -> need-state    data/output/basket_need_state_clusters.parquet
    tpnb  -> tpna           PIPELINE_TPNB_TO_TPNA_MAPPING   (parquet)
    tpna  -> description    PIPELINE_PRODUCT_ATTRIBUTES_TPNA (parquet)

The aggregation runs inside DuckDB, streaming. At 57.1M baskets the exploded
basket-product relation is on the order of a billion rows; it is never
materialised in Python.

USAGE
─────
    python -u profile_need_states.py
    python -u profile_need_states.py --top-n 15 --min-product-baskets 500
    python -u profile_need_states.py --column need_state_cluster_gmm
"""

import argparse
import os

import pandas as pd

import config
import duckdb_manager
from cluster_basket_embeddings import UNCLUSTERED, fmt_duration, progress_step

OUTPUT_DIR = config.OUTPUT_DIR
CLUSTERS_PATH = os.path.join(OUTPUT_DIR, "basket_need_state_clusters.parquet")


def outputs_for(clusters_path: str):
    """
    Profile outputs named after the label file they describe.

    A fixed pair of filenames meant every run overwrote the last — the
    full-population profiles were destroyed by the first <=10-item run, and the
    only way to compare the two afterwards was numbers pasted into a chat. The
    edge and label caches already encode their settings; these now match, so
    runs accumulate instead of replacing each other.

        basket_need_state_clusters_k10_onedir_max10_r1p5.parquet
          -> need_state_profiles_k10_onedir_max10_r1p5.parquet
          -> need_state_summary_k10_onedir_max10_r1p5.parquet
    """
    stem = os.path.splitext(os.path.basename(clusters_path))[0]
    stem = stem.replace("basket_need_state_clusters", "").strip("_")
    suffix = f"_{stem}" if stem else ""
    return (os.path.join(OUTPUT_DIR, f"need_state_profiles{suffix}.parquet"),
            os.path.join(OUTPUT_DIR, f"need_state_summary{suffix}.parquet"))


def _glob(path: str) -> str:
    """
    Each configured input may be a single .parquet OR a folder of part-files
    (the Databricks/Spark export shape). DuckDB's read_parquet takes a glob
    for the second case.
    """
    return os.path.join(path, "*.parquet") if os.path.isdir(path) else path


def _columns_of(con, relation: str) -> set:
    """Actual column names, so a missing optional column degrades instead of crashing."""
    return set(con.execute(f"SELECT * FROM {relation} LIMIT 0").df().columns)


def build_product_lookup(con) -> str:
    """
    tpnb -> description + commercial hierarchy, as a temp view.

    Returns the view name. Descriptions are NOT in product_embeddings.parquet
    (that file carries tpnb + embedding only), so they come from the two raw
    lookup extracts, joined tpnb -> tpna -> description.
    """
    mapping = _glob(config.TPNB_TO_TPNA_MAPPING)
    lookup = _glob(config.PRODUCT_ATTRIBUTES_TPNA)

    lookup_cols = _columns_of(con, f"read_parquet('{lookup}')")
    # commercial_hierarchy_department is the useful rollup for naming a
    # need-state, but treat it as optional — the extract may predate it.
    dept = ("commercial_hierarchy_department"
            if "commercial_hierarchy_department" in lookup_cols else "NULL")
    desc = "description" if "description" in lookup_cols else "CAST(tpna AS VARCHAR)"

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW product_lookup AS
        SELECT CAST(m.tpnb AS VARCHAR)  AS tpnb,
               MAX(l.{desc})            AS description,
               MAX({dept})              AS department
        FROM read_parquet('{mapping}') m
        LEFT JOIN read_parquet('{lookup}') l
               ON CAST(m.tpna AS VARCHAR) = CAST(l.tpna AS VARCHAR)
        GROUP BY 1
    """)
    n = con.execute("SELECT COUNT(*) FROM product_lookup").fetchone()[0]
    has_desc = con.execute(
        "SELECT COUNT(*) FROM product_lookup WHERE description IS NOT NULL"
    ).fetchone()[0]
    if not n:
        raise SystemExit(
            f"product lookup is empty — nothing joined between\n"
            f"  {mapping}\nand\n  {lookup}\n"
            f"Without it every need-state label would be a bare tpnb. Check both "
            f"paths exist and that their `tpna` columns overlap."
        )
    print(f"  product lookup: {n:,} tpnb, {has_desc:,} with a description "
          f"({has_desc / n:.1%})")
    if has_desc < n * 0.5:
        print(f"  WARNING: over half the products have no description. Labels will "
              f"mostly read as raw tpnb numbers.")
    return "product_lookup"


def build_counts(con, baskets_table: str, cluster_column: str):
    """
    Per (need-state, product) basket counts, and the population baseline.

    BOTH SIDES COUNT THE SAME BASKETS — the labelled ones.

    The baseline used to scan the whole table with no cluster filter. That is
    harmless when a run labels every basket and badly wrong when it does not:
    the <=10-item experiment clustered 23.3M of 57.1M baskets and scored them
    against a baseline that still contained the 33.8M big baskets it had
    deliberately excluded, so every need-state in it was measured against a
    population it was not drawn from. Its headline lift — 5.03 against the full
    population's 1.95, the number that motivated the size split — is not
    comparable to anything until both runs are re-profiled through this code.

    One streaming pass each. The exploded relation — one row per basket per
    product, ~1B rows at full scale — exists only inside DuckDB's pipeline;
    nothing here pulls it into Python.
    """
    with progress_step("counting products per need-state", 1, 2):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE ns_product_counts AS
            SELECT ns, tpnb, COUNT(*) AS n_baskets
            FROM (
                SELECT c."{cluster_column}"        AS ns,
                       CAST(UNNEST(b.products) AS VARCHAR) AS tpnb
                FROM "{baskets_table}" b
                JOIN clusters c ON c.basket_id = b.basket_id
                WHERE c."{cluster_column}" <> {UNCLUSTERED}
            )
            GROUP BY ns, tpnb
        """)

    with progress_step("counting products across the labelled population", 2, 2):
        # Same JOIN and same filter as ns_product_counts above — that identity
        # is the whole point, and the item-row assertion below enforces it.
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE product_baseline AS
            SELECT tpnb, COUNT(*) AS n_baskets
            FROM (
                SELECT CAST(UNNEST(b.products) AS VARCHAR) AS tpnb
                FROM "{baskets_table}" b
                JOIN clusters c ON c.basket_id = b.basket_id
                WHERE c."{cluster_column}" <> {UNCLUSTERED}
            )
            GROUP BY tpnb
        """)

    pairs = con.execute("SELECT COUNT(*) FROM ns_product_counts").fetchone()[0]
    print(f"  {pairs:,} (need-state, product) pairs")

    # Count what was left out, out loud. Silence here is exactly what let the
    # mismatched baseline survive a full production run and a stakeholder
    # comparison without anything in the log mentioning it.
    n_all = con.execute(f'SELECT COUNT(*) FROM "{baskets_table}"').fetchone()[0]
    n_base = con.execute(f"""
        SELECT COUNT(*)
        FROM "{baskets_table}" b
        JOIN clusters c ON c.basket_id = b.basket_id
        WHERE c."{cluster_column}" <> {UNCLUSTERED}
    """).fetchone()[0]
    pct = f" ({n_base / n_all:.1%})" if n_all else ""
    print(f"  baseline population: {n_base:,} of {n_all:,} baskets in {baskets_table}{pct}")
    if n_all - n_base:
        print(f"  {n_all - n_base:,} baskets carry no need-state label and are excluded from "
              f"BOTH sides of the lift ratio.\n"
              f"  Lift in this run therefore describes the labelled subpopulation only, and "
              f"is NOT\n  comparable to a run that labelled a different share of the table.")

    # The two tables are built from the same JOIN, so they must cover exactly
    # the same (basket, product) rows. Anything else means the numerator and
    # the denominator are describing different populations again — the bug
    # this function was rewritten to close. Fail before the profile is written,
    # not after someone quotes it.
    ns_items = con.execute("SELECT SUM(n_baskets) FROM ns_product_counts").fetchone()[0] or 0
    base_items = con.execute("SELECT SUM(n_baskets) FROM product_baseline").fetchone()[0] or 0
    if ns_items != base_items:
        raise SystemExit(
            f"lift numerator and denominator cover different populations: "
            f"{ns_items:,} item-rows across need-states vs {base_items:,} in the baseline.\n"
            f"Both queries in build_counts() must use the same JOIN and the same "
            f"'<> {UNCLUSTERED}' filter; one of them has been changed."
        )
    print(f"  numerator and baseline cover the same {base_items:,} item-rows — OK")


def build_summary(con, baskets_table: str, cluster_column: str) -> pd.DataFrame:
    """One row per need-state: size, households, average basket size."""
    return con.execute(f"""
        SELECT c."{cluster_column}"              AS need_state,
               COUNT(*)                          AS n_baskets,
               COUNT(DISTINCT b.household_number) AS n_households,
               AVG(len(b.products))              AS avg_products_per_basket
        FROM "{baskets_table}" b
        JOIN clusters c ON c.basket_id = b.basket_id
        WHERE c."{cluster_column}" <> {UNCLUSTERED}
        GROUP BY 1
        ORDER BY n_baskets DESC
    """).df()


def build_profiles(con, top_n: int, min_product_baskets: int) -> pd.DataFrame:
    """
    Top-N products per need-state by lift, with the raw shares alongside.

    min_product_baskets is the guard that stops lift being won by a product
    appearing in a handful of baskets — with 154,597 products and 356
    need-states, unfiltered lift surfaces noise almost every time.

    `ns_product_rows` and `total` are ITEM counts (sum of basket sizes), not
    basket counts. That is deliberate and load-bearing — dividing by baskets
    instead makes every lift in a need-state scale with its trip size. The
    module docstring has the measured null; test_pipeline.py pins it.
    """
    total = con.execute("SELECT SUM(n_baskets) FROM product_baseline").fetchone()[0]
    return con.execute(f"""
        WITH ns_totals AS (
            SELECT ns, SUM(n_baskets) AS ns_product_rows
            FROM ns_product_counts GROUP BY ns
        ),
        scored AS (
            SELECT p.ns                                  AS need_state,
                   p.tpnb,
                   p.n_baskets                           AS n_baskets_in_need_state,
                   b.n_baskets                           AS n_baskets_overall,
                   (p.n_baskets * 1.0 / t.ns_product_rows)
                     / (b.n_baskets * 1.0 / {total})     AS lift,
                   p.n_baskets * 1.0 / t.ns_product_rows AS share_within_need_state
            FROM ns_product_counts p
            JOIN ns_totals t        ON t.ns = p.ns
            JOIN product_baseline b ON b.tpnb = p.tpnb
            WHERE p.n_baskets >= {min_product_baskets}
        )
        SELECT s.*, l.description, l.department,
               ROW_NUMBER() OVER (PARTITION BY s.need_state ORDER BY s.lift DESC) AS rank_by_lift
        FROM scored s
        LEFT JOIN product_lookup l ON l.tpnb = s.tpnb
        QUALIFY rank_by_lift <= {top_n}
        ORDER BY s.need_state, rank_by_lift
    """).df()


# Bands for the insight tables. The size edges are on a need-state's AVERAGE
# basket size, so they are ranges of a mean rather than integer item counts.
SIZE_BANDS = ([0, 2, 5, 10, 20, 50, float("inf")],
              ["<=2", "2-5", "5-10", "10-20", "20-50", ">50"])
LIFT_BANDS = ([-float("inf"), 1.5, 3, 5, 10, float("inf")],
              ["< 1.5", "1.5-3", "3-5", "5-10", ">= 10"])
COVERAGE_BANDS = ([0, 0.10, 0.25, 0.50, 0.75, 1.01],
                  ["< 10%", "10-25%", "25-50%", "50-75%", ">= 75%"])
DEPT_BANDS = ([0, 1, 3, 6, float("inf")], ["1", "2-3", "4-6", "7+"])


def enrich_summary(summary: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Per-need-state columns derived from its profile, for the insight tables and
    for anyone reading the summary parquet directly.

    THE ONE THAT MATTERS IS top_product_coverage
    ────────────────────────────────────────────
    `share_within_need_state` is a share of ITEMS, so it is mechanically capped
    by basket size: if EVERY basket in a need-state contains milk, milk's item
    share is only 1/avg_basket_size. A 40-item need-state therefore cannot put
    any product above 2.5%, and its shares are not comparable with a 3-item
    need-state's.

        top_product_coverage = top_share * avg_products_per_basket
                             = the fraction of this need-state's baskets that
                               contain its signature product

    That divides the cap back out, giving a 0-1 number that IS comparable
    across sizes. It is deliberately a separate column and not part of lift —
    lift answers "what do these people buy", coverage answers "how much of this
    need-state does that actually describe". Keeping them apart is what lets
    you tell a genuine occasion from a large basket.

    This is the number behind "a high lift on a tiny share is a marker, not a
    description": lift 40 at 4% coverage is a curiosity, lift 6 at 80% is a
    definition.
    """
    # head(1) rather than first(): GroupBy.first() returns the first NON-NULL
    # value per column independently, so a null description would silently pull
    # its lift and share from different rows.
    ranked = profiles.sort_values(["need_state", "rank_by_lift"])
    top1 = ranked.groupby("need_state", as_index=False).head(1).set_index("need_state")

    stats = profiles.groupby("need_state")["lift"].agg(
        max_lift="max", avg_lift="mean", median_lift="median")
    summary = summary.drop(columns=[c for c in stats.columns if c in summary.columns])
    summary = summary.merge(stats, left_on="need_state", right_index=True, how="left")

    summary = summary.merge(
        top1["share_within_need_state"].rename("top_share"),
        left_on="need_state", right_index=True, how="left")
    summary["top_product_coverage"] = (
        summary["top_share"] * summary["avg_products_per_basket"]).clip(upper=1.0)

    depts = profiles.groupby("need_state")["department"].nunique().rename("n_departments_top_n")
    summary = summary.merge(depts, left_on="need_state", right_index=True, how="left")

    summary["size_band"] = pd.cut(summary["avg_products_per_basket"],
                                  bins=SIZE_BANDS[0], labels=SIZE_BANDS[1],
                                  include_lowest=True)
    return summary


def _band_table(summary: pd.DataFrame, bands, title: str, first_header: str, note: str = ""):
    """One banded cross-tab. Same columns every time so the tables read together."""
    total = summary["n_baskets"].sum()
    g = summary.groupby(bands, observed=False).agg(
        n_ns=("need_state", "size"),
        n_baskets=("n_baskets", "sum"),
        med_lift=("max_lift", "median"),
        pct3=("max_lift", lambda s: (s > 3).mean()),
        med_size=("avg_products_per_basket", "median"),
        med_cov=("top_product_coverage", "median"),
    )
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {first_header:<14}{'need-states':>12}{'baskets':>15}{'% of pop':>10}"
          f"{'med lift':>10}{'lift>3':>8}{'med size':>10}{'coverage':>10}")
    print("  " + "-" * 89)
    for band, r in g.iterrows():
        if not r["n_ns"]:
            print(f"  {str(band):<14}{0:>12}{'-':>15}{'-':>10}{'-':>10}{'-':>8}{'-':>10}{'-':>10}")
            continue
        print(f"  {str(band):<14}{int(r['n_ns']):>12,}{int(r['n_baskets']):>15,}"
              f"{r['n_baskets'] / total:>10.1%}{r['med_lift']:>10.2f}{r['pct3']:>8.0%}"
              f"{r['med_size']:>10.1f}{r['med_cov']:>10.1%}")


def print_insights(summary: pd.DataFrame, top_n: int):
    """
    Four cross-tabs plus the headline numbers, all from columns already
    computed. Every one exists to answer a question the per-need-state listing
    cannot: it shows 15 need-states out of 356 and hides the shape of the rest.
    """
    print("\n" + "=" * 91)
    print("INSIGHTS — the shape of this run")
    print("=" * 91)
    print("  Columns are the same in every table. `med lift` is the median across "
          "need-states of\n  each one's BEST product lift; `coverage` is the median "
          "share of a need-state's baskets\n  that actually contain its signature "
          "product. Read lift and coverage together.")

    _band_table(summary, summary["size_band"],
                "1. BY BASKET SIZE — does trip size still drive the metric?",
                "avg basket",
                "Lift is an item share ratio, so it should NOT slope with size. A strong "
                "slope here\n  means either real structure (small trips genuinely are more "
                "distinctive) or a leak.")

    _band_table(summary,
                pd.cut(summary["max_lift"], bins=LIFT_BANDS[0], labels=LIFT_BANDS[1]),
                "2. BY HOW WELL CHARACTERISED — and how much of the population is there",
                "max lift",
                "The % of pop column is the honest stakeholder number. A run where 90% of "
                "need-states\n  clear lift 3 but the weak ones hold half the baskets is not "
                "a good run.")

    _band_table(summary,
                pd.cut(summary["top_product_coverage"], bins=COVERAGE_BANDS[0],
                       labels=COVERAGE_BANDS[1], include_lowest=True),
                "3. BY SIGNATURE STRENGTH — marker, or description?",
                "top product",
                "What share of a need-state's baskets contain its top-lift product. Low "
                "coverage with\n  high lift is a marker on a niche product, not a "
                "description of the occasion.")

    # department is optional — build_product_lookup falls back to NULL when the
    # attributes extract predates commercial_hierarchy_department. nunique() is
    # then 0 everywhere and pd.cut drops it, so say why the table is missing
    # rather than printing four empty rows.
    if summary["n_departments_top_n"].fillna(0).sum() == 0:
        print("\n4. BY DEPARTMENT SPREAD — skipped: no department in the product lookup "
              "(commercial_hierarchy_department\n   is absent from "
              "PIPELINE_PRODUCT_ATTRIBUTES_TPNA, so there is nothing to spread over).")
    else:
        _band_table(summary,
                    pd.cut(summary["n_departments_top_n"], bins=DEPT_BANDS[0],
                           labels=DEPT_BANDS[1]),
                    f"4. BY DEPARTMENT SPREAD — is the top-{top_n} one aisle or a supermarket?",
                    "departments",
                    f"Distinct departments among the top-{top_n} products. A need-state whose "
                    f"signature spans\n  most of the store is a region of embedding space, not "
                    f"a shopping occasion.")

    # Headline scalars. The weighted-vs-unweighted gap is the one that tends to
    # surprise: it says whether the weak need-states are the big ones.
    total = summary["n_baskets"].sum()
    unweighted = summary["max_lift"].median()
    over3 = summary["max_lift"] > 3
    pop_over3 = summary.loc[over3, "n_baskets"].sum() / total if total else float("nan")
    # Spearman by hand — pandas routes method="spearman" through scipy, and
    # rank-then-pearson is the same number with no extra dependency.
    rho = summary["avg_products_per_basket"].rank().corr(summary["max_lift"].rank())

    print("\n" + "-" * 91)
    print("HEADLINES")
    print(f"  need-states                                  {len(summary):>12,}")
    print(f"  labelled baskets                             {int(total):>12,}")
    print(f"  median max lift (per need-state)             {unweighted:>12.2f}")
    print(f"  need-states with max lift > 3                {over3.mean():>11.0%}")
    print(f"  BASKETS in a need-state with max lift > 3    {pop_over3:>11.0%}   "
          f"<- the number to quote")
    print(f"  median signature coverage                    "
          f"{summary['top_product_coverage'].median():>11.0%}")
    print(f"  Spearman(avg basket size, max lift)          {rho:>12.2f}")
    if pd.notna(rho) and abs(rho) >= 0.4:
        print(f"    NOTE: lift still tracks basket size at rho={rho:.2f}. Item-share lift "
              f"divides trip\n    size out arithmetically, so this is either genuine "
              f"(small trips are more focused)\n    or the clustering has largely "
              f"rediscovered basket size. Compare table 1's med lift\n    against table 1's "
              f"coverage: if coverage slopes the same way, it is genuine.")
    if pd.notna(pop_over3) and pop_over3 < over3.mean() - 0.10:
        print(f"    NOTE: {over3.mean():.0%} of need-states clear lift 3 but only "
              f"{pop_over3:.0%} of baskets sit in one.\n    The weakest need-states are the "
              f"largest — the clusters that matter most are the\n    least characterised.")


def label_from_profile(profiles: pd.DataFrame, n_terms: int = 3) -> pd.DataFrame:
    """
    A short human-readable label per need-state, from its highest-lift products.

    Crude on purpose — it concatenates the top descriptions rather than trying
    to name the occasion. Naming is a judgement call for whoever reads these;
    this exists so the graph has something other than an integer on each node.
    """
    # Plain iteration over groups rather than groupby().apply(). The apply form
    # needs `include_groups=`, which only exists from pandas 2.2 and raises a
    # TypeError before it — and there are 356 groups here, so the loop costs
    # nothing and works on any version.
    ordered = profiles.sort_values(["need_state", "rank_by_lift"])
    rows = []
    for need_state, group in ordered.groupby("need_state", sort=True):
        terms = [str(d).strip() for d in group.head(n_terms)["description"].tolist()
                 if isinstance(d, str) and d.strip()]
        rows.append({
            "need_state": need_state,
            "label": " / ".join(terms) if terms else "(no description available)",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Describe each need-state by the products over-represented in it."
    )
    parser.add_argument("--clusters", default=CLUSTERS_PATH)
    parser.add_argument("--column", default="need_state_cluster",
                        help="which label column to profile "
                             "(need_state_cluster | need_state_cluster_gmm)")
    parser.add_argument("--dataset-tag", default=config.TRAIN_DATASET_TAG)
    parser.add_argument("--top-n", type=int, default=10,
                        help="products kept per need-state (default 10)")
    parser.add_argument("--min-product-baskets", type=int, default=200,
                        help="a product must appear in at least this many baskets of a "
                             "need-state to be eligible — without it, lift is won by "
                             "products seen a handful of times (default 200)")
    parser.add_argument("--show", type=int, default=15,
                        help="how many need-states to print (default 15)")
    args = parser.parse_args()

    con = duckdb_manager.get_connection()
    baskets_table = f"baskets_{args.dataset_tag}"

    with progress_step(f"loading {args.clusters}"):
        clusters = pd.read_parquet(args.clusters)
    if args.column not in clusters.columns:
        raise SystemExit(f"{args.clusters} has no column '{args.column}'. "
                         f"Available: {list(clusters.columns)}")
    print(f"  {len(clusters):,} labelled baskets, "
          f"{clusters[args.column].nunique():,} distinct {args.column}")
    con.register("clusters", clusters[["basket_id", args.column]])

    build_product_lookup(con)
    build_counts(con, baskets_table, args.column)

    with progress_step("summarising need-states"):
        summary = build_summary(con, baskets_table, args.column)
    with progress_step(f"ranking top-{args.top_n} products by lift"):
        profiles = build_profiles(con, args.top_n, args.min_product_baskets)

    labels = label_from_profile(profiles)
    summary = summary.merge(labels, left_on="need_state", right_on="need_state", how="left")
    summary["share_of_baskets"] = summary["n_baskets"] / summary["n_baskets"].sum()

    # How many products actually cleared --min-product-baskets, per need-state.
    # Without this there is no way to tell a real top-10 signature from a
    # cluster that had exactly 10 eligible products and no choice about which.
    # Those look identical in the profile and are not: comparing two
    # low-support need-states found them sharing a Jaccard of 1.0, which read
    # as catastrophic over-splitting and was actually the threshold leaking in.
    counts = profiles.groupby("need_state").size().rename("n_products_profiled")
    summary = summary.merge(counts, left_on="need_state", right_index=True, how="left")
    summary["n_products_profiled"] = summary["n_products_profiled"].fillna(0).astype(int)
    thin = int((summary["n_products_profiled"] < args.top_n).sum())

    summary = enrich_summary(summary, profiles)

    profiles_path, summary_path = outputs_for(args.clusters)
    profiles.to_parquet(profiles_path, index=False)
    # size_band stays an ordered Categorical in memory so the insight tables
    # come out in band order, but it is written as plain text: the export
    # concatenates summaries across runs, and concatenating Categoricals with
    # one run's column missing turns the whole column to NaN.
    on_disk = summary.copy()
    on_disk["size_band"] = summary["size_band"].astype(str).where(
        summary["size_band"].notna())
    on_disk.to_parquet(summary_path, index=False)
    print(f"\nSaved {profiles_path} ({len(profiles):,} rows)")
    print(f"Saved {summary_path} ({len(summary):,} need-states)")
    if thin:
        print(f"  NOTE: {thin:,} need-states have fewer than {args.top_n} products above "
              f"the {args.min_product_baskets:,}-basket threshold. Their profiles are "
              f"whatever qualified, not a ranking — treat those labels as weak.")

    print("\n" + "=" * 78)
    print(f"LARGEST {args.show} NEED-STATES")
    print("=" * 78)
    for row in summary.head(args.show).itertuples():
        print(f"\n  need-state {row.need_state}  —  {row.n_baskets:,} baskets "
              f"({row.share_of_baskets:.2%}), {row.n_households:,} households, "
              f"{row.avg_products_per_basket:.1f} products/basket")
        top = profiles[profiles["need_state"] == row.need_state].head(5)
        for p in top.itertuples():
            desc = p.description if isinstance(p.description, str) else f"tpnb {p.tpnb}"
            print(f"      lift {p.lift:6.1f}  {desc[:58]}")

    print_insights(summary, args.top_n)

    print("\n" + "=" * 78)
    print("Read need_state_profiles.parquet for the full ranking. `lift` is the "
          "column\nthat distinguishes need-states: this product's share of the "
          "need-state's ITEMS\nover its share of all items. `share_within_need_state` is "
          "that first share on\nits own, and sums to 1.0 across a need-state's products. "
          "A high lift on a tiny\nshare is a marker, not a description.")


if __name__ == "__main__":
    main()
