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

    lift = P(product | need-state) / P(product)

A lift of 8 means "baskets in this need-state contain this product eight times
more often than baskets in general". That is what distinguishes need-states
from each other. Both numbers are reported, because lift alone can crown a
product that appears in forty baskets — hence MIN_PRODUCT_BASKETS.

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
PROFILES_PATH = os.path.join(OUTPUT_DIR, "need_state_profiles.parquet")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "need_state_summary.parquet")


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

    with progress_step("counting products across the whole population", 2, 2):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE product_baseline AS
            SELECT tpnb, COUNT(*) AS n_baskets
            FROM (SELECT CAST(UNNEST(products) AS VARCHAR) AS tpnb
                  FROM "{baskets_table}")
            GROUP BY tpnb
        """)

    pairs = con.execute("SELECT COUNT(*) FROM ns_product_counts").fetchone()[0]
    print(f"  {pairs:,} (need-state, product) pairs")


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

    profiles.to_parquet(PROFILES_PATH, index=False)
    summary.to_parquet(SUMMARY_PATH, index=False)
    print(f"\nSaved {PROFILES_PATH} ({len(profiles):,} rows)")
    print(f"Saved {SUMMARY_PATH} ({len(summary):,} need-states)")

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

    print("\n" + "=" * 78)
    print("Read need_state_profiles.parquet for the full ranking. `lift` is the "
          "column\nthat distinguishes need-states; `share_within_need_state` tells you "
          "how much\nof the need-state a product actually accounts for. A high lift on "
          "a tiny\nshare is a marker, not a description.")


if __name__ == "__main__":
    main()
