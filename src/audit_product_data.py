"""
audit_product_data.py

Is the product reference data good enough to trust the need-state labels?

This exists because `tpnb 54739758` — present in 1,767,393 household-weeks,
3.1% of the population — carries the description "PENDRIVE FLASH DRIVE 12GB
SLIQ" and the department "FRESH FRUIT/VEG/SALAD". Those cannot both be true.
It maps to exactly one tpna, so the contradiction is in the source
`product.product` extract, not in how anything here joins it.

That matters twice over:

  1. Need-state labels come from `description`. A wrong description gives a
     cluster a wrong name, and nobody reading the name can tell.
  2. Product EMBEDDINGS are built from description text (see
     build_product_embeddings.py). A wrong description gave that product a
     wrong 384-dim vector, which fed the GNN, the basket embeddings and the
     clustering. The commercial hierarchy is embedded alongside and provides a
     corrective signal, so this degrades rather than destroys — but the extent
     of it is unknown until measured.

The single most informative output is the frequency table: the products in the
most baskets are the ones driving the embedding space. If the top sellers read
as milk, bananas and bread with sensible departments, the data is broadly
sound and 54739758 is an oddity. If they read as pendrives and shoe racks,
the reference data is broken and every label downstream is suspect.

USAGE
    python -u audit_product_data.py
    python -u audit_product_data.py --top 50
"""

import argparse
import os

import config
import duckdb_manager


def _glob(path: str) -> str:
    """Single .parquet file, or a folder of part-files (the Spark export shape)."""
    return os.path.join(path, "*.parquet") if os.path.isdir(path) else path


def main():
    parser = argparse.ArgumentParser(
        description="Check whether product descriptions can be trusted."
    )
    parser.add_argument("--top", type=int, default=30,
                        help="how many most-frequent products to show (default 30)")
    parser.add_argument("--dataset-tag", default=config.TRAIN_DATASET_TAG)
    args = parser.parse_args()

    con = duckdb_manager.get_connection()
    baskets = f"baskets_{args.dataset_tag}"
    mapping = _glob(config.TPNB_TO_TPNA_MAPPING)
    lookup = _glob(config.PRODUCT_ATTRIBUTES_TPNA)

    print("=" * 78)
    print("1. DOES ONE tpnb MAP TO MORE THAN ONE tpna?")
    print("=" * 78)
    print("   If it does, any single description picked per tpnb is arbitrary.")
    fanout = con.execute(f"""
        SELECT n_tpna, COUNT(*) AS n_tpnb
        FROM (SELECT tpnb, COUNT(DISTINCT tpna) AS n_tpna
              FROM read_parquet('{mapping}') GROUP BY tpnb)
        GROUP BY 1 ORDER BY 1
    """).df()
    print(fanout.to_string(index=False))
    multi = int(fanout.loc[fanout.n_tpna > 1, "n_tpnb"].sum()) if len(fanout) else 0
    print(f"\n   {multi:,} tpnb map to more than one tpna."
          f" {'Descriptions are ambiguous for these.' if multi else ' Mapping is clean.'}")

    print()
    print("=" * 78)
    print(f"2. THE {args.top} PRODUCTS IN THE MOST BASKETS")
    print("=" * 78)
    print("   These dominate the embedding space. Do they read like groceries?")
    top = con.execute(f"""
        WITH counts AS (
            SELECT CAST(UNNEST(products) AS VARCHAR) AS tpnb
            FROM "{baskets}"
        ),
        agg AS (
            SELECT tpnb, COUNT(*) AS n_baskets FROM counts GROUP BY tpnb
        ),
        total AS (SELECT COUNT(*) AS n FROM "{baskets}")
        SELECT a.n_baskets,
               ROUND(100.0 * a.n_baskets / (SELECT n FROM total), 2) AS pct_of_baskets,
               a.tpnb,
               MAX(l.description)                     AS description,
               MAX(l.commercial_hierarchy_department) AS department
        FROM agg a
        LEFT JOIN read_parquet('{mapping}') m ON CAST(m.tpnb AS VARCHAR) = a.tpnb
        LEFT JOIN read_parquet('{lookup}')  l ON CAST(l.tpna AS VARCHAR) = CAST(m.tpna AS VARCHAR)
        GROUP BY a.n_baskets, a.tpnb
        ORDER BY a.n_baskets DESC
        LIMIT {args.top}
    """).df()
    print(top.to_string(index=False))

    print()
    print("=" * 78)
    print("3. DESCRIPTION COVERAGE, OVER PRODUCTS THAT ACTUALLY APPEAR")
    print("=" * 78)
    print("   Measured against purchased products, not the whole warehouse catalog —")
    print("   the catalog denominator makes coverage look far worse than it is.")
    cover = con.execute(f"""
        WITH purchased AS (
            SELECT DISTINCT CAST(UNNEST(products) AS VARCHAR) AS tpnb FROM "{baskets}"
        )
        SELECT COUNT(*)                                                   AS purchased_tpnb,
               COUNT(l.description)                                       AS with_description,
               COUNT(l.commercial_hierarchy_department)                   AS with_department
        FROM purchased p
        LEFT JOIN read_parquet('{mapping}') m ON CAST(m.tpnb AS VARCHAR) = p.tpnb
        LEFT JOIN read_parquet('{lookup}')  l ON CAST(l.tpna AS VARCHAR) = CAST(m.tpna AS VARCHAR)
    """).df()
    print(cover.to_string(index=False))
    row = cover.iloc[0]
    if row.purchased_tpnb:
        print(f"\n   description: {row.with_description / row.purchased_tpnb:.1%} of "
              f"purchased products")
        print(f"   department : {row.with_department / row.purchased_tpnb:.1%}")

    print()
    print("=" * 78)
    print("WHAT TO CONCLUDE")
    print("=" * 78)
    print("""   Section 2 is the one that matters. If the most-purchased products read as
   ordinary groceries with matching departments, the reference data is sound
   and tpnb 54739758 is an isolated bad row — note it and move on.

   If several of the top sellers carry descriptions that contradict their
   department, the `description` field is unreliable at scale. Then:
     - need-state labels cannot be trusted as written, and
     - the product embeddings were partly built from wrong text, so the
       clustering itself inherited the noise.
   In that case the fix is upstream, in whoever owns product.product — not
   anything in this pipeline.""")


if __name__ == "__main__":
    main()
