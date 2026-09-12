-- ============================================================
-- 01_product_attributes_tpna.sql
--
-- Feeds  : build_product_embeddings.py's item_lookup
-- Grain  : one row per tpna (style) — ONLY tpna's with at least one tpnb
--          actually bought in the period below (see purchased_tpnb CTE).
--          Catalog products with zero recent purchases are skipped
--          entirely — no point embedding something no basket needs.
-- Cols required downstream: tpna, description, brand,
--   commercial_hierarchy_department, commercial_hierarchy_class,
--   commercial_hierarchy_subclass
--
-- Materializes into a table (rather than a plain SELECT) so it can be
-- checked in a notebook and then downloaded/exported as parquet — save the
-- export as data/product_attributes_tpna.parquet.
--
-- Source: product.product — real table, no placeholders needed for the
-- attributes themselves. tpna_description is used where populated (already
-- at the right grain), falling back to a mode-aggregated description across
-- sibling tpnbs where it's blank, same pattern as your own
-- cge_item_lookup_tpna's attr_counts CTE.
--
-- IMPORTANT: the <START_YEAR_PERIOD> / <END_YEAR_PERIOD> placeholders below
-- should be the SAME values you use in
-- ns_household_tpnb_week_agg_train.sql. Different ranges here vs. there
-- means some products your training baskets actually contain would have no
-- embedding at all (GraphBuilder.py silently falls back to a zero vector for
-- those, rather than erroring — worth avoiding, not relying on).
-- ============================================================

DROP TABLE IF EXISTS lab_customer_value_analytics.ns_item_lookup_tpna;

CREATE TABLE lab_customer_value_analytics.ns_item_lookup_tpna
USING PARQUET
AS

WITH purchased_tpnb AS (
    -- <<< CONFIRM: same period range as ns_household_tpnb_week_agg_train.sql.
    -- Run diagnostic_check_periods.sql first if you haven't confirmed real
    -- values yet.
    SELECT DISTINCT CAST(tpnb AS STRING) AS tpnb
    FROM lab_customer_value_analytics.cltv_hh_metrics_tpnb_base
    WHERE (year_number * 100 + period_number)
          BETWEEN 202603 AND 202604   -- <<< must match sql/04
),

live_products AS (
    SELECT
        CAST(tpna AS STRING)                                   AS tpna,
        CAST(tpnb AS STRING)                                   AS tpnb,
        NULLIF(TRIM(CAST(tpna_description AS STRING)), '')     AS tpna_description,
        CAST(description AS STRING)                            AS description,
        COALESCE(NULLIF(TRIM(CAST(brand AS STRING)), ''), 'UNKNOWN') AS brand,
        CAST(commercial_hierarchy_department_name AS STRING)   AS commercial_hierarchy_department,
        CAST(commercial_hierarchy_class_name AS STRING)        AS commercial_hierarchy_class,
        CAST(commercial_hierarchy_subclass_name AS STRING)     AS commercial_hierarchy_subclass,
        CAST(load_date_time AS TIMESTAMP)                      AS load_date_time
    FROM product.product
    WHERE tpna IS NOT NULL
      AND COALESCE(is_deleted, FALSE)  = FALSE
      AND COALESCE(is_archived, FALSE) = FALSE
),

-- Only keep tpna's that have at least one sibling tpnb actually bought in
-- the target period — this is the "last year only" scope
purchased_tpna AS (
    SELECT DISTINCT lp.tpna
    FROM live_products lp
    INNER JOIN purchased_tpnb pt
            ON lp.tpnb = pt.tpnb
),

live_products_scoped AS (
    SELECT lp.*
    FROM live_products lp
    INNER JOIN purchased_tpna scope
            ON lp.tpna = scope.tpna
),

-- Mode-aggregate hierarchy + fallback description per tpna, same pattern
-- as your cge_item_lookup_tpna's attr_counts CTE
attr_counts AS (
    SELECT
        tpna, description, brand,
        commercial_hierarchy_department,
        commercial_hierarchy_class,
        commercial_hierarchy_subclass,
        COUNT(*) AS n_occurrences,
        ROW_NUMBER() OVER (PARTITION BY tpna ORDER BY COUNT(*) DESC) AS rn
    FROM live_products_scoped
    GROUP BY tpna, description, brand,
             commercial_hierarchy_department, commercial_hierarchy_class, commercial_hierarchy_subclass
),

-- tpna_description, when populated, is already the right grain — prefer it
-- over the mode-aggregated tpnb-level description above
tpna_desc AS (
    SELECT tpna, tpna_description,
           ROW_NUMBER() OVER (PARTITION BY tpna ORDER BY load_date_time DESC) AS rn
    FROM live_products_scoped
    WHERE tpna_description IS NOT NULL
)

SELECT
    a.tpna,
    COALESCE(d.tpna_description, a.description)   AS description,
    a.brand,
    a.commercial_hierarchy_department,
    a.commercial_hierarchy_class,
    a.commercial_hierarchy_subclass
FROM attr_counts a
LEFT JOIN tpna_desc d
       ON a.tpna = d.tpna AND d.rn = 1
WHERE a.rn = 1
;

-- ============================================================
-- VALIDATION
-- ============================================================

SELECT
    COUNT(*)                AS row_count,
    COUNT(DISTINCT tpna)     AS distinct_tpna,
    SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END) AS null_description_count
FROM lab_customer_value_analytics.ns_item_lookup_tpna
;