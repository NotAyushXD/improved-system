-- ============================================================
-- 03_product_theme_mapping.sql
--
-- Feeds  : build_product_embeddings.py's theme_name column
-- Grain  : one row per tpnb
-- Cols required downstream: tpnb, theme_name
--
-- Materializes into a table so it can be downloaded/exported as parquet —
-- save the export as data/product_theme_mapping.parquet.
--
-- Sourced from your real IN22915286_UPDATED_PRODUCT_TABLE. Naming note:
-- this is a CATEGORY (category_area_description), not the old JB "theme"
-- concept — no bridge flag exists on this table, so pipeline_main.py's old
-- theme_jb_is_bridge filter has been removed.
--
-- No category filter here — this pulls the mapping for your WHOLE
-- assortment. Add a WHERE clause if you want the need-state model scoped
-- to specific categories, the way your sample query does for its 5.
-- ============================================================

DROP TABLE IF EXISTS lab_customer_value_analytics.ns_product_theme_mapping;

CREATE TABLE lab_customer_value_analytics.ns_product_theme_mapping
USING PARQUET
AS
SELECT DISTINCT
    CAST(tpnb AS STRING)                      AS tpnb,
    CAST(category_area_description AS STRING) AS theme_name
FROM LAB_INSIGHT_CUSTOMER_ANALYTICS.IN22915286_UPDATED_PRODUCT_TABLE
WHERE tpnb IS NOT NULL
  AND category_area_description IS NOT NULL
;

-- ============================================================
-- VALIDATION
-- ============================================================

SELECT
    COUNT(*)                AS row_count,
    COUNT(DISTINCT tpnb)     AS distinct_tpnb,
    COUNT(DISTINCT theme_name) AS distinct_categories
FROM lab_customer_value_analytics.ns_product_theme_mapping
;