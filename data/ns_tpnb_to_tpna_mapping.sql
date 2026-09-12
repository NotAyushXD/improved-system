-- ============================================================
-- 02_tpnb_to_tpna_mapping.sql
--
-- Feeds  : build_product_embeddings.py's tpnb -> tpna mapping
-- Grain  : one row per tpnb
-- Cols required downstream: tpnb, tpna
--
-- Materializes into a table so it can be downloaded/exported as parquet —
-- save the export as data/tpnb_to_tpna_mapping.parquet.
--
-- Source: product.product — tpnb and tpna sit on the same row here, so no
-- separate mapping table is needed.
--
-- No purchase-recency filter here on purpose — this stays the FULL mapping.
-- build_product_embeddings.py inner-joins this against sql/01's output
-- (which IS scoped to last-year-purchased tpna's only), so any tpnb whose
-- tpna wasn't purchased last year gets dropped there automatically. Keeping
-- this one unfiltered means it stays reusable if the "last year" scope
-- changes later without needing to re-pull this table.
-- ============================================================

DROP TABLE IF EXISTS lab_customer_value_analytics.ns_tpnb_to_tpna_mapping;

CREATE TABLE lab_customer_value_analytics.ns_tpnb_to_tpna_mapping
USING PARQUET
AS
SELECT DISTINCT
    CAST(tpnb AS STRING) AS tpnb,
    CAST(tpna AS STRING) AS tpna
FROM product.product
WHERE tpnb IS NOT NULL
  AND tpna IS NOT NULL
  AND COALESCE(is_deleted, FALSE)  = FALSE
  AND COALESCE(is_archived, FALSE) = FALSE
;

-- ============================================================
-- VALIDATION
-- ============================================================

SELECT
    COUNT(*)             AS row_count,
    COUNT(DISTINCT tpnb) AS distinct_tpnb,
    COUNT(DISTINCT tpna) AS distinct_tpna
FROM lab_customer_value_analytics.ns_tpnb_to_tpna_mapping
;