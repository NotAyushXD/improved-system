-- ============================================================
-- 04_household_tpnb_period_agg.sql
--
-- Feeds  : pipeline_main.py's tpnb_x_hh -> baskets_theme construction, product_units_avg
-- Grain  : one row per household_number x tpnb x year_number x period_number
-- Cols required downstream: household_number, tpnb, year_number, period_number, quantity
--
-- Materializes into a table so it can be downloaded/exported as parquet —
-- save the export as data/household_tpnb_period_agg.parquet.
--
-- Source: lab_customer_value_analytics.cltv_hh_metrics_tpnb_base — real
-- table, confirmed columns. It's WEEK-grain (year_number, period_number,
-- week_number, household_number, tpnb) — summed across week_number here to
-- reach the period grain the rest of the pipeline expects.
-- ============================================================

DROP TABLE IF EXISTS lab_customer_value_analytics.ns_household_tpnb_period_agg_train;

CREATE TABLE lab_customer_value_analytics.ns_household_tpnb_period_agg_train
USING PARQUET
AS
SELECT
    year_number,
    period_number,
    household_number,
    tpnb,
    SUM(quantity)       AS quantity,
    SUM(orders)         AS orders,          -- not consumed downstream yet, kept for future profiling
    SUM(sales_inc_vat)  AS sales_inc_vat    -- not consumed downstream yet, kept for future profiling
FROM lab_customer_value_analytics.cltv_hh_metrics_tpnb_base
WHERE (year_number * 100 + period_number)
      BETWEEN 202603 AND 202604   -- <<< CONFIRM: training period range, e.g. 202401 AND 202413
  AND household_number IS NOT NULL
  AND tpnb IS NOT NULL
GROUP BY year_number, period_number, household_number, tpnb
;

-- ============================================================
-- VALIDATION
-- ============================================================

SELECT
    MIN(year_number * 100 + period_number) AS min_year_period,
    MAX(year_number * 100 + period_number) AS max_year_period,
    COUNT(*)                        AS row_count,
    COUNT(DISTINCT household_number) AS household_count,
    COUNT(DISTINCT tpnb)             AS product_count
FROM lab_customer_value_analytics.ns_household_tpnb_period_agg_train
;