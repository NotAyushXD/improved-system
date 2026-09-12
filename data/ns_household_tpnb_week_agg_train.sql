-- ============================================================
-- ns_household_tpnb_week_agg_train.sql
--   (was ns_household_tpnb_period_agg_train.sql — renamed because the
--   basket grain changed from PERIOD to WEEK; see the note below)
--
-- Feeds  : pipeline_main.py's tpnb_x_hh -> basket construction, product_units_avg
-- Grain  : one row per household_number x tpnb x year_number x week_number
-- Cols required downstream: household_number, tpnb, year_number, period_number,
--   week_number, quantity
--
-- BASKET GRAIN — WEEK, NOT A TRUE SINGLE-VISIT BASKET:
-- None of the tables available in this warehouse (product.product,
-- LAB_INSIGHT_CUSTOMER_ANALYTICS.IN22915286_UPDATED_PRODUCT_TABLE,
-- lab_customer_value_analytics.cltv_hh_metrics_tpnb_base,
-- product.buyer_hierarchy — see data/TABLE_REFERENCE.md) carry a
-- transaction/order/checkout identifier. cltv_hh_metrics_tpnb_base (the
-- only household-purchase table available) is already pre-aggregated to
-- WEEK grain, and its `orders` column is a COUNT of separate orders folded
-- into that week's total, not a preserved per-order identity. A true
-- single-visit "basket" cannot be reconstructed from what's available here.
--
-- WEEK is the finest grain this data supports, so it's used as the basket
-- unit instead of the previous PERIOD grain (~4 weeks). This materially
-- reduces — does not eliminate — how many distinct shopping occasions get
-- merged into one "basket": a household's whole month of shopping no longer
-- collapses into a single node, only its whole week does. If a genuine
-- transaction/order-level fact table is ever found upstream of
-- cltv_hh_metrics_tpnb_base, basket construction should move to that
-- instead — this is a data-availability compromise, not the target design.
--
-- Source: lab_customer_value_analytics.cltv_hh_metrics_tpnb_base — already
-- at week grain (year_number, period_number, week_number, household_number,
-- tpnb) per its own schema, so the GROUP BY below is a defensive dedupe (in
-- case of any duplicate rows), not a real aggregation across anything.
-- ============================================================

DROP TABLE IF EXISTS lab_customer_value_analytics.ns_household_tpnb_week_agg_train;

CREATE TABLE lab_customer_value_analytics.ns_household_tpnb_week_agg_train
USING PARQUET
AS
SELECT
    year_number,
    period_number,
    week_number,
    household_number,
    tpnb,
    SUM(quantity)       AS quantity,
    SUM(orders)         AS orders,          -- count of orders folded into this week — not consumed downstream yet, kept for future profiling
    SUM(sales_inc_vat)  AS sales_inc_vat    -- not consumed downstream yet, kept for future profiling
FROM lab_customer_value_analytics.cltv_hh_metrics_tpnb_base
WHERE (year_number * 100 + period_number)
      BETWEEN 202603 AND 202604   -- <<< CONFIRM: training period range, e.g. 202401 AND 202413
  AND household_number IS NOT NULL
  AND tpnb IS NOT NULL
GROUP BY year_number, period_number, week_number, household_number, tpnb
;

-- ============================================================
-- VALIDATION
-- ============================================================

SELECT
    MIN(year_number * 100 + period_number)          AS min_year_period,
    MAX(year_number * 100 + period_number)          AS max_year_period,
    COUNT(*)                                        AS row_count,
    COUNT(DISTINCT household_number)                AS household_count,
    COUNT(DISTINCT tpnb)                            AS product_count,
    COUNT(DISTINCT year_number * 100 + week_number) AS distinct_weeks
FROM lab_customer_value_analytics.ns_household_tpnb_week_agg_train
;
