-- ============================================================
-- ns_household_tpnb_week_agg_score.sql
--   (was ns_household_tpnb_period_agg_score.sql — renamed to match the
--   basket grain change; see ns_household_tpnb_week_agg_train.sql for the
--   full explanation of why WEEK, not PERIOD, and not a true single-visit
--   basket)
--
-- Feeds  : score_new_baskets.py's --new-transactions input
-- Grain  : identical to ns_household_tpnb_week_agg_train.sql — same real,
-- confirmed table, just a different (later / held-out) period range so
-- you're scoring baskets the model never trained on.
--
-- Materializes into a table so it can be downloaded/exported as parquet —
-- save the export as data/new_basket_source.parquet.
-- ============================================================

DROP TABLE IF EXISTS lab_customer_value_analytics.ns_household_tpnb_week_agg_score;

CREATE TABLE lab_customer_value_analytics.ns_household_tpnb_week_agg_score
USING PARQUET
AS
SELECT
    year_number,
    period_number,
    week_number,
    household_number,
    tpnb,
    SUM(quantity)       AS quantity,
    SUM(orders)         AS orders,
    SUM(sales_inc_vat)  AS sales_inc_vat
FROM lab_customer_value_analytics.cltv_hh_metrics_tpnb_base
WHERE (year_number * 100 + period_number)
      BETWEEN 202603 AND 202604   -- <<< CONFIRM: periods AFTER training's range
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
FROM lab_customer_value_analytics.ns_household_tpnb_week_agg_score
;
