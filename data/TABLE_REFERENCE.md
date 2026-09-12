# Warehouse table reference

What each source table actually contains, and how (or whether) this pipeline
uses it. Based on `DESCRIBE TABLE` output shared directly — not guessed.

---

## `lab_customer_value_analytics.cltv_hh_metrics_tpnb_base`

**The only household-purchase table available**, and the single most
important table for this pipeline — it's what `ns_household_tpnb_week_agg_train.sql`
and `ns_household_tpnb_week_agg_score.sql` read from.

| Column | Type | Notes |
|---|---|---|
| `year_number` | int | |
| `period_number` | int | Retail period (~13/year, ~4 weeks each) |
| `week_number` | int | **Finest time grain in this table** |
| `household_number` | bigint | Customer/household identifier |
| `tpnb` | int | Product identifier (item/barcode grain) |
| `orders` | bigint | **Count** of separate orders folded into this row — not a preserved order/transaction ID. See "basket grain" below. |
| `sales_inc_vat` | double | |
| `quantity` | double | |

**Grain: one row per `household_number` × `tpnb` × `week_number` (within a
year).** This is already the finest grain this table offers — there is no
day-level or transaction-level breakdown underneath it.

**Why this matters for basket construction:** a "basket" in this pipeline is
built by grouping this table's rows by household and time window. There is
**no transaction/order/checkout identifier anywhere in this table** — the
`orders` column tells you *how many* separate orders got summed into a row,
but not *which* products were in *which* order. That means a true
single-visit basket cannot be reconstructed from this table, no matter how
the SQL is written. The best available proxy is grouping by
`household_number` × `week_number` (used by
`ns_household_tpnb_week_agg_train.sql` / `_score.sql`) rather than by
`period_number` (~4 weeks) as an earlier version did — narrower window, less
occasion-mixing, but still not a real single-visit basket.

---

## `product.product`

The full product master table — **~200 columns**, most of them irrelevant to
this pipeline (clothing attributes, pharmacy fields, nutrition/GDA panels,
digital imagery, packaging/recycling info, etc.). Only a handful are actually
read, by `ns_item_lookup_tpna.sql` and `ns_tpnb_to_tpna_mapping.sql`:

| Column | Type | Used for |
|---|---|---|
| `tpnb` | int | Item/barcode-level product ID — the grain everything downstream joins on |
| `tpna` | string | Style/archetype-level ID — groups size/color variants of the same product; product embeddings are built at this grain, then broadcast down to every `tpnb` |
| `tpna_description` | string | Preferred description when populated (already at TPNA grain) |
| `description` | string | Fallback description (TPNB grain, mode-aggregated up to TPNA if `tpna_description` is blank) |
| `brand` | string | Read but **deliberately excluded** from the text embedded for product similarity (found to dominate similarity for short/generic descriptions) |
| `commercial_hierarchy_department_name` / `_class_name` / `_subclass_name` | string | Embedded alongside description — gives the embedding model category-flavored context without ever being used to *pre-group* products (theme-free design) |
| `is_deleted`, `is_archived` | boolean | Filtered to `FALSE` — live products only |
| `load_date_time` | timestamp | Used to pick the most-recently-loaded `tpna_description` when more than one exists |

Everything else on this table (nutrition, allergens, digital content,
clothing/pharmacy-specific fields, packaging, dimensions, GPC/GTIN metadata,
etc.) is **not used anywhere in this pipeline**. If future work wants
richer product text for embeddings (e.g. `marketing_text`,
`customer_friendly_description`, `ingredients`), this is the table to pull
from.

Also carries `commercial_hierarchy_subclass_code` (and department/class/
division/section codes) — the same kind of code that keys
`product.buyer_hierarchy` below, though nothing in this pipeline currently
joins the two.

---

## `LAB_INSIGHT_CUSTOMER_ANALYTICS.IN22915286_UPDATED_PRODUCT_TABLE`

| Column | Type |
|---|---|
| `TPNB` | int |
| `DESCRIPTION` | string |
| `PSG` | string |
| `JUNIOR_AREA_DESCRIPTION` | string |
| `BUYER_AREA_DESCRIPTION` | string |
| `PRODUCT_AREA_DESCRIPTION` | string |
| `CATEGORY_AREA_DESCRIPTION` | string |
| `CATEGORY` | string |
| `COMMERCIAL_AREA_DESCRIPTION` | string |
| `BUSINESS_AREA_DESCRIPTION` | string |

**Grain: one row per `TPNB`** — a simplified, already-flattened merchandising
hierarchy directly at product-barcode grain (no join needed). Used by
`ns_product_theme_mapping.sql`, which pulls just `TPNB` →
`CATEGORY_AREA_DESCRIPTION` (aliased `theme_name`).

**Not currently consumed by any Python file in this pipeline** — the
pipeline is theme-free by design, so no category concept feeds the GNN or
clustering. This mapping is kept only for optional *post-hoc* profiling —
e.g. once need-state clusters are found, joining this table against each
cluster's basket contents to describe what the cluster dominantly contains,
in human-readable category terms. See `PIPELINE_GUIDE.md` §7.

The column names here (`JUNIOR_AREA_DESCRIPTION`, `BUYER_AREA_DESCRIPTION`,
`PRODUCT_AREA_DESCRIPTION`, `CATEGORY_AREA_DESCRIPTION`) line up closely with
`product.buyer_hierarchy`'s columns below — this table is very likely a
pre-joined, TPNB-grain materialization of that hierarchy (possibly via
`product.product`'s `commercial_hierarchy_subclass_code`), built for
convenience so consumers don't have to do that join themselves.

---

## `product.buyer_hierarchy`

| Column | Type |
|---|---|
| `subclass_code` | string |
| `junior_area_code` / `junior_area_description` | string |
| `buyer_area_code` / `buyer_area_description` | string |
| `product_area_code` / `product_area_description` | string |
| `category_area_code` / `category_area_description` | string |
| `commercial_area_code` / `commercial_area_description` | string |
| `business_area_code` / `business_area_description` | string |
| `country_code` | string |
| `cost_centre` | string |
| `product_code` | string |
| `effective_date` | date |

**Grain: one row per `subclass_code`** (plus country/date — hierarchy
mappings can change over time, hence `effective_date`) — a code-to-description
lookup for the full merchandising hierarchy (junior area → buyer area →
product area → category area → commercial area → business area), keyed by
the same subclass code that appears on `product.product` as
`commercial_hierarchy_subclass_code`.

**Not currently used by any SQL file in this pipeline.** Likely the
authoritative source that `IN22915286_UPDATED_PRODUCT_TABLE` was built from
(see above) — if that table ever goes stale or you need a hierarchy level it
doesn't carry (e.g. `cost_centre`, `effective_date`-aware historical
mappings), this table plus a join to `product.product` on
`commercial_hierarchy_subclass_code = subclass_code` is the fallback path.

---

## Summary: what feeds what

```
product.product ──────────────────────────┐
                                            ├──► ns_item_lookup_tpna.sql ──────┐
                                            │                                   ├──► build_product_embeddings.py
                                            └──► ns_tpnb_to_tpna_mapping.sql ──┘

LAB_INSIGHT_CUSTOMER_ANALYTICS
.IN22915286_UPDATED_PRODUCT_TABLE ─────────────► ns_product_theme_mapping.sql
                                                  (output currently unused downstream —
                                                   kept for future post-hoc cluster profiling)

product.buyer_hierarchy ─────────────────────────  not read by any SQL file here
                                                    (probable source of the table above)

lab_customer_value_analytics
.cltv_hh_metrics_tpnb_base ─────────────────────► ns_household_tpnb_week_agg_train.sql
                                                  ns_household_tpnb_week_agg_score.sql
                                                  (basket construction — grain is WEEK,
                                                   the finest this table supports; no
                                                   transaction/order ID exists here)
```
