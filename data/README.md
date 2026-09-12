# `data/` — Data Folder Reference

All folders here are Spark/Databricks exports — each is a **folder of part-files** (e.g. `part-00000-....snappy.parquet`), not a single file. `pandas.read_parquet()` can read an entire folder directly.

---

## Input Tables (Warehouse Exports)

---

### `ns_item_lookup_tpna/`

**What it is:** Product attribute catalogue at **TPNA grain** (style level — one row per style, independent of size/colour variants).

**Used by:** `build_product_embeddings.py`

| Column | Description |
|--------|-------------|
| `tpna` | Trade Parent Number (style-level product identifier). Primary key at this grain. |
| `description` | Product display name / short description |
| `commercial_hierarchy_department` | Top-level commercial department (e.g. "Grocery", "Clothing") |
| `commercial_hierarchy_class` | Mid-level product class within the department |
| `commercial_hierarchy_subclass` | Fine-grained product subclass |

> **Note:** Brand is intentionally **not** embedded into the product text — it was found to dominate similarity for short/generic descriptions, causing products to cluster by brand rather than by what they are.

---

### `ns_tpnb_to_tpna_mapping/`

**What it is:** A mapping table that links **TPNB** (individual SKU — includes size, colour, pack variants) back up to **TPNA** (the parent style). Used to broadcast style-level embeddings down to SKU grain.

**Used by:** `build_product_embeddings.py`

| Column | Description |
|--------|-------------|
| `tpnb` | Trade Parent Number (base) — the individual SKU identifier. Many TPNBs map to one TPNA. |
| `tpna` | Trade Parent Number (style) — the parent style. Should be a clean many-to-one mapping; any TPNB with more than one TPNA is flagged as a data quality warning. |

> **Data quality note:** If a TPNB maps to more than one TPNA, `build_product_embeddings.py` logs a warning and takes the first mapping as a stopgap. This is worth investigating upstream.

---

### `ns_household_tpnb_period_agg_train/`

**What it is:** Aggregated household transaction data used to **train** the GNN. Each row is one product purchased by one household in one period. This is the primary training signal — it defines what products were bought together (co-purchase) and in what quantities.

**Used by:** `pipeline_main.py` (via `parquet_loader.load_household_tpnb_period()`)

| Column | Description |
|--------|-------------|
| `household_number` | Anonymised household identifier. Combined with `year_period_number` to form a basket. |
| `tpnb` | SKU purchased (Trade Parent Number base). Joined to product embeddings on this key. |
| `year_number` | Calendar year of the shopping period (e.g. `2023`). |
| `period_number` | Period number within the year (e.g. Tesco uses 4-week periods: 1–13). |
| `quantity` | Number of units of this TPNB purchased by this household in this period. Used as the volume weight (`log1p(quantity)`) in GNN node features and basket pooling. |

> **Derived columns (computed at runtime, not stored):**
> - `year_period_number` = `year_number * 100 + period_number` — used as a single period key
> - `basket_id` = `"{household_number}_{year_period_number}"` — unique basket identifier

---

### `ns_household_tpnb_period_agg_score/`

**What it is:** The same schema as `ns_household_tpnb_period_agg_train/`, but for a **different time window or held-out cohort** — used as the input to `score_new_baskets.py` to assign need-states to new baskets without retraining.

**Used by:** `score_new_baskets.py`

| Column | Description |
|--------|-------------|
| `household_number` | Same as train table — anonymised household identifier |
| `tpnb` | SKU purchased |
| `year_number` | Calendar year |
| `period_number` | Period number within the year |
| `quantity` | Units purchased |

> Pass the path to this folder as the `--new-transactions` argument:
> ```bash
> python score_new_baskets.py --new-transactions ../data/ns_household_tpnb_period_agg_score
> ```

---

## Output Files

---

### `output/product_embeddings.parquet`

**What it is:** The computed product embedding table. Single parquet file (not a folder). Written by `build_product_embeddings.py`, read by `pipeline_main.py` and `score_new_baskets.py`.

| Column | Description |
|--------|-------------|
| `tpnb` | SKU identifier — joins to the transaction tables on this key |
| `embedding` | Fixed-length float32 vector (MiniLM-L6-v2 output dimension: 384, optionally anisotropy-corrected). Represents the product's meaning in semantic space, based on its description and commercial hierarchy. |

> One embedding per TPNB. TPNBs sharing the same TPNA (style) get **identical** embeddings — the embedding is generated at style grain and broadcast down.

---

## Runtime Artifacts (written to `src/`, not `data/`)

These are produced during a pipeline run and live alongside the scripts in `src/`. Listed here for completeness:

| File | Written by | Description |
|------|-----------|-------------|
| `basket_gnn_embeddings.parquet` | `pipeline_main.py` | 64-dim GNN embedding per basket. Columns: `basket_id`, `gnn_embedding` (np.ndarray) |
| `basket_need_state_clusters.parquet` | `pipeline_main.py` | Cluster assignments per basket. Columns: `basket_id`, `need_state_cluster` (Leiden int), `need_state_cluster_gmm` (GMM int), `gmm_confidence` (float) |
| `basket_gnn_model.pt` | `pipeline_main.py` | Trained GNN model weights (PyTorch state dict) |
| `gmm_basket_model.pkl` | `pipeline_main.py` | Fitted GMM model — used by `score_new_baskets.py` to score new baskets directly |
| `copurchase_sparse.npz` | `pipeline_main.py` | Sparse product×product co-purchase matrix (scipy CSR) |
| `product_id_to_index.pkl` | `pipeline_main.py` | Dict mapping TPNB string → integer row index in the embedding/co-purchase matrices |
| `product_theme.pkl` | `pipeline_main.py` | Dict mapping TPNB → `"ALL_PRODUCTS"` (constant — satisfies GraphBuilder interface) |
| `product_units_avg.pkl` | `pipeline_main.py` | Dict mapping TPNB → mean quantity purchased across all training transactions |
| `product_subclusters.pkl` | `GraphBuilder.py` | Cached KMeans sub-cluster assignments and distinctiveness scores per product. **Delete before re-running if product data changed.** |
| `training_graphs.pkl` | `GraphBuilder.py` | Cached PyTorch Geometric graph list for training. **Delete before re-running if transaction or product data changed.** |

---

## Key Identifiers — Glossary

| Term | Grain | Description |
|------|-------|-------------|
| `tpnb` | SKU | Trade Parent Number (base) — the finest product identifier, distinguishing individual sizes/colours/pack formats |
| `tpna` | Style | Trade Parent Number (style) — the parent style grouping multiple TPNBs |
| `household_number` | Household | Anonymised loyalty card / customer identifier |
| `period_number` | Period | Sub-annual shopping period (e.g. 4-week period in a 13-period year) |
| `basket_id` | Basket | `"{household_number}_{year_period_number}"` — all products a household bought in one period |
| `need_state_cluster` | Basket | Leiden community label — which need-state this basket belongs to |
| `need_state_cluster_gmm` | Basket | GMM component label — alternative need-state assignment |
