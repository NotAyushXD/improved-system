"""
build_product_embeddings.py

Product embeddings don't exist yet, so this builds them from scratch — one
new step that runs BEFORE pipeline_main.py.

Adapted from multiview_clustering_v5.py's get_tpna_archetype_embeddings_no_brand()
/ build_item_text_no_brand(): same approach — embed at TPNA (style) grain,
leave brand out of the embedded text (brand was found to dominate similarity
for short/generic descriptions), optional anisotropy correction. Packaged
here as a standalone step feeding the GNN pipeline instead of the item-
clustering script.

Output shape matches exactly what parquet_loader.load_product_embeddings()
already expects — tpnb, embedding — so nothing downstream needs to change.
The embedding itself is generated at TPNA grain, then broadcast down: every
tpnb belonging to the same tpna (different sizes/colors of one style) gets
an IDENTICAL embedding vector.

No theme/category mapping here — need-states are found across the whole
assortment, with no category segmentation anywhere in this pipeline.
GraphBuilder.py's product sub-clustering (a separate, internal step from
this file) also runs globally across the whole catalog now, with no
category pre-grouping of any kind — see REFACTOR_NOTES.md.

SCOPE: sql/01_product_attributes_tpna.sql now only pulls tpna's actually
purchased in a given period (matching sql/04's training window) — the
catalog's long tail of never/rarely-bought products doesn't get an
embedding. Nothing to change in this file for that; it's inherited
automatically from what's in the downloaded parquet.

Requires, from running the two SQL files on your warehouse and downloading each
as parquet:
    data/product_attributes_tpna.parquet   <- sql/01_product_attributes_tpna.sql
    data/tpnb_to_tpna_mapping.parquet      <- sql/02_tpnb_to_tpna_mapping.sql

Produces:
    data/output/product_embeddings.parquet   (the file pipeline_main.py reads)

Install:  pip install sentence-transformers --break-system-packages

Usage:
    python build_product_embeddings.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import config

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# ── Update these to wherever you saved the three downloaded query results ──
# Note: Databricks/Spark exports a table as a FOLDER of part-files
# (part-00000-....snappy.parquet, maybe a _SUCCESS marker), not one single
# .parquet file. pandas can read that folder directly via pd.read_parquet(),
# so these constants point straight at the folder names as downloaded —
# no need to merge them into one file yourself.
PRODUCT_ATTRIBUTES_TPNA_PARQUET = Path(config.PRODUCT_ATTRIBUTES_TPNA)
TPNB_TO_TPNA_MAPPING_PARQUET    = Path(config.TPNB_TO_TPNA_MAPPING)

OUTPUT_PARQUET = Path(config.out("product_embeddings.parquet"))

# Swap for a local model path if you're on an offline machine — same as
# multiview_clustering_v5.py's local E:\...\miniLMV6L2 copy.
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME

# "All-but-the-top" anisotropy correction — removes the top N principal
# components after mean-centering. multiview_clustering_v5.py found this
# helped for short, domain-homogeneous retail text (most embedding variance
# otherwise sits in a few generic/boilerplate directions). N=2 is a starting
# point carried over from there, not re-validated for this product set —
# check downstream clustering quality with and without this before trusting it.
APPLY_ANISOTROPY_CORRECTION = config.APPLY_ANISOTROPY_CORRECTION
N_TOP_PCS_TO_REMOVE = config.N_TOP_PCS_TO_REMOVE

SEED = config.SEED


# ─────────────────────────────────────────────
# TEXT CONSTRUCTION — brand deliberately excluded
# ─────────────────────────────────────────────

def build_item_text_no_brand(row: pd.Series) -> str:
    """
    Brand is left out on purpose — multiview_clustering_v5.py found brand
    dominates similarity for short/generic product descriptions, causing
    items to cluster by brand rather than by what they actually are.
    """
    parts = [
        str(row.get("description", "") or ""),
        str(row.get("commercial_hierarchy_department", "") or ""),
        str(row.get("commercial_hierarchy_class", "") or ""),
        str(row.get("commercial_hierarchy_subclass", "") or ""),
    ]
    return " | ".join(p for p in parts if p.strip())


def deanisotropize_embeddings(embedding_matrix: np.ndarray, n_components: int = N_TOP_PCS_TO_REMOVE) -> np.ndarray:
    mean_vec = embedding_matrix.mean(axis=0)
    centered = embedding_matrix - mean_vec
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(centered)
    projection = centered @ pca.components_.T @ pca.components_
    return centered - projection


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    from sentence_transformers import SentenceTransformer

    print("Loading TPNA product attributes...")
    item_lookup = pd.read_parquet(PRODUCT_ATTRIBUTES_TPNA_PARQUET)
    item_lookup["tpna"] = item_lookup["tpna"].astype(str)
    item_lookup = item_lookup.drop_duplicates(subset=["tpna"])
    print(f"  {len(item_lookup):,} unique TPNA styles")

    item_lookup["item_text"] = item_lookup.apply(build_item_text_no_brand, axis=1)

    print(f"\nEmbedding {len(item_lookup):,} TPNA styles with {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        item_lookup["item_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if APPLY_ANISOTROPY_CORRECTION:
        print(f"Applying anisotropy correction (removing top {N_TOP_PCS_TO_REMOVE} PCs)...")
        embeddings = deanisotropize_embeddings(embeddings, N_TOP_PCS_TO_REMOVE)

    tpna_embeddings = pd.DataFrame({
        "tpna": item_lookup["tpna"].tolist(),
        "embedding": [e.astype(np.float32) for e in embeddings],
    })

    print("\nLoading tpnb -> tpna mapping...")
    mapping = pd.read_parquet(TPNB_TO_TPNA_MAPPING_PARQUET)
    mapping["tpnb"] = mapping["tpnb"].astype(str)
    mapping["tpna"] = mapping["tpna"].astype(str)

    dupes = mapping.groupby("tpnb")["tpna"].nunique()
    n_dupes = int((dupes > 1).sum())
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes:,} tpnb values map to more than one tpna — this "
              f"contradicts the stated clean hierarchy and is worth investigating. "
              f"Taking the first mapping for each as a stopgap.")
        mapping = mapping.drop_duplicates(subset="tpnb", keep="first")

    print("Broadcasting TPNA embeddings down to every tpnb...")
    product_df = mapping.merge(tpna_embeddings, on="tpna", how="inner")
    dropped = len(mapping) - len(product_df)
    print(f"  {len(product_df):,} tpnb rows now have an embedding "
          f"({dropped:,} dropped — their tpna had no attribute row to embed)")

    product_df = product_df[["tpnb", "embedding"]]

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    product_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved {OUTPUT_PARQUET} ({len(product_df):,} rows)")
    print("This is exactly the file pipeline_main.py's PRODUCT_EMBEDDINGS_PARQUET expects — "
          "nothing downstream needs to change.")


if __name__ == "__main__":
    main()