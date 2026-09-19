"""
reset_inference.py

Clears the state left behind by an interrupted inference run, so the next
`python pipeline_main.py` starts the embedding pass cleanly instead of
resuming into an inconsistent mixture.

WHY THIS IS NEEDED
──────────────────
Until recently, GNN_Train.train_and_embed() saved the trained model AFTER the
inference pass finished. Inference over a full basket population runs for
hours or days, so a run that died partway through lost its weights entirely —
they only ever existed in that process's memory, and with no torch seed set
they could not be reproduced by retraining.

The chunk queue, however, does not know that. It still shows those chunks as
'complete'. Left alone, the next run would keep them and embed only the
REMAINING chunks — using a newly trained model with different weights. The
result is one embeddings file containing two incompatible embedding spaces,
clustered together as if they were one. Nothing would error; the need-states
would simply be wrong in a way that is very hard to detect afterwards.

So: if the model file is missing, every chunk embedded by that run is an
orphan and must go. This script finds and removes them.

(Both underlying problems are now fixed — the model is saved BEFORE inference,
and training is seeded — so a future interrupted run genuinely resumes. This
script exists for runs that predate those fixes, and as a general reset.)

WHAT IT TOUCHES
───────────────
  deletes  data/output/embeddings_chunk_<tag>_*.parquet
  deletes  data/output/basket_gnn_embeddings.parquet   (the merged result)
  resets   the inference_chunks_<tag> queue table in DuckDB

WHAT IT LEAVES ALONE — these stay valid and are expensive to rebuild:
  baskets_<tag>              the DuckDB basket table   (Stage 0)
  copurchase_sparse.npz      the co-purchase matrix    (Stage 1a, hours)
  product_embeddings.parquet product vectors
  product_subclusters.pkl    global product clustering
  product_id_to_index.pkl / product_units_avg.pkl

Usage:
    python reset_inference.py              # show what would be removed
    python reset_inference.py --yes        # actually remove it
"""

import argparse
import glob
import os

import config
import duckdb_manager


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-tag", default=config.TRAIN_DATASET_TAG)
    ap.add_argument("--yes", action="store_true",
                    help="Actually delete. Without this, only reports (dry run).")
    ap.add_argument("--force", action="store_true",
                    help="Reset even if a trained model file DOES exist (i.e. even "
                         "if the completed chunks look salvageable).")
    args = ap.parse_args()

    tag = args.dataset_tag
    out = config.OUTPUT_DIR
    model_path = os.path.join(out, "basket_gnn_model.pt")

    print(f"Output dir : {out}")
    print(f"Dataset tag: {tag}\n")

    # ── Is there a model to match the completed chunks against? ──
    model_exists = os.path.exists(model_path)
    if model_exists:
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"FOUND trained model: {model_path} ({size_mb:.1f} MB)")
        print("  The chunks already embedded were produced by SOME model. If this file")
        print("  is that model, they are still valid and you can resume instead of")
        print("  resetting — rerun pipeline_main.py and it will skip completed chunks.")
        if not args.force:
            print("\n  Refusing to reset while a model file exists. Re-run with --force if")
            print("  you are sure these chunks should be discarded anyway (for example,")
            print("  because graph construction changed since they were written).")
            return
        print("\n  --force given: resetting anyway.\n")
    else:
        print(f"NO trained model at {model_path}")
        print("  The interrupted run never got far enough to save its weights, so every")
        print("  chunk it embedded is an orphan: it cannot be reproduced, and cannot be")
        print("  mixed with chunks from a future (differently-weighted) model.")
        print("  These chunks must be discarded.\n")

    # ── Chunk parquet files ──
    pattern = os.path.join(out, f"embeddings_chunk_{tag}_*.parquet")
    chunk_files = sorted(glob.glob(pattern))
    total_mb = sum(os.path.getsize(p) for p in chunk_files) / 1e6 if chunk_files else 0
    print(f"Chunk files : {len(chunk_files):,} matching {os.path.basename(pattern)} "
          f"({total_mb:,.0f} MB)")

    merged = os.path.join(out, "basket_gnn_embeddings.parquet")
    merged_exists = os.path.exists(merged)
    if merged_exists:
        print(f"Merged file : {merged} ({os.path.getsize(merged)/1e6:,.0f} MB)")

    # ── Chunk queue table ──
    con = duckdb_manager.get_connection()
    chunks_table = f"inference_chunks_{tag}"
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [chunks_table]).fetchone()[0] > 0

    if exists:
        rows = con.execute(
            f'SELECT status, COUNT(*) FROM "{chunks_table}" GROUP BY status ORDER BY status'
        ).fetchall()
        print(f"Queue table : {chunks_table} -> " +
              ", ".join(f"{s}={c:,}" for s, c in rows))
    else:
        print(f"Queue table : {chunks_table} does not exist (nothing to reset)")

    if not args.yes:
        print("\nDRY RUN — nothing changed. Re-run with --yes to apply.")
        return

    # ── Apply ──
    print("\nApplying...")
    for p in chunk_files:
        os.remove(p)
    print(f"  removed {len(chunk_files):,} chunk parquet files")

    if merged_exists:
        os.remove(merged)
        print(f"  removed {merged}")

    if exists:
        # Dropping the table (rather than setting every row back to 'pending')
        # lets ensure_inference_chunk_plan() rebuild it from the CURRENT basket
        # count and chunk size — which may differ from the interrupted run's.
        con.execute(f'DROP TABLE IF EXISTS "{chunks_table}"')
        print(f"  dropped {chunks_table} (it will be replanned on the next run)")

    print("\nDone. The next `python pipeline_main.py` will:")
    print("  - reuse baskets_%s and copurchase_sparse.npz (no rebuild)" % tag)
    print("  - retrain the GNN and SAVE THE MODEL BEFORE inference")
    print("  - embed every basket from scratch, resumably this time")


if __name__ == "__main__":
    main()
