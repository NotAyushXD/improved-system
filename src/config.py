"""
config.py

Single place where every tunable parameter in this pipeline is defined,
typed, validated, and documented. Values come from (highest priority first):

    1. a real environment variable        PIPELINE_EPOCHS=30 python pipeline_main.py
    2. a .env file                        see .env.example
    3. the default baked in below         (identical to the previous hardcoded values)

Every default here reproduces the behaviour this pipeline had when these
values were hardcoded, so adding this file changes nothing until you
actually set something.

WHERE THE .env FILE IS LOOKED FOR
─────────────────────────────────
    $PIPELINE_ENV_FILE   if set (and it must then exist — a typo'd path is an
                         error, not a silent fallback to defaults)
    ./.env               when running from inside src/
    ../.env              the project root — the usual place
If none exists, defaults are used and a single line says so.

NO NEW DEPENDENCY: the .env parser below is ~40 lines of stdlib. This
pipeline already had one dependency blocked outright by machine policy
(pgserver, see duckdb_manager.py), so adding python-dotenv for this was not
worth the risk.

⚠ CHANGING SOME OF THESE INVALIDATES CACHED ARTIFACTS
──────────────────────────────────────────────────────
Several parameters change the MEANING of files this pipeline caches on
disk. Before this module existed those values were hardcoded, so it took a
code edit to hit that problem; now it takes an env var, which is exactly why
the fingerprinting below exists.

  graph_fingerprint()      covers everything that changes per-basket graph
                           construction (TOP_K, the sub-clustering settings,
                           SEED). It is written into the LMDB training-graph
                           manifest, so changing any of them forces an
                           automatic rebuild instead of silently training on
                           graphs built under different settings.

  subcluster_fingerprint() covers what product_subclusters.pkl depends on.
                           GraphBuilder writes it alongside that cache and
                           refuses to reuse a cache built under different
                           settings.

Artifacts that are NOT fingerprinted, and must be deleted by hand if you
change what feeds them: basket_gnn_model.pt, basket_gnn_embeddings.parquet,
copurchase_sparse.npz, gmm_basket_model.pkl, and the DuckDB tables
(basket_store.drop_all). See pipeline_main.py's header for the full list.

USAGE
─────
    import config
    for epoch in range(config.EPOCHS): ...

    config.describe()        # print every effective value and where it came from
    python config.py         # same, from the shell — a quick "what will run?" check
"""

import hashlib
import json
import os
from pathlib import Path

# ─────────────────────────────────────────────
# .env loading
# ─────────────────────────────────────────────

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}

_USED = {}      # key -> (effective value, source) — populated as values are read
_ENV_FILE = None


def _find_env_file():
    """
    An explicitly-requested file must exist and be readable — anything else is
    a silent, invisible config change. An AUTO-DISCOVERED file that cannot even
    be stat'd (locked-down machine, unreadable parent directory) is warned
    about and skipped rather than being fatal: the pipeline still has a
    complete set of defaults, and refusing to start at all would be worse than
    running on them loudly.
    """
    explicit = os.environ.get("PIPELINE_ENV_FILE")
    if explicit:
        path = Path(explicit)
        try:
            exists = path.exists()
        except OSError as e:
            raise OSError(
                f"PIPELINE_ENV_FILE={explicit!r} could not be read ({type(e).__name__}: {e}). "
                f"Fix the permissions or unset the variable — refusing to silently fall "
                f"back to defaults when a config file was explicitly requested."
            ) from None
        if not exists:
            raise FileNotFoundError(
                f"PIPELINE_ENV_FILE={explicit!r} does not exist. Fix the path or unset "
                f"the variable — failing loudly here rather than silently falling back "
                f"to defaults, which would be an invisible config change."
            )
        return path

    for candidate in (Path(".env"), Path("..") / ".env"):
        try:
            if candidate.exists():
                return candidate
        except OSError as e:
            print(f"Config WARNING: {candidate} exists but could not be checked "
                  f"({type(e).__name__}: {e}) — continuing with defaults. Set "
                  f"PIPELINE_ENV_FILE explicitly if you need that file to be used.")
    return None


def _parse_env_file(path: Path) -> dict:
    # encoding is explicit on purpose. Path.read_text() with no encoding uses
    # the LOCALE default, which is UTF-8 on Linux/macOS but cp1252 on a
    # Western-European Windows install (Python 3.14 still behaves this way;
    # PEP 686's UTF-8 default lands later). The shipped .env contains box-drawing
    # characters and arrows, so a locale-default read crashed on Windows with
    # `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90` — a config
    # file being unreadable purely because of which OS wrote it.
    # utf-8-sig also tolerates the BOM that Notepad and some Windows editors add.
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{path} is not valid UTF-8 ({e}). Re-save it as UTF-8 — on Windows, "
            f"Notepad's 'Save as' has an Encoding dropdown, and VS Code shows the "
            f"current encoding in the status bar. .env files are read as UTF-8 "
            f"regardless of the system locale so that the same file works on every "
            f"machine."
        ) from None
    except OSError as e:
        raise OSError(
            f"Found {path} but could not read it ({type(e).__name__}: {e}). Fix the "
            f"permissions, or move/remove the file — a config file that exists but is "
            f"unreadable must not be silently ignored."
        ) from None

    values = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key in {raw!r}")
        # A quoted value is taken literally; an unquoted one has any trailing
        # ` # comment` stripped. This is what lets a value legitimately
        # contain a '#' as long as it is quoted.
        if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        else:
            val = val.split("#", 1)[0].strip()
        values[key] = val
    return values


_ENV_FILE = _find_env_file()
_FILE_VALUES = _parse_env_file(_ENV_FILE) if _ENV_FILE else {}


def _raw(key: str):
    """Returns (value_or_None, source)."""
    if key in os.environ:
        return os.environ[key], "environment"
    if key in _FILE_VALUES:
        return _FILE_VALUES[key], f"{_ENV_FILE}"
    return None, "default"


def _record(key, value, source):
    _USED[key] = (value, source)
    return value


def _bad(key, value, expected):
    """Returns the exception rather than raising it, so call sites can
    `raise _bad(...) from None` and not show a confusing chained traceback
    from the underlying int()/float() failure."""
    return ValueError(
        f"Config error: {key}={value!r} is not a valid {expected}. "
        f"Set it correctly in your .env file or environment, or unset it to use the default."
    )


def _str(key, default, choices=None):
    raw, source = _raw(key)
    value = default if raw is None else raw
    if choices is not None and value not in choices:
        raise _bad(key, value, f"choice from {sorted(choices)}") from None
    return _record(key, value, source)


def _int(key, default, minimum=None, maximum=None):
    raw, source = _raw(key)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            raise _bad(key, raw, "integer") from None
    if minimum is not None and value < minimum:
        raise _bad(key, value, f"integer >= {minimum}") from None
    if maximum is not None and value > maximum:
        raise _bad(key, value, f"integer <= {maximum}") from None
    return _record(key, value, source)


def _float(key, default, minimum=None, maximum=None):
    raw, source = _raw(key)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            raise _bad(key, raw, "number") from None
    if minimum is not None and value < minimum:
        raise _bad(key, value, f"number >= {minimum}") from None
    if maximum is not None and value > maximum:
        raise _bad(key, value, f"number <= {maximum}") from None
    return _record(key, value, source)


def _bool(key, default):
    raw, source = _raw(key)
    if raw is None:
        value = default
    else:
        low = raw.strip().lower()
        if low in _TRUE:
            value = True
        elif low in _FALSE:
            value = False
        else:
            raise _bad(key, raw, "boolean (true/false, 1/0, yes/no, on/off)") from None
    return _record(key, value, source)


def _int_list(key, default, minimum=None):
    raw, source = _raw(key)
    if raw is None:
        value = list(default)
    else:
        try:
            value = [int(p.strip()) for p in raw.split(",") if p.strip()]
        except ValueError:
            raise _bad(key, raw, "comma-separated list of integers (e.g. 50,100,200,400)") from None
        if not value:
            raise _bad(key, raw, "non-empty comma-separated list of integers") from None
        if minimum is not None and min(value) < minimum:
            raise _bad(key, raw, f"list of integers all >= {minimum}") from None
    return _record(key, value, source)


def _opt_int(key, default, minimum=None):
    """An int that may also be the literal 'none' (mapped to Python None)."""
    raw, source = _raw(key)
    if raw is None:
        return _record(key, default, source)
    if raw.strip().lower() in {"none", "null", ""}:
        return _record(key, None, source)
    return _int(key, default, minimum=minimum)


# ═════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════

OUTPUT_DIR = _str("PIPELINE_OUTPUT_DIR", "../data/output")
DATA_DIR = _str("PIPELINE_DATA_DIR", "../data")

DUCKDB_PATH = _str("PIPELINE_DUCKDB_PATH", os.path.join("..", "data", "pipeline.duckdb"))

HOUSEHOLD_TPNB_WEEK_TRAIN = _str(
    "PIPELINE_HOUSEHOLD_WEEK_TRAIN", "../data/ns_household_tpnb_week_agg_train")
PRODUCT_ATTRIBUTES_TPNA = _str(
    "PIPELINE_PRODUCT_ATTRIBUTES_TPNA", "../data/ns_item_lookup_tpna")
TPNB_TO_TPNA_MAPPING = _str(
    "PIPELINE_TPNB_TO_TPNA_MAPPING", "../data/ns_tpnb_to_tpna_mapping")


def out(name: str) -> str:
    """Path to a pipeline-produced artifact under OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, name)


# ═════════════════════════════════════════════
# GENERAL
# ═════════════════════════════════════════════

SEED = _int("PIPELINE_SEED", 42)

# ═════════════════════════════════════════════
# PRODUCT EMBEDDINGS  (build_product_embeddings.py)
# ═════════════════════════════════════════════

EMBEDDING_MODEL_NAME = _str("PIPELINE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
APPLY_ANISOTROPY_CORRECTION = _bool("PIPELINE_APPLY_ANISOTROPY_CORRECTION", True)
N_TOP_PCS_TO_REMOVE = _int("PIPELINE_N_TOP_PCS_TO_REMOVE", 2, minimum=1)

# ═════════════════════════════════════════════
# BASKET CONSTRUCTION  (basket_store.py, pipeline_main.py)
# ═════════════════════════════════════════════

MIN_BASKET_PRODUCTS = _int("PIPELINE_MIN_BASKET_PRODUCTS", 2, minimum=1)
BASKET_COUNT_WARN_THRESHOLD = _int("PIPELINE_BASKET_COUNT_WARN_THRESHOLD", 2_000_000, minimum=0)

# ═════════════════════════════════════════════
# GRAPH CONSTRUCTION  (GraphBuilder.py)
# ⚠ everything in this block feeds graph_fingerprint()
# ═════════════════════════════════════════════

TOP_K = _int("PIPELINE_TOP_K", 10, minimum=1)

SUBCL_K_CANDIDATES = _int_list("PIPELINE_SUBCL_K_CANDIDATES", [50, 100, 200, 400], minimum=2)
SUBCL_N_INIT = _int("PIPELINE_SUBCL_N_INIT", 5, minimum=1)
SUBCL_BATCH_SIZE = _int("PIPELINE_SUBCL_BATCH_SIZE", 4096, minimum=1)
SUBCL_SIL_SAMPLE = _int("PIPELINE_SUBCL_SIL_SAMPLE", 5000, minimum=2)

# ═════════════════════════════════════════════
# CO-PURCHASE MATRIX  (pipeline_main.py)
# ═════════════════════════════════════════════

COPURCHASE_CHUNK_BASKETS = _int("PIPELINE_COPURCHASE_CHUNK_BASKETS", 500_000, minimum=1)

# ═════════════════════════════════════════════
# GNN  (GNN_Train.py)
# ═════════════════════════════════════════════

HIDDEN_DIM = _int("PIPELINE_HIDDEN_DIM", 128, minimum=1)
OUT_DIM = _int("PIPELINE_OUT_DIM", 64, minimum=1)
DROPOUT = _float("PIPELINE_DROPOUT", 0.1, minimum=0.0, maximum=1.0)
EDGE_DIM = _int("PIPELINE_EDGE_DIM", 2, minimum=1)
TRAIN_BATCH = _int("PIPELINE_TRAIN_BATCH", 256, minimum=1)
EPOCHS = _int("PIPELINE_EPOCHS", 20, minimum=1)
LR = _float("PIPELINE_LR", 1e-4, minimum=0.0)
WEIGHT_DECAY = _float("PIPELINE_WEIGHT_DECAY", 1e-5, minimum=0.0)
N_TRAIN_SAMPLES = _int("PIPELINE_N_TRAIN_SAMPLES", 300_000, minimum=1)
NUM_WORKERS = _int("PIPELINE_NUM_WORKERS", 0 if os.name == "nt" else 4, minimum=0)

INFERENCE_CHUNK_BASKETS = _int("PIPELINE_INFERENCE_CHUNK_BASKETS", 50_000, minimum=1)
INFERENCE_BATCH_SIZE = _int("PIPELINE_INFERENCE_BATCH_SIZE", 8192, minimum=1)
CHUNK_STALE_AFTER_SECONDS = _int("PIPELINE_CHUNK_STALE_AFTER_SECONDS", 3600, minimum=1)
LMDB_MAP_SIZE_GB = _int("PIPELINE_LMDB_MAP_SIZE_GB", 200, minimum=1)

# ═════════════════════════════════════════════
# CLUSTERING  (cluster_basket_embeddings.py, pipeline_main.py)
# ═════════════════════════════════════════════

BASKET_KNN_K = _int("PIPELINE_BASKET_KNN_K", 15, minimum=1)
USE_MUTUAL_KNN = _bool("PIPELINE_USE_MUTUAL_KNN", True)
LEIDEN_RESOLUTION = _float("PIPELINE_LEIDEN_RESOLUTION", 1.0, minimum=0.0)
# leidenalg.find_partition()'s own default is 2. Named here because
# run_leiden_on_basket_graph() drives the optimiser one iteration at a time so
# each one can report progress, which needs the count to be explicit.
LEIDEN_N_ITERATIONS = _int("PIPELINE_LEIDEN_N_ITERATIONS", 2, minimum=1)

GMM_N_COMPONENTS = _int("PIPELINE_GMM_N_COMPONENTS", 30, minimum=1)
GMM_K_MIN = _int("PIPELINE_GMM_K_MIN", 5, minimum=1)
GMM_K_MAX = _int("PIPELINE_GMM_K_MAX", 60, minimum=1)
GMM_K_STEP = _int("PIPELINE_GMM_K_STEP", 5, minimum=1)
GMM_N_INIT = _int("PIPELINE_GMM_N_INIT", 3, minimum=1)
GMM_COVARIANCE = _str("PIPELINE_GMM_COVARIANCE", "diag",
                      choices={"full", "tied", "diag", "spherical"})

ASSIGN_NEW_BASKET_K = _int("PIPELINE_ASSIGN_NEW_BASKET_K", 15, minimum=1)

# The basket kNN graph is the most expensive artifact in Stage 2 (hours of
# approximate-NN search over the full basket population) and was the only one
# never written to disk — a Stage 2 that died in Leiden or GMM rebuilt it from
# scratch on the next run. Cached to parquet with a fingerprint sidecar, so a
# rerun resumes from the edge list instead. Costs a few GB of disk; set false
# if that drive is tight.
CACHE_BASKET_EDGES = _bool("PIPELINE_CACHE_BASKET_EDGES", True)

# ═════════════════════════════════════════════
# PROGRESS REPORTING  (cluster_basket_embeddings.py)
# ═════════════════════════════════════════════

# How often a long-running clustering step reports that it is still alive.
# Stage 2's steps are single opaque calls that can run for a long time with no
# output; this is the interval of the "still running, N elapsed" line. Raise it
# if it makes a captured log noisy.
PROGRESS_HEARTBEAT_SECS = _int("PIPELINE_PROGRESS_HEARTBEAT_SECS", 30, minimum=1)

# ═════════════════════════════════════════════
# NEED-STATE GRAPHS  (need_state_graph.py)
# ═════════════════════════════════════════════

BASKET_EDGE_WRITE_CAP = _int("PIPELINE_BASKET_EDGE_WRITE_CAP", 50_000_000, minimum=0)
MIN_TRANSITION_SUPPORT = _int("PIPELINE_MIN_TRANSITION_SUPPORT", 30, minimum=1)
WRITE_BASKET_EDGES = _bool("PIPELINE_WRITE_BASKET_EDGES", True)
# 1 = consecutive calendar weeks only; 'none' = consecutive OBSERVED baskets
# however far apart (see need_state_graph.build_need_state_transitions).
TRANSITION_MAX_WEEK_GAP = _opt_int("PIPELINE_TRANSITION_MAX_WEEK_GAP", 1, minimum=1)
JOURNEY_DEPTH = _int("PIPELINE_JOURNEY_DEPTH", 3, minimum=1)
JOURNEY_BEAM = _int("PIPELINE_JOURNEY_BEAM", 5, minimum=1)

# ═════════════════════════════════════════════
# SCORING  (score_new_baskets.py)
# ═════════════════════════════════════════════

SCORE_DATASET_TAG = _str("PIPELINE_SCORE_DATASET_TAG", "score")
TRAIN_DATASET_TAG = _str("PIPELINE_TRAIN_DATASET_TAG", "train")


# ═════════════════════════════════════════════
# VALIDATION ACROSS PARAMETERS
# ═════════════════════════════════════════════

def _validate():
    problems = []
    if GMM_K_MIN > GMM_K_MAX:
        problems.append(
            f"PIPELINE_GMM_K_MIN ({GMM_K_MIN}) > PIPELINE_GMM_K_MAX ({GMM_K_MAX}) — "
            f"select_k_via_bic() would sweep an empty range.")
    if OUT_DIM > HIDDEN_DIM:
        problems.append(
            f"PIPELINE_OUT_DIM ({OUT_DIM}) > PIPELINE_HIDDEN_DIM ({HIDDEN_DIM}) — the "
            f"projection head would expand rather than compress. Allowed, but almost "
            f"certainly a mistake; set them deliberately if you meant it.")
    if MIN_BASKET_PRODUCTS < 2:
        problems.append(
            f"PIPELINE_MIN_BASKET_PRODUCTS ({MIN_BASKET_PRODUCTS}) < 2 — a 1-product "
            f"basket has no co-purchase pair, so every node gets cp_score 0 and no "
            f"edges. build_one_graph handles it, but such baskets carry no signal.")
    if EDGE_DIM != 2:
        problems.append(
            f"PIPELINE_EDGE_DIM ({EDGE_DIM}) != 2 — GraphBuilder._build_edges_numba "
            f"emits exactly 2 edge features (log co-purchase, relative strength). "
            f"Changing this alone will fail at the first GINEConv.")
    if problems:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {p}" for p in problems))


_validate()


# ═════════════════════════════════════════════
# FINGERPRINTS — cache invalidation
# ═════════════════════════════════════════════

def subcluster_fingerprint() -> dict:
    """
    Everything product_subclusters.pkl depends on. GraphBuilder writes this
    next to that cache and refuses to reuse a cache built under different
    settings — previously the cache was keyed by filename alone, so changing
    any of these silently reused stale sub-clusters.
    """
    return {
        "subcl_k_candidates": list(SUBCL_K_CANDIDATES),
        "subcl_n_init": SUBCL_N_INIT,
        "subcl_batch_size": SUBCL_BATCH_SIZE,
        "subcl_sil_sample": SUBCL_SIL_SAMPLE,
        "seed": SEED,
    }


def graph_fingerprint() -> dict:
    """
    Everything that changes per-basket GRAPH construction. Goes into the LMDB
    training-graph manifest. TOP_K in particular was NOT previously tracked
    there, so changing it silently reused a cache whose graphs had a
    different number of edges per node.
    """
    fp = {"top_k": TOP_K}
    fp.update(subcluster_fingerprint())
    return fp


def fingerprint_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ═════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════

def describe(only_overridden: bool = False):
    """Print every effective value and where it came from."""
    if _ENV_FILE:
        print(f"Config: loaded {_ENV_FILE}")
    else:
        print("Config: no .env file found — using defaults "
              "(create one from .env.example to change anything)")

    overridden = {k: v for k, v in _USED.items() if v[1] != "default"}
    items = overridden if only_overridden else _USED
    if only_overridden and not overridden:
        print("  (nothing overridden — every value is its default)")
        return

    width = max(len(k) for k in items) if items else 0
    for key in sorted(items):
        value, source = _USED[key]
        marker = " " if source == "default" else "*"
        print(f"  {marker} {key:<{width}}  = {value!r}   [{source}]")

    if not only_overridden and overridden:
        print(f"\n  {len(overridden)} value(s) overridden, marked with *")
    print(f"\n  graph fingerprint      : {fingerprint_hash(graph_fingerprint())}  {graph_fingerprint()}")
    print(f"  sub-cluster fingerprint: {fingerprint_hash(subcluster_fingerprint())}")


if __name__ == "__main__":
    describe()
