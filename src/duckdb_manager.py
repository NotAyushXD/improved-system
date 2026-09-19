"""
duckdb_manager.py

Self-contained embedded database for this pipeline's basket storage —
replaces the earlier pgserver/Postgres-based design (pg_manager.py,
now removed). The reason for the switch: pgserver's `initdb` step needs to
change Windows directory permissions (an ACL lock-down, like `chmod 700` on
Linux) on its data directory, and on a heavily-managed corporate machine
that operation was blocked outright — confirmed across three different
local NTFS drives, even running as Administrator. DuckDB never does
anything like that: it just opens an ordinary file with normal read/write
calls, exactly like any other file this pipeline already reads or writes,
so there's no privileged operation to be blocked.

Trade-off worth knowing: DuckDB is a single-writer embedded engine — it has
no row-level locking (`FOR UPDATE SKIP LOCKED`, which the Postgres-based
restartable inference queue used) and isn't built for concurrent
multi-process writers the way Postgres is. Everything here still works
correctly for this pipeline's actual usage (one process at a time), but the
"ready for independent machines to claim chunks later" aspiration from the
original design is weaker under DuckDB than it was under Postgres — see the
note in basket_store.claim_next_chunk().

DuckDB has no server process and no connection URI — a "connection" here
IS the open database file handle, shared and reused for the life of one
Python process (unlike psycopg2, where reconnecting per call was cheap).
"""

import os
import config

# Overridable via an env var for the same reason pg_manager.py's PGDATA_DIR
# was — in case the default location ever needs to move.
DB_PATH = config.DUCKDB_PATH

_con = None  # module-level singleton — one shared connection per process


def get_connection():
    """
    Opens (or reuses) the local DuckDB database file at DB_PATH. Safe to
    call repeatedly within one process — the same open connection is
    reused, not reopened, after the first call. Callers should hold onto
    and reuse the returned connection object rather than calling this
    repeatedly per query (DuckDB is happiest with one long-lived connection
    per process, not many short-lived open/close cycles).
    """
    global _con
    if _con is None:
        import duckdb  # requirements.txt dependency

        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        print(f"Opening local DuckDB database at {DB_PATH} (embedded, no server process)...")
        _con = duckdb.connect(DB_PATH)
    return _con


def close():
    """
    Explicit close — not required for normal operation, but exposed for
    scripts/tests that want deterministic cleanup rather than relying on
    process exit.
    """
    global _con
    if _con is not None:
        _con.close()
        _con = None
