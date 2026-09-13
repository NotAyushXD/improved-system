"""
pg_manager.py

Self-contained PostgreSQL lifecycle for this pipeline — no manual install,
no manual `pg_ctl start`, no port/config wrangling. `pgserver` (a normal
`requirements.txt` dependency, installed ahead of time like every other
package here — this module NEVER shells out to pip or any other package
manager at runtime) bundles real Postgres binaries and knows how to
initialize + start a server rooted at a local data directory.

Why this exists: every basket-storage function in basket_store.py needs a
live Postgres connection, and none of them should have to know how that
server got started. This module is the one place that answers "is a server
running, and what's its connection string" — call get_connection_uri() and
get back a URI usable directly with psycopg2/pandas/sqlalchemy.

The server's data directory is deliberately NOT under data/output/ — output/
is "everything this pipeline WRITES as a deliverable" (per pipeline_main.py's
own convention); PGDATA_DIR is server state, not a deliverable, so it lives
alongside output/ instead of inside it.
"""

import atexit
import os

# Overridable via an env var so this can be pointed at a plain local folder
# if the default location turns out to sit on a drive/filesystem that
# doesn't support the permission changes `initdb` needs (e.g. a cloud-sync
# folder like OneDrive, or certain mapped/network drives) — initdb needs to
# set restrictive ACLs on this directory, which some drive types refuse.
PGDATA_DIR = os.environ.get("PIPELINE_PGDATA_DIR") or os.path.join("..", "data", "pgdata")

_server = None  # module-level singleton — one server per process


def get_connection_uri() -> str:
    """
    Starts (or reuses) the local pgserver-managed Postgres instance rooted at
    PGDATA_DIR, and returns a connection URI usable directly with psycopg2 /
    pandas / sqlalchemy.

    Safe to call repeatedly within one process — the same running server is
    reused, not restarted, after the first call.
    """
    global _server
    if _server is None:
        import pgserver  # requirements.txt dependency — never pip-installed here

        # Deliberately NOT pre-creating PGDATA_DIR here: initdb behaves
        # differently (and more reliably, permission-wise) when it creates
        # its data directory itself from scratch versus being handed one
        # that already exists — it takes an extra "fix up permissions on
        # an existing directory" path in the latter case, which can fail
        # outright on some drives/filesystems even though initdb creating
        # the directory itself, fresh, would have succeeded.
        print(f"Starting local self-contained Postgres instance at {PGDATA_DIR} "
              f"(via pgserver — no manual install/config needed)...")
        _server = pgserver.get_server(PGDATA_DIR)
        print(f"  Postgres ready: {_server.get_uri()}")
        atexit.register(shutdown)
    return _server.get_uri()


def shutdown():
    """
    Explicit shutdown — not required for normal operation (pgserver stops the
    server automatically when the owning process exits), but exposed for
    scripts/tests that want deterministic cleanup timing rather than relying
    on process-exit behavior.
    """
    global _server
    if _server is not None:
        try:
            _server.cleanup()
        except Exception as e:
            print(f"  (non-fatal) Postgres shutdown raised {type(e).__name__}: {e}")
        _server = None
