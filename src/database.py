"""SQLite layer: load both trade feeds and answer the orphan-finder query.

This module covers the SQL half of reconciliation. The Pandas half
(tolerance + fuzzy comparison) lives in ``src/reconcile.py`` and is added
in Phase 3.

Responsibilities:

* thin wrappers around ``sqlite3`` for connecting, loading DataFrames, and
  running parameterised queries that return DataFrames
* a single canonical query that finds every orphaned record on both sides
  (ledger row with no broker counterpart, broker row with no ledger
  counterpart) using a FULL OUTER JOIN emulated via two LEFT JOINs unioned

Run as a module from the project root to load the CSVs produced by
``src/ingest.py`` and print orphan counts versus the ground-truth table::

    python -m src.database
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from src.ingest import PROJECT_ROOT, configure_logging, load_config


LEDGER_TABLE = "internal_ledger"
BROKER_TABLE = "broker_feed"


# ---------------------------------------------------------------------------
# Connection / IO helpers
# ---------------------------------------------------------------------------


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (and if necessary create) the SQLite database at ``db_path``."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # Enforce foreign keys + return columns by name for ergonomic debugging.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def open_db(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Context-managed wrapper around :func:`connect`."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def load_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
) -> int:
    """Write ``df`` to ``table_name`` and return the row count written."""
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    return len(df)


def query(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence | dict | None = None,
) -> pd.DataFrame:
    """Run ``sql`` and return the result as a DataFrame."""
    return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Schema + bulk load
# ---------------------------------------------------------------------------


def load_feeds_from_csv(
    conn: sqlite3.Connection,
    ledger_csv: str | Path,
    broker_csv: str | Path,
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    """Load both CSV feeds into the DB and return per-table row counts."""
    log = logger or logging.getLogger("database")
    ledger = pd.read_csv(ledger_csv, parse_dates=["timestamp"])
    broker = pd.read_csv(broker_csv, parse_dates=["timestamp"])

    n_ledger = load_dataframe(conn, ledger, LEDGER_TABLE)
    n_broker = load_dataframe(conn, broker, BROKER_TABLE)

    # Indexes on the join key keep the orphan query fast as the data grows.
    # Note: broker can contain duplicate trade_ids (the broker-reported-twice
    # defect), so this is an ordinary index, not unique.
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_tid ON {LEDGER_TABLE}(trade_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{BROKER_TABLE}_tid ON {BROKER_TABLE}(trade_id)")
    conn.commit()

    log.info("Loaded %d rows into %s", n_ledger, LEDGER_TABLE)
    log.info("Loaded %d rows into %s", n_broker, BROKER_TABLE)
    return {LEDGER_TABLE: n_ledger, BROKER_TABLE: n_broker}


# ---------------------------------------------------------------------------
# Orphan-finder (FULL OUTER JOIN emulation)
# ---------------------------------------------------------------------------

# SQLite added native FULL OUTER JOIN in 3.39, but the LEFT-JOIN + UNION
# emulation works on every version and is the canonical illustration of the
# pattern in interviews -- so use it explicitly here.
ORPHANS_SQL = f"""
SELECT
    'missing_from_broker' AS orphan_side,
    l.trade_id            AS trade_id,
    l.timestamp           AS timestamp,
    l.symbol              AS symbol,
    l.side                AS side,
    l.quantity            AS quantity,
    l.price               AS price
FROM {LEDGER_TABLE} AS l
LEFT JOIN {BROKER_TABLE} AS b ON l.trade_id = b.trade_id
WHERE b.trade_id IS NULL

UNION ALL

SELECT
    'missing_from_ledger' AS orphan_side,
    b.trade_id            AS trade_id,
    b.timestamp           AS timestamp,
    b.symbol              AS symbol,
    b.side                AS side,
    b.quantity            AS quantity,
    b.price               AS price
FROM {BROKER_TABLE} AS b
LEFT JOIN {LEDGER_TABLE} AS l ON b.trade_id = l.trade_id
WHERE l.trade_id IS NULL

ORDER BY orphan_side, trade_id
"""


def find_orphans(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return every record present on exactly one side of the join.

    Result columns: ``orphan_side`` (``missing_from_broker`` |
    ``missing_from_ledger``), ``trade_id``, ``timestamp``, ``symbol``,
    ``side``, ``quantity``, ``price``.
    """
    return query(conn, ORPHANS_SQL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _verify_against_ground_truth(
    orphans: pd.DataFrame,
    ground_truth_csv: str | Path,
    logger: logging.Logger,
) -> dict[str, dict[str, int]]:
    """Compare orphan counts to the injected-defect ground truth.

    Returns a nested dict ``{side: {"observed": n, "expected": n}}`` and
    logs a warning if any count disagrees.
    """
    gt = pd.read_csv(ground_truth_csv)
    expected = {
        "missing_from_broker": int((gt["defect_type"] == "missing_from_broker").sum()),
        "missing_from_ledger": int((gt["defect_type"] == "missing_from_ledger").sum()),
    }
    observed = orphans["orphan_side"].value_counts().to_dict()

    summary: dict[str, dict[str, int]] = {}
    for side, want in expected.items():
        got = int(observed.get(side, 0))
        summary[side] = {"observed": got, "expected": want}
        match = "OK" if got == want else "MISMATCH"
        logger.info("%-20s observed=%-4d expected=%-4d  %s", side, got, want, match)
        if got != want:
            logger.warning(
                "Orphan count for %s disagrees with ground truth (%d vs %d)",
                side,
                got,
                want,
            )
    return summary


def run(config_path: str | Path = "config.yaml") -> dict:
    """Load feeds into SQLite, run the orphan query, verify against truth."""
    cfg = load_config(config_path)
    logger = configure_logging(cfg)
    log = logging.getLogger("database")

    db_path = PROJECT_ROOT / cfg["paths"]["sqlite_db"]
    ledger_csv = PROJECT_ROOT / cfg["paths"]["ledger_csv"]
    broker_csv = PROJECT_ROOT / cfg["paths"]["broker_csv"]
    truth_csv = PROJECT_ROOT / cfg["paths"]["ground_truth_csv"]

    for required in (ledger_csv, broker_csv, truth_csv):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing input {required} -- run `python -m src.ingest` first."
            )

    with open_db(db_path) as conn:
        counts = load_feeds_from_csv(conn, ledger_csv, broker_csv, logger=log)
        orphans = find_orphans(conn)

    log.info("Orphan query returned %d rows", len(orphans))
    verification = _verify_against_ground_truth(orphans, truth_csv, log)

    return {
        "db_path": db_path,
        "row_counts": counts,
        "orphans": orphans,
        "verification": verification,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
