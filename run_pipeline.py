"""Single end-to-end pipeline entry point.

Chains all five stages of the trade-reconciliation pipeline, driven
entirely by ``config.yaml``::

    ingest  ->  load  ->  reconcile  ->  quality_checks  ->  report

Each stage logs its own banner to ``logs/pipeline.log`` (and stderr) so a
re-run leaves a readable trail of what happened. Stages share a single
SQLite connection so the database is loaded once, not once per module.

Run from the project root::

    python run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when this script runs from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.database import (
    BROKER_TABLE,
    LEDGER_TABLE,
    find_orphans,
    load_feeds_from_csv,
    open_db,
    query,
)
from src.ingest import PROJECT_ROOT, configure_logging, load_config
from src.ingest import run as ingest_run
from src.quality_checks import run_all as quality_run_all
from src.quality_checks import summarise as quality_summarise
from src.quality_checks import to_dataframe as quality_to_df
from src.reconcile import reconcile, score_against_ground_truth
from src.report import build_summary, log_headline, write_summary


def _banner(logger: logging.Logger, stage_num: int, stage_name: str) -> None:
    bar = "=" * 64
    logger.info(bar)
    logger.info("STAGE %d/5 :: %s", stage_num, stage_name)
    logger.info(bar)


def run(config_path: str | Path = "config.yaml") -> dict:
    cfg = load_config(config_path)
    logger = configure_logging(cfg)
    log = logging.getLogger("pipeline")

    paths = cfg["paths"]
    rec_cfg = cfg["reconciliation"]
    qc_cfg = cfg["quality_checks"]

    db_path = PROJECT_ROOT / paths["sqlite_db"]
    ledger_csv = PROJECT_ROOT / paths["ledger_csv"]
    broker_csv = PROJECT_ROOT / paths["broker_csv"]
    truth_csv = PROJECT_ROOT / paths["ground_truth_csv"]
    exceptions_csv = PROJECT_ROOT / rec_cfg["exceptions_csv"]
    quality_csv = PROJECT_ROOT / qc_cfg["results_csv"]
    summary_json = PROJECT_ROOT / cfg["report"]["summary_json"]

    # ----- Stage 1: ingest -------------------------------------------------
    _banner(log, 1, "ingest -- generate ledger, broker feed, ground truth")
    ingest_run(config_path)

    # ----- Stage 2: load + SQL orphans ------------------------------------
    _banner(log, 2, "load -- write feeds to SQLite + FULL OUTER JOIN orphans")
    with open_db(db_path) as conn:
        load_feeds_from_csv(conn, ledger_csv, broker_csv, logger=log)
        orphans = find_orphans(conn)
        ledger = query(conn, f"SELECT * FROM {LEDGER_TABLE}")
        broker = query(conn, f"SELECT * FROM {BROKER_TABLE}")
    # SQLite has no native datetime type; round-tripped timestamps come back as
    # strings. Parse here so every downstream stage sees the canonical dtype.
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"])
    broker["timestamp"] = pd.to_datetime(broker["timestamp"])
    log.info("Orphans returned by SQL: %d", len(orphans))

    # ----- Stage 3: reconcile ---------------------------------------------
    _banner(log, 3, "reconcile -- classify every defect (Pandas + tolerance)")
    exceptions = reconcile(ledger, broker, orphans, cfg)
    exceptions_csv.parent.mkdir(parents=True, exist_ok=True)
    exceptions.to_csv(exceptions_csv, index=False)
    log.info(
        "Wrote %s (%d exceptions across %d unique trade_ids)",
        exceptions_csv.relative_to(PROJECT_ROOT),
        len(exceptions),
        exceptions["trade_id"].nunique() if not exceptions.empty else 0,
    )

    # ----- Stage 4: quality checks ----------------------------------------
    _banner(log, 4, "quality_checks -- schema / nulls / ranges / vocab")
    qc_results = quality_run_all(ledger, broker, cfg)
    qc_df = quality_to_df(qc_results)
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(quality_csv, index=False)
    qc_summary = quality_summarise(qc_results)
    log.info(
        "Quality checks: %d run, %d failed (info=%d warn=%d error=%d)",
        qc_summary["total"],
        qc_summary["failed"],
        qc_summary.get("info", 0),
        qc_summary.get("warn", 0),
        qc_summary.get("error", 0),
    )
    for r in qc_results:
        if not r.passed:
            log.warning("QC [%s] %s :: %s", r.table, r.check, r.detail)

    # ----- Stage 5: report ------------------------------------------------
    _banner(log, 5, "report -- roll up accuracy + headline metrics")
    ground_truth = pd.read_csv(truth_csv)
    accuracy = score_against_ground_truth(exceptions, ground_truth)
    summary = build_summary(ledger, broker, exceptions, qc_results, accuracy)
    write_summary(summary, summary_json)
    log.info("Wrote %s", summary_json.relative_to(PROJECT_ROOT))
    log_headline(summary, log)

    return {
        "exceptions": exceptions,
        "quality_results": qc_results,
        "accuracy": accuracy,
        "summary": summary,
        "summary_path": summary_json,
        "exceptions_path": exceptions_csv,
        "quality_results_path": quality_csv,
    }


if __name__ == "__main__":
    run()
