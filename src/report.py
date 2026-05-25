"""Pipeline reporting: roll exceptions + QC + accuracy into one summary.

The dashboard (Phase 5) reads this summary plus the exceptions CSV.
Splitting the rollup out of reconcile.py keeps reconcile focused on
classification logic and gives the pipeline a clean final stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.ingest import PROJECT_ROOT, configure_logging, load_config
from src.quality_checks import CheckResult, summarise as summarise_quality


def build_summary(
    ledger: pd.DataFrame,
    broker: pd.DataFrame,
    exceptions: pd.DataFrame,
    quality_results: list[CheckResult],
    accuracy: dict,
) -> dict:
    """Assemble the headline numbers shown in the dashboard."""
    n_ledger = int(len(ledger))
    n_broker = int(len(broker))

    # A trade is "clean" iff it appears on both sides without any exception
    # rows. Count via set difference: distinct ledger trade_ids minus those
    # that show up in the exceptions table.
    exception_ids = (
        set(exceptions["trade_id"].astype(str)) if not exceptions.empty else set()
    )
    ledger_ids = set(ledger["trade_id"].astype(str))
    clean_ids = ledger_ids - exception_ids
    reconciliation_rate = (len(clean_ids) / n_ledger) if n_ledger else 0.0

    by_type = (
        exceptions["classification"].value_counts().to_dict()
        if not exceptions.empty
        else {}
    )
    by_severity = (
        {int(k): int(v) for k, v in exceptions["severity"].value_counts().items()}
        if not exceptions.empty
        else {}
    )

    return {
        "trades": {
            "ledger_rows": n_ledger,
            "broker_rows": n_broker,
            "clean_trades": len(clean_ids),
            "trades_with_exceptions": len(exception_ids & ledger_ids),
            "broker_only_trade_ids": len(
                set(broker["trade_id"].astype(str)) - ledger_ids
            ),
        },
        "reconciliation_rate": round(reconciliation_rate, 6),
        "exceptions": {
            "total_rows": int(len(exceptions)),
            "unique_trade_ids": int(len(exception_ids)),
            "by_type": {str(k): int(v) for k, v in by_type.items()},
            "by_severity": by_severity,
        },
        "quality_checks": summarise_quality(quality_results),
        "detection_accuracy": {
            "precision": round(accuracy.get("precision", 0.0), 6),
            "recall": round(accuracy.get("recall", 0.0), 6),
            "f1": round(accuracy.get("f1", 0.0), 6),
            "tp": int(accuracy.get("tp", 0)),
            "fp": int(accuracy.get("fp", 0)),
            "fn": int(accuracy.get("fn", 0)),
            "per_class": {
                k: {
                    "tp": int(v["tp"]),
                    "fp": int(v["fp"]),
                    "fn": int(v["fn"]),
                    "precision": round(v["precision"], 6),
                    "recall": round(v["recall"], 6),
                }
                for k, v in accuracy.get("per_class", {}).items()
            },
        },
    }


def write_summary(summary: dict, out_path: str | Path) -> Path:
    """Write the summary dict as pretty-printed JSON. Returns the path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
    return path


def log_headline(summary: dict, logger: logging.Logger) -> None:
    """Emit the one-line headline numbers an analyst would skim first."""
    t = summary["trades"]
    e = summary["exceptions"]
    a = summary["detection_accuracy"]
    logger.info(
        "RECONCILIATION  rate=%.2f%%  clean=%d/%d  exceptions=%d (across %d ids)",
        summary["reconciliation_rate"] * 100.0,
        t["clean_trades"],
        t["ledger_rows"],
        e["total_rows"],
        e["unique_trade_ids"],
    )
    logger.info(
        "DETECTION ACC   precision=%.3f recall=%.3f f1=%.3f (tp=%d fp=%d fn=%d)",
        a["precision"],
        a["recall"],
        a["f1"],
        a["tp"],
        a["fp"],
        a["fn"],
    )


def run(config_path: str | Path = "config.yaml") -> dict:
    """Build the summary from already-produced artefacts on disk."""
    cfg = load_config(config_path)
    configure_logging(cfg)
    log = logging.getLogger("report")

    paths = cfg["paths"]
    rec_cfg = cfg["reconciliation"]
    ledger = pd.read_csv(PROJECT_ROOT / paths["ledger_csv"])
    broker = pd.read_csv(PROJECT_ROOT / paths["broker_csv"])
    exceptions_path = PROJECT_ROOT / rec_cfg["exceptions_csv"]
    if not exceptions_path.exists():
        raise FileNotFoundError(
            f"{exceptions_path} not found -- run `python -m src.reconcile` first."
        )
    exceptions = pd.read_csv(exceptions_path)

    # Re-score against ground truth so the summary is self-contained.
    from src.reconcile import score_against_ground_truth

    gt = pd.read_csv(PROJECT_ROOT / paths["ground_truth_csv"])
    accuracy = score_against_ground_truth(exceptions, gt)

    # No quality results available standalone -- run them too.
    from src.quality_checks import run_all as qc_run_all

    quality_results = qc_run_all(ledger, broker, cfg)

    summary = build_summary(ledger, broker, exceptions, quality_results, accuracy)
    out_path = PROJECT_ROOT / cfg["report"]["summary_json"]
    write_summary(summary, out_path)
    log.info("Wrote %s", out_path.relative_to(PROJECT_ROOT))
    log_headline(summary, log)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
