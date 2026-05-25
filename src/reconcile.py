"""Reconciliation engine: classify every trade and score against ground truth.

Phase 3 combines the SQL half (orphan detection from :mod:`src.database`)
with a Pandas comparison of the matched pairs. Each trade -- and each
defect on a trade -- ends up as a row in a long-format exceptions table.

Design notes
------------
* The single-comparison classifiers (:func:`classify_price_diff`,
  :func:`classify_quantity_diff`, :func:`classify_symbol_diff`,
  :func:`classify_timestamp_diff`) are pure, side-effect-free, and
  scalar-in / scalar-out. They are the unit-test surface called from
  ``tests/test_reconcile.py`` in Phase 4.
* The orchestration layer (:func:`reconcile`) applies the classifiers
  vectorised across the joined frame, then concatenates orphan rows
  (from SQL) and duplicate rows (from a group-by on the broker side).
* One trade_id can legitimately produce multiple exception rows -- e.g.
  a single broker row could differ on both price and timestamp. This
  matches the structure of the ground-truth table, which records one
  row per injected defect.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.database import (
    BROKER_TABLE,
    LEDGER_TABLE,
    find_orphans,
    load_feeds_from_csv,
    open_db,
    query,
)
from src.ingest import PROJECT_ROOT, configure_logging, load_config


# ---------------------------------------------------------------------------
# Classification labels (single source of truth so tests + scoring agree)
# ---------------------------------------------------------------------------

MATCHED = "matched"
MISSING_FROM_BROKER = "missing_from_broker"
MISSING_FROM_LEDGER = "missing_from_ledger"
DUPLICATE = "duplicate"
PRICE_WITHIN_TOL = "price_mismatch_within_tol"
PRICE_BREACH = "price_mismatch_breach"
QUANTITY_MISMATCH = "quantity_mismatch"
SYMBOL_MISMATCH = "symbol_mismatch"
TIMESTAMP_MISMATCH = "timestamp_mismatch"

# Maps the defect_type labels emitted by the ingest generator to the
# classification labels the reconciliation engine emits. Used only for
# scoring against ground truth.
GROUND_TRUTH_TO_CLASSIFICATION: dict[str, str] = {
    "missing_from_broker": MISSING_FROM_BROKER,
    "missing_from_ledger": MISSING_FROM_LEDGER,
    "duplicate": DUPLICATE,
    "price_mismatch_within_tol": PRICE_WITHIN_TOL,
    "price_mismatch_breach": PRICE_BREACH,
    "quantity_mismatch": QUANTITY_MISMATCH,
    "symbol_typo": SYMBOL_MISMATCH,
    "timestamp_shift": TIMESTAMP_MISMATCH,
}


# ---------------------------------------------------------------------------
# Pure classifier functions (the unit-test surface)
# ---------------------------------------------------------------------------


def classify_price_diff(
    ledger_price: float,
    broker_price: float,
    tolerance_bps: float,
) -> str:
    """Classify the price relationship between a ledger fill and a broker fill.

    Returns one of :data:`MATCHED`, :data:`PRICE_WITHIN_TOL`,
    :data:`PRICE_BREACH`. ``tolerance_bps`` is the allowable difference
    expressed in basis points of the ledger price.
    """
    if ledger_price <= 0:
        # Pathological: divide-by-zero guard. Any deviation is a breach.
        return MATCHED if ledger_price == broker_price else PRICE_BREACH
    diff_bps = abs(broker_price - ledger_price) / ledger_price * 10_000.0
    if diff_bps == 0:
        return MATCHED
    if diff_bps <= tolerance_bps:
        return PRICE_WITHIN_TOL
    return PRICE_BREACH


def classify_quantity_diff(ledger_quantity: int, broker_quantity: int) -> str:
    """Return :data:`MATCHED` if quantities agree, else :data:`QUANTITY_MISMATCH`."""
    return MATCHED if int(ledger_quantity) == int(broker_quantity) else QUANTITY_MISMATCH


def classify_symbol_diff(ledger_symbol: str, broker_symbol: str) -> str:
    """Return :data:`MATCHED` if symbols agree, else :data:`SYMBOL_MISMATCH`."""
    return MATCHED if str(ledger_symbol) == str(broker_symbol) else SYMBOL_MISMATCH


def classify_timestamp_diff(
    ledger_timestamp: pd.Timestamp | str,
    broker_timestamp: pd.Timestamp | str,
    tolerance_minutes: float,
) -> str:
    """Flag broker timestamps that drift from the ledger beyond ``tolerance_minutes``.

    Designed primarily to catch timezone-handling bugs where the broker
    reports a fill several hours off from the firm's local-time record.
    """
    lt = pd.Timestamp(ledger_timestamp)
    bt = pd.Timestamp(broker_timestamp)
    diff_minutes = abs((bt - lt).total_seconds()) / 60.0
    return MATCHED if diff_minutes <= tolerance_minutes else TIMESTAMP_MISMATCH


def severity_score(classification: str, severities: dict[str, int]) -> int:
    """Map a classification label to its configured severity (default 0)."""
    return int(severities.get(classification, 0))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


_EXCEPTION_COLUMNS = [
    "trade_id",
    "classification",
    "severity",
    "symbol",
    "side",
    "ledger_quantity",
    "broker_quantity",
    "ledger_price",
    "broker_price",
    "ledger_timestamp",
    "broker_timestamp",
    "details",
]


def _empty_exceptions() -> pd.DataFrame:
    return pd.DataFrame(columns=_EXCEPTION_COLUMNS)


def detect_duplicates(broker: pd.DataFrame, severities: dict[str, int]) -> pd.DataFrame:
    """Return one exception row per broker trade_id that appears more than once."""
    counts = broker.groupby("trade_id").size()
    dup_ids = counts[counts > 1].index
    if len(dup_ids) == 0:
        return _empty_exceptions()
    first_rows = (
        broker[broker["trade_id"].isin(dup_ids)]
        .drop_duplicates("trade_id", keep="first")
        .reset_index(drop=True)
    )
    n = len(first_rows)
    out = pd.DataFrame(
        {
            "trade_id": first_rows["trade_id"],
            "classification": DUPLICATE,
            "severity": severity_score(DUPLICATE, severities),
            "symbol": first_rows["symbol"],
            "side": first_rows["side"],
            "ledger_quantity": pd.Series([pd.NA] * n, dtype="Int64"),
            "broker_quantity": first_rows["quantity"].astype("Int64"),
            "ledger_price": pd.Series([np.nan] * n, dtype=float),
            "broker_price": first_rows["price"].astype(float),
            "ledger_timestamp": pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]"),
            "broker_timestamp": pd.to_datetime(first_rows["timestamp"]),
            "details": [f"copies={counts.loc[tid]}" for tid in first_rows["trade_id"]],
        }
    )
    return out[_EXCEPTION_COLUMNS]


def _orphans_to_exceptions(orphans: pd.DataFrame, severities: dict[str, int]) -> pd.DataFrame:
    """Reshape the SQL orphan frame into the canonical exceptions schema."""
    if orphans.empty:
        return _empty_exceptions()
    orphans = orphans.reset_index(drop=True)
    n = len(orphans)
    is_missing_broker = orphans["orphan_side"].eq("missing_from_broker").to_numpy()
    ts = pd.to_datetime(orphans["timestamp"])

    # Build NaT-filled columns then fill per-side; this preserves datetime64
    # dtype, which np.where would otherwise collapse to int64 nanoseconds.
    ledger_ts = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    broker_ts = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    ledger_ts.loc[is_missing_broker] = ts.loc[is_missing_broker]
    broker_ts.loc[~is_missing_broker] = ts.loc[~is_missing_broker]

    ledger_qty = pd.Series([pd.NA] * n, dtype="Int64")
    broker_qty = pd.Series([pd.NA] * n, dtype="Int64")
    ledger_qty.loc[is_missing_broker] = orphans.loc[is_missing_broker, "quantity"].astype("Int64").to_numpy()
    broker_qty.loc[~is_missing_broker] = orphans.loc[~is_missing_broker, "quantity"].astype("Int64").to_numpy()

    ledger_price = pd.Series([np.nan] * n, dtype=float)
    broker_price = pd.Series([np.nan] * n, dtype=float)
    ledger_price.loc[is_missing_broker] = orphans.loc[is_missing_broker, "price"].to_numpy()
    broker_price.loc[~is_missing_broker] = orphans.loc[~is_missing_broker, "price"].to_numpy()

    out = pd.DataFrame(
        {
            "trade_id": orphans["trade_id"],
            "classification": orphans["orphan_side"],
            "severity": orphans["orphan_side"].map(lambda s: severity_score(s, severities)),
            "symbol": orphans["symbol"],
            "side": orphans["side"],
            "ledger_quantity": ledger_qty,
            "broker_quantity": broker_qty,
            "ledger_price": ledger_price,
            "broker_price": broker_price,
            "ledger_timestamp": ledger_ts,
            "broker_timestamp": broker_ts,
            "details": "",
        }
    )
    return out[_EXCEPTION_COLUMNS]


def _compare_matched_pairs(
    ledger: pd.DataFrame,
    broker: pd.DataFrame,
    tolerance_bps: float,
    timestamp_tol_minutes: float,
    severities: dict[str, int],
) -> pd.DataFrame:
    """Inner-join the two feeds and emit one exception row per detected defect.

    Vectorised counterpart to the scalar classifier functions: applying
    ``classify_*`` row-by-row would be O(n) Python calls; here the same
    decision rules run in numpy.
    """
    if ledger.empty or broker.empty:
        return _empty_exceptions()

    broker_unique = broker.drop_duplicates("trade_id", keep="first")
    pairs = ledger.merge(broker_unique, on="trade_id", suffixes=("_l", "_b"), how="inner")
    if pairs.empty:
        return _empty_exceptions()

    pairs["ledger_timestamp"] = pd.to_datetime(pairs["timestamp_l"])
    pairs["broker_timestamp"] = pd.to_datetime(pairs["timestamp_b"])

    ledger_price = pairs["price_l"].to_numpy(dtype=float)
    broker_price = pairs["price_b"].to_numpy(dtype=float)
    ledger_qty = pairs["quantity_l"].to_numpy(dtype=int)
    broker_qty = pairs["quantity_b"].to_numpy(dtype=int)
    ledger_sym = pairs["symbol_l"].to_numpy()
    broker_sym = pairs["symbol_b"].to_numpy()
    ts_diff_min = (
        np.abs((pairs["broker_timestamp"] - pairs["ledger_timestamp"]).dt.total_seconds())
        / 60.0
    ).to_numpy()

    # Price classification, vectorised. Guard the divide-by-zero with a mask.
    safe_ledger = np.where(ledger_price > 0, ledger_price, np.nan)
    price_diff_bps = np.abs(broker_price - ledger_price) / safe_ledger * 10_000.0
    price_within = (price_diff_bps > 0) & (price_diff_bps <= tolerance_bps)
    price_breach = price_diff_bps > tolerance_bps

    qty_mismatch = ledger_qty != broker_qty
    sym_mismatch = ledger_sym != broker_sym
    ts_mismatch = ts_diff_min > timestamp_tol_minutes

    exceptions: list[pd.DataFrame] = []

    def _emit(mask: np.ndarray, label: str, details: Iterable[str] | str = "") -> None:
        if not mask.any():
            return
        sub = pairs.loc[mask]
        details_arr = details if isinstance(details, str) else list(details)
        exceptions.append(
            pd.DataFrame(
                {
                    "trade_id": sub["trade_id"].to_numpy(),
                    "classification": label,
                    "severity": severity_score(label, severities),
                    "symbol": sub["symbol_l"].to_numpy(),
                    "side": sub["side_l"].to_numpy(),
                    "ledger_quantity": sub["quantity_l"].to_numpy(),
                    "broker_quantity": sub["quantity_b"].to_numpy(),
                    "ledger_price": sub["price_l"].to_numpy(),
                    "broker_price": sub["price_b"].to_numpy(),
                    "ledger_timestamp": sub["ledger_timestamp"].to_numpy(),
                    "broker_timestamp": sub["broker_timestamp"].to_numpy(),
                    "details": details_arr,
                }
            )[_EXCEPTION_COLUMNS]
        )

    _emit(
        price_within,
        PRICE_WITHIN_TOL,
        details=[f"diff_bps={d:.2f}" for d in price_diff_bps[price_within]],
    )
    _emit(
        price_breach,
        PRICE_BREACH,
        details=[f"diff_bps={d:.2f}" for d in price_diff_bps[price_breach]],
    )
    _emit(qty_mismatch, QUANTITY_MISMATCH)
    _emit(
        sym_mismatch,
        SYMBOL_MISMATCH,
        details=[
            f"ledger={l} broker={b}"
            for l, b in zip(ledger_sym[sym_mismatch], broker_sym[sym_mismatch])
        ],
    )
    _emit(
        ts_mismatch,
        TIMESTAMP_MISMATCH,
        details=[f"diff_minutes={m:.1f}" for m in ts_diff_min[ts_mismatch]],
    )

    if not exceptions:
        return _empty_exceptions()
    return pd.concat(exceptions, ignore_index=True)


def reconcile(
    ledger: pd.DataFrame,
    broker: pd.DataFrame,
    orphans: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Return a long-format exceptions table for every defect detected."""
    rec_cfg = cfg["reconciliation"]
    severities = rec_cfg["severity"]
    parts = [
        _orphans_to_exceptions(orphans, severities),
        detect_duplicates(broker, severities),
        _compare_matched_pairs(
            ledger=ledger,
            broker=broker,
            tolerance_bps=float(rec_cfg["price_tolerance_bps"]),
            timestamp_tol_minutes=float(rec_cfg["timestamp_tolerance_minutes"]),
            severities=severities,
        ),
    ]
    exceptions = pd.concat(parts, ignore_index=True)
    return exceptions.sort_values(
        ["severity", "classification", "trade_id"], ascending=[False, True, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------


def score_against_ground_truth(
    exceptions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """Compute precision/recall/F1 of detected vs injected defects.

    Returns a dict with overall metrics plus a per-class breakdown. Both
    sides are reduced to the set ``(trade_id, classification)`` --
    duplicate emissions on either side count once.
    """
    detected = set(
        zip(
            exceptions["trade_id"].astype(str),
            exceptions["classification"].astype(str),
        )
    )
    truth = set(
        zip(
            ground_truth["trade_id"].astype(str),
            ground_truth["defect_type"].map(GROUND_TRUTH_TO_CLASSIFICATION).astype(str),
        )
    )

    tp = detected & truth
    fp = detected - truth
    fn = truth - detected

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Per-class breakdown -- helps diagnose which detector is leaking.
    classes = sorted({c for _, c in truth} | {c for _, c in detected})
    per_class: dict[str, dict[str, int | float]] = {}
    for cls in classes:
        d_cls = {tid for (tid, c) in detected if c == cls}
        t_cls = {tid for (tid, c) in truth if c == cls}
        tp_c = len(d_cls & t_cls)
        fp_c = len(d_cls - t_cls)
        fn_c = len(t_cls - d_cls)
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
        per_class[cls] = {
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": p_c,
            "recall": r_c,
        }

    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _log_scoring(metrics: dict, logger: logging.Logger) -> None:
    logger.info(
        "Detection accuracy: precision=%.3f recall=%.3f f1=%.3f  (tp=%d fp=%d fn=%d)",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["tp"],
        metrics["fp"],
        metrics["fn"],
    )
    logger.info("Per-class breakdown:")
    for cls, m in sorted(metrics["per_class"].items()):
        logger.info(
            "  %-28s tp=%-4d fp=%-4d fn=%-4d precision=%.3f recall=%.3f",
            cls,
            m["tp"],
            m["fp"],
            m["fn"],
            m["precision"],
            m["recall"],
        )


def run(config_path: str | Path = "config.yaml") -> dict:
    """End-to-end Phase 3: load -> reconcile -> score -> write exceptions CSV."""
    cfg = load_config(config_path)
    logger = configure_logging(cfg)
    log = logging.getLogger("reconcile")

    db_path = PROJECT_ROOT / cfg["paths"]["sqlite_db"]
    ledger_csv = PROJECT_ROOT / cfg["paths"]["ledger_csv"]
    broker_csv = PROJECT_ROOT / cfg["paths"]["broker_csv"]
    truth_csv = PROJECT_ROOT / cfg["paths"]["ground_truth_csv"]
    out_csv = PROJECT_ROOT / cfg["reconciliation"]["exceptions_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for required in (ledger_csv, broker_csv, truth_csv):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing input {required} -- run `python -m src.ingest` first."
            )

    with open_db(db_path) as conn:
        load_feeds_from_csv(conn, ledger_csv, broker_csv, logger=log)
        orphans = find_orphans(conn)
        ledger = query(conn, f"SELECT * FROM {LEDGER_TABLE}")
        broker = query(conn, f"SELECT * FROM {BROKER_TABLE}")

    exceptions = reconcile(ledger, broker, orphans, cfg)
    exceptions.to_csv(out_csv, index=False)
    log.info(
        "Wrote %s (%d exception rows across %d unique trade_ids)",
        out_csv.relative_to(PROJECT_ROOT),
        len(exceptions),
        exceptions["trade_id"].nunique() if not exceptions.empty else 0,
    )

    ground_truth = pd.read_csv(truth_csv)
    metrics = score_against_ground_truth(exceptions, ground_truth)
    _log_scoring(metrics, log)

    if metrics["recall"] < 0.95:
        log.warning(
            "Recall %.3f is below the 95%% acceptance target -- check classifiers",
            metrics["recall"],
        )

    return {
        "exceptions": exceptions,
        "metrics": metrics,
        "exceptions_csv": out_csv,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
