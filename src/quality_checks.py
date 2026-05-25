"""Structural data-quality checks on the raw trade feeds.

These run *before* (or alongside) reconciliation and answer questions
like "does the file even look right?" -- missing columns, nulls in key
fields, prices outside a sane range, sides that aren't BUY/SELL.

Design choices:

* Checks **return** structured :class:`CheckResult` records; they never
  raise. The caller decides whether to halt the pipeline. This matches
  how a real ops tool surfaces issues for triage rather than crashing.
* All thresholds (price/quantity bounds, allowed sides) come from
  ``config.yaml`` so the contract for "valid" is configurable per
  environment without code changes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.ingest import PROJECT_ROOT, configure_logging, load_config


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Logical type names rather than literal pandas dtypes -- pandas dtypes vary
# between platforms (e.g. int64 vs Int64) and aren't useful for end-user
# error messages. The check normalises before comparing.
LEDGER_SCHEMA: dict[str, str] = {
    "trade_id": "string",
    "timestamp": "datetime",
    "symbol": "string",
    "side": "string",
    "quantity": "int",
    "price": "float",
}
BROKER_SCHEMA: dict[str, str] = dict(LEDGER_SCHEMA)

NON_NULL_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "symbol",
    "side",
    "quantity",
    "price",
)

VALID_SIDES: frozenset[str] = frozenset({"BUY", "SELL"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single data-quality check on a single table."""

    table: str
    check: str
    severity: str  # "info" | "warn" | "error"
    passed: bool
    rows_failed: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _logical_dtype(series: pd.Series) -> str:
    """Map a pandas Series dtype to our logical-type vocabulary."""
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "string"


def check_schema(df: pd.DataFrame, expected: dict[str, str], table: str) -> list[CheckResult]:
    """Verify every expected column is present with a compatible type."""
    results: list[CheckResult] = []
    missing = [c for c in expected if c not in df.columns]
    if missing:
        results.append(
            CheckResult(
                table=table,
                check="schema.missing_columns",
                severity="error",
                passed=False,
                rows_failed=0,
                detail=f"missing columns: {sorted(missing)}",
            )
        )
    extras = [c for c in df.columns if c not in expected]
    if extras:
        # Extras are informational, not an error -- downstream code only reads
        # the columns it knows about.
        results.append(
            CheckResult(
                table=table,
                check="schema.extra_columns",
                severity="info",
                passed=True,
                rows_failed=0,
                detail=f"extra columns present: {sorted(extras)}",
            )
        )
    type_errors = []
    for col, want in expected.items():
        if col not in df.columns:
            continue
        got = _logical_dtype(df[col])
        if got != want:
            type_errors.append(f"{col}: expected {want}, got {got}")
    if type_errors:
        results.append(
            CheckResult(
                table=table,
                check="schema.type_mismatch",
                severity="error",
                passed=False,
                rows_failed=0,
                detail="; ".join(type_errors),
            )
        )
    if not results:
        results.append(
            CheckResult(
                table=table,
                check="schema",
                severity="info",
                passed=True,
                detail=f"all {len(expected)} expected columns present with correct types",
            )
        )
    return results


def check_nulls(
    df: pd.DataFrame,
    columns: Iterable[str],
    table: str,
) -> list[CheckResult]:
    """Flag any non-nullable column that contains a null."""
    results: list[CheckResult] = []
    for col in columns:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        results.append(
            CheckResult(
                table=table,
                check=f"nulls.{col}",
                severity="error" if n_null else "info",
                passed=n_null == 0,
                rows_failed=n_null,
                detail=f"{n_null} null(s) in {col}",
            )
        )
    return results


def check_value_ranges(
    df: pd.DataFrame,
    qc_cfg: dict,
    table: str,
) -> list[CheckResult]:
    """Detect prices and quantities outside the configured sane bounds."""
    results: list[CheckResult] = []
    if "price" in df.columns:
        lo, hi = qc_cfg["price_min"], qc_cfg["price_max"]
        bad = df["price"].lt(lo) | df["price"].gt(hi) | df["price"].isna()
        n_bad = int(bad.sum())
        results.append(
            CheckResult(
                table=table,
                check="range.price",
                severity="warn" if n_bad else "info",
                passed=n_bad == 0,
                rows_failed=n_bad,
                detail=f"price outside [{lo}, {hi}]: {n_bad} row(s)",
            )
        )
    if "quantity" in df.columns:
        lo, hi = qc_cfg["quantity_min"], qc_cfg["quantity_max"]
        bad = df["quantity"].lt(lo) | df["quantity"].gt(hi) | df["quantity"].isna()
        n_bad = int(bad.sum())
        results.append(
            CheckResult(
                table=table,
                check="range.quantity",
                severity="warn" if n_bad else "info",
                passed=n_bad == 0,
                rows_failed=n_bad,
                detail=f"quantity outside [{lo}, {hi}]: {n_bad} row(s)",
            )
        )
    return results


def check_vocabulary(df: pd.DataFrame, table: str) -> list[CheckResult]:
    """Flag rows whose ``side`` is not BUY or SELL."""
    if "side" not in df.columns:
        return []
    bad = ~df["side"].isin(VALID_SIDES)
    n_bad = int(bad.sum())
    return [
        CheckResult(
            table=table,
            check="vocab.side",
            severity="error" if n_bad else "info",
            passed=n_bad == 0,
            rows_failed=n_bad,
            detail=f"side not in {sorted(VALID_SIDES)}: {n_bad} row(s)",
        )
    ]


def check_unique_ids(df: pd.DataFrame, table: str, allow_duplicates: bool) -> list[CheckResult]:
    """trade_id uniqueness -- warned on broker (duplicates are real), error on ledger."""
    if "trade_id" not in df.columns:
        return []
    dup_count = int(df["trade_id"].duplicated().sum())
    return [
        CheckResult(
            table=table,
            check="unique.trade_id",
            severity="warn" if allow_duplicates else "error",
            passed=dup_count == 0,
            rows_failed=dup_count,
            detail=f"{dup_count} duplicate trade_id(s)",
        )
    ]


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_all(
    ledger: pd.DataFrame,
    broker: pd.DataFrame,
    cfg: dict,
) -> list[CheckResult]:
    """Run every check on both feeds and return a flat list of results."""
    qc_cfg = cfg["quality_checks"]
    results: list[CheckResult] = []

    for table, df, schema, allow_dup in (
        ("internal_ledger", ledger, LEDGER_SCHEMA, False),
        ("broker_feed", broker, BROKER_SCHEMA, True),
    ):
        results.extend(check_schema(df, schema, table))
        results.extend(check_nulls(df, NON_NULL_COLUMNS, table))
        results.extend(check_value_ranges(df, qc_cfg, table))
        results.extend(check_vocabulary(df, table))
        results.extend(check_unique_ids(df, table, allow_duplicates=allow_dup))

    return results


def to_dataframe(results: list[CheckResult]) -> pd.DataFrame:
    """Flatten check results into a DataFrame for CSV / dashboard rendering."""
    if not results:
        return pd.DataFrame(columns=["table", "check", "severity", "passed", "rows_failed", "detail"])
    return pd.DataFrame([r.as_dict() for r in results])


def summarise(results: list[CheckResult]) -> dict[str, int]:
    """Return a count of results by severity (handy for the report stage)."""
    summary = {"info": 0, "warn": 0, "error": 0, "total": len(results), "failed": 0}
    for r in results:
        summary[r.severity] = summary.get(r.severity, 0) + 1
        if not r.passed:
            summary["failed"] += 1
    return summary


# ---------------------------------------------------------------------------
# Module CLI -- load feeds from CSV and report
# ---------------------------------------------------------------------------


def run(config_path: str | Path = "config.yaml") -> dict:
    cfg = load_config(config_path)
    configure_logging(cfg)
    log = logging.getLogger("quality_checks")

    ledger_csv = PROJECT_ROOT / cfg["paths"]["ledger_csv"]
    broker_csv = PROJECT_ROOT / cfg["paths"]["broker_csv"]
    for required in (ledger_csv, broker_csv):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing input {required} -- run `python -m src.ingest` first."
            )

    ledger = pd.read_csv(ledger_csv, parse_dates=["timestamp"])
    broker = pd.read_csv(broker_csv, parse_dates=["timestamp"])

    results = run_all(ledger, broker, cfg)
    summary = summarise(results)

    log.info(
        "Quality checks: %d run, %d failed (info=%d warn=%d error=%d)",
        summary["total"],
        summary["failed"],
        summary.get("info", 0),
        summary.get("warn", 0),
        summary.get("error", 0),
    )
    for r in results:
        if not r.passed:
            log.warning("[%s] %s :: %s", r.table, r.check, r.detail)

    return {"results": results, "summary": summary}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
