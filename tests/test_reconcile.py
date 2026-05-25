"""Unit tests for the reconciliation engine.

Two layers of coverage:

* the scalar classifier functions (pure, no I/O) -- one test per branch
* an end-to-end ``reconcile`` test built from a hand-rolled mini fixture
  that contains exactly one trade of each defect type plus a clean
  control, so every classification path is exercised against data we
  fully control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src` importable when pytest runs from the project root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.reconcile import (  # noqa: E402  (import after sys.path tweak)
    DUPLICATE,
    MATCHED,
    MISSING_FROM_BROKER,
    MISSING_FROM_LEDGER,
    PRICE_BREACH,
    PRICE_WITHIN_TOL,
    QUANTITY_MISMATCH,
    SYMBOL_MISMATCH,
    TIMESTAMP_MISMATCH,
    classify_price_diff,
    classify_quantity_diff,
    classify_symbol_diff,
    classify_timestamp_diff,
    reconcile,
    score_against_ground_truth,
    severity_score,
)


# ---------------------------------------------------------------------------
# Pure-function tests -- one per branch of each classifier
# ---------------------------------------------------------------------------


class TestClassifyPriceDiff:
    def test_exact_match(self):
        assert classify_price_diff(100.0, 100.0, tolerance_bps=10) == MATCHED

    def test_within_tolerance(self):
        # 100 -> 100.05 is 5 bps, tolerance 10 -> within
        assert (
            classify_price_diff(100.0, 100.05, tolerance_bps=10)
            == PRICE_WITHIN_TOL
        )

    def test_at_tolerance_boundary_is_within(self):
        # 100 -> 100.10 is exactly 10 bps -- inclusive boundary
        assert (
            classify_price_diff(100.0, 100.10, tolerance_bps=10)
            == PRICE_WITHIN_TOL
        )

    def test_breach(self):
        # 100 -> 101.0 is 100 bps, far outside tolerance 10
        assert classify_price_diff(100.0, 101.0, tolerance_bps=10) == PRICE_BREACH

    def test_negative_direction_uses_absolute_diff(self):
        assert classify_price_diff(100.0, 99.0, tolerance_bps=10) == PRICE_BREACH

    def test_zero_ledger_price_handled(self):
        # Pathological input: shouldn't divide by zero
        assert classify_price_diff(0.0, 0.0, tolerance_bps=10) == MATCHED
        assert classify_price_diff(0.0, 1.0, tolerance_bps=10) == PRICE_BREACH


class TestClassifyQuantityDiff:
    def test_match(self):
        assert classify_quantity_diff(100, 100) == MATCHED

    def test_mismatch(self):
        assert classify_quantity_diff(100, 101) == QUANTITY_MISMATCH

    def test_string_inputs_coerced(self):
        # Defensive: SQLite occasionally returns ints as strings
        assert classify_quantity_diff("500", 500) == MATCHED


class TestClassifySymbolDiff:
    def test_match(self):
        assert classify_symbol_diff("AAPL", "AAPL") == MATCHED

    def test_typo(self):
        assert classify_symbol_diff("AAPL", "AAPK") == SYMBOL_MISMATCH

    def test_case_sensitive(self):
        # Mismatched casing is a real defect: tickers are uppercase by convention
        assert classify_symbol_diff("AAPL", "aapl") == SYMBOL_MISMATCH


class TestClassifyTimestampDiff:
    def test_match_within_tolerance(self):
        a = pd.Timestamp("2026-01-15 10:00:00")
        b = pd.Timestamp("2026-01-15 10:30:00")  # 30 min, tolerance 60
        assert classify_timestamp_diff(a, b, tolerance_minutes=60) == MATCHED

    def test_mismatch_beyond_tolerance(self):
        a = pd.Timestamp("2026-01-15 10:00:00")
        b = pd.Timestamp("2026-01-15 15:00:00")  # 5 hours -- a TZ bug
        assert classify_timestamp_diff(a, b, tolerance_minutes=60) == TIMESTAMP_MISMATCH

    def test_negative_diff_uses_absolute(self):
        a = pd.Timestamp("2026-01-15 15:00:00")
        b = pd.Timestamp("2026-01-15 10:00:00")
        assert classify_timestamp_diff(a, b, tolerance_minutes=60) == TIMESTAMP_MISMATCH

    def test_accepts_string_input(self):
        assert (
            classify_timestamp_diff("2026-01-15 10:00", "2026-01-15 10:00", 60)
            == MATCHED
        )


class TestSeverityScore:
    def test_known_label(self):
        sev = {MATCHED: 0, PRICE_BREACH: 3}
        assert severity_score(PRICE_BREACH, sev) == 3

    def test_unknown_label_defaults_zero(self):
        assert severity_score("nonsense", {MATCHED: 0}) == 0


# ---------------------------------------------------------------------------
# End-to-end ``reconcile`` test on a hand-built mini fixture
# ---------------------------------------------------------------------------


def _minimal_config() -> dict:
    """Self-contained config dict for tests -- no file IO."""
    return {
        "reconciliation": {
            "price_tolerance_bps": 10,
            "timestamp_tolerance_minutes": 60,
            "severity": {
                MATCHED: 0,
                PRICE_WITHIN_TOL: 1,
                SYMBOL_MISMATCH: 1,
                TIMESTAMP_MISMATCH: 1,
                QUANTITY_MISMATCH: 2,
                DUPLICATE: 2,
                PRICE_BREACH: 3,
                MISSING_FROM_BROKER: 3,
                MISSING_FROM_LEDGER: 3,
            },
        }
    }


@pytest.fixture
def mini_fixture():
    """Build (ledger, broker, orphans, expected_defects) covering every defect type.

    Eight trades total:
    * T001 -- clean match (no exception expected)
    * T002 -- missing_from_broker (dropped from broker)
    * T003 -- price within tolerance (+5 bps)
    * T004 -- price breach (+100 bps)
    * T005 -- quantity mismatch (qty differs)
    * T006 -- symbol mismatch (typo)
    * T007 -- timestamp mismatch (5h shift)
    * T008 -- duplicate (broker reports twice)
    * B001 -- missing_from_ledger (broker-only)
    """
    base_ts = pd.Timestamp("2026-01-15 10:00:00")
    ledger = pd.DataFrame(
        [
            {"trade_id": "T001", "timestamp": base_ts, "symbol": "AAPL", "side": "BUY", "quantity": 100, "price": 200.00},
            {"trade_id": "T002", "timestamp": base_ts, "symbol": "MSFT", "side": "SELL", "quantity": 200, "price": 300.00},
            {"trade_id": "T003", "timestamp": base_ts, "symbol": "GOOGL", "side": "BUY", "quantity": 100, "price": 100.00},
            {"trade_id": "T004", "timestamp": base_ts, "symbol": "AMZN", "side": "BUY", "quantity": 100, "price": 150.00},
            {"trade_id": "T005", "timestamp": base_ts, "symbol": "META", "side": "BUY", "quantity": 500, "price": 250.00},
            {"trade_id": "T006", "timestamp": base_ts, "symbol": "NVDA", "side": "BUY", "quantity": 100, "price": 400.00},
            {"trade_id": "T007", "timestamp": base_ts, "symbol": "TSLA", "side": "SELL", "quantity": 100, "price": 180.00},
            {"trade_id": "T008", "timestamp": base_ts, "symbol": "JPM", "side": "BUY", "quantity": 100, "price": 175.00},
        ]
    )
    broker_rows = [
        # T001 clean copy
        {"trade_id": "T001", "timestamp": base_ts, "symbol": "AAPL", "side": "BUY", "quantity": 100, "price": 200.00},
        # T002 deliberately absent (missing_from_broker)
        # T003 +5 bps -> 100.05 (within tol 10)
        {"trade_id": "T003", "timestamp": base_ts, "symbol": "GOOGL", "side": "BUY", "quantity": 100, "price": 100.05},
        # T004 +100 bps -> 151.50 (breach)
        {"trade_id": "T004", "timestamp": base_ts, "symbol": "AMZN", "side": "BUY", "quantity": 100, "price": 151.50},
        # T005 quantity wrong
        {"trade_id": "T005", "timestamp": base_ts, "symbol": "META", "side": "BUY", "quantity": 400, "price": 250.00},
        # T006 symbol typo
        {"trade_id": "T006", "timestamp": base_ts, "symbol": "NVDB", "side": "BUY", "quantity": 100, "price": 400.00},
        # T007 timestamp shifted 5 hours
        {"trade_id": "T007", "timestamp": base_ts + pd.Timedelta(hours=5), "symbol": "TSLA", "side": "SELL", "quantity": 100, "price": 180.00},
        # T008 broker reports twice -- duplicate
        {"trade_id": "T008", "timestamp": base_ts, "symbol": "JPM", "side": "BUY", "quantity": 100, "price": 175.00},
        {"trade_id": "T008", "timestamp": base_ts, "symbol": "JPM", "side": "BUY", "quantity": 100, "price": 175.00},
        # B001 broker-only fill -- missing_from_ledger
        {"trade_id": "B001", "timestamp": base_ts, "symbol": "WFC", "side": "BUY", "quantity": 100, "price": 50.00},
    ]
    broker = pd.DataFrame(broker_rows)

    # Build the orphans frame in the shape ``database.find_orphans`` returns.
    orphans = pd.DataFrame(
        [
            {
                "orphan_side": "missing_from_broker",
                "trade_id": "T002",
                "timestamp": base_ts,
                "symbol": "MSFT",
                "side": "SELL",
                "quantity": 200,
                "price": 300.00,
            },
            {
                "orphan_side": "missing_from_ledger",
                "trade_id": "B001",
                "timestamp": base_ts,
                "symbol": "WFC",
                "side": "BUY",
                "quantity": 100,
                "price": 50.00,
            },
        ]
    )

    expected = {
        ("T002", MISSING_FROM_BROKER),
        ("B001", MISSING_FROM_LEDGER),
        ("T003", PRICE_WITHIN_TOL),
        ("T004", PRICE_BREACH),
        ("T005", QUANTITY_MISMATCH),
        ("T006", SYMBOL_MISMATCH),
        ("T007", TIMESTAMP_MISMATCH),
        ("T008", DUPLICATE),
    }
    return ledger, broker, orphans, expected


def test_reconcile_detects_every_defect_type(mini_fixture):
    ledger, broker, orphans, expected = mini_fixture
    cfg = _minimal_config()

    exceptions = reconcile(ledger, broker, orphans, cfg)

    detected = set(zip(exceptions["trade_id"].astype(str), exceptions["classification"]))
    missing = expected - detected
    extra = detected - expected
    assert not missing, f"reconcile failed to detect: {sorted(missing)}"
    assert not extra, f"reconcile produced unexpected exceptions: {sorted(extra)}"


def test_reconcile_emits_correct_severities(mini_fixture):
    ledger, broker, orphans, _ = mini_fixture
    cfg = _minimal_config()
    sev_map = cfg["reconciliation"]["severity"]

    exceptions = reconcile(ledger, broker, orphans, cfg)

    for _, row in exceptions.iterrows():
        assert int(row["severity"]) == sev_map[row["classification"]], (
            f"severity mismatch for {row['classification']}"
        )


def test_reconcile_leaves_clean_trade_out_of_exceptions(mini_fixture):
    ledger, broker, orphans, _ = mini_fixture
    cfg = _minimal_config()

    exceptions = reconcile(ledger, broker, orphans, cfg)

    assert "T001" not in set(exceptions["trade_id"]), (
        "clean trade T001 should produce zero exception rows"
    )


def test_scoring_against_ground_truth_round_trips(mini_fixture):
    ledger, broker, orphans, expected = mini_fixture
    cfg = _minimal_config()
    exceptions = reconcile(ledger, broker, orphans, cfg)

    # Build a ground-truth table using the ingest-side defect_type vocabulary.
    gt_rows = [
        ("T002", "missing_from_broker"),
        ("B001", "missing_from_ledger"),
        ("T003", "price_mismatch_within_tol"),
        ("T004", "price_mismatch_breach"),
        ("T005", "quantity_mismatch"),
        ("T006", "symbol_typo"),
        ("T007", "timestamp_shift"),
        ("T008", "duplicate"),
    ]
    ground_truth = pd.DataFrame(gt_rows, columns=["trade_id", "defect_type"])

    metrics = score_against_ground_truth(exceptions, ground_truth)
    assert metrics["tp"] == len(expected)
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
