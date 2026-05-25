"""Data ingestion: market prices, synthetic ledger, and corrupted broker feed.

This module produces the three CSVs that drive every downstream phase:

* ``internal_ledger.csv``  -- the firm's own record of executed trades
* ``broker_feed.csv``      -- the broker's confirmation, with synthetic defects
* ``ground_truth.csv``     -- one row per injected defect, used later to score
                              reconciliation accuracy

All tunable values (tickers, corruption rates, file paths, RNG seed) are read
from ``config.yaml`` -- nothing in this file is hardcoded.

Run as a module from the project root::

    python -m src.ingest
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


# ---------------------------------------------------------------------------
# Config / logging helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load ``config.yaml`` from the project root and return a plain dict."""
    cfg_path = (PROJECT_ROOT / path) if not Path(path).is_absolute() else Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def configure_logging(cfg: dict) -> logging.Logger:
    """Configure root logging to both the configured log file and the console."""
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s %(levelname)s %(name)s :: %(message)s")

    log_path = PROJECT_ROOT / cfg["paths"]["log_file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any pre-existing handlers so reruns don't double-log.
    root.handlers.clear()

    formatter = logging.Formatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, ticker: str, period: str, interval: str) -> Path:
    return cache_dir / f"{ticker}_{period}_{interval}.csv"


def fetch_market_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """Pull daily bars for the configured tickers, caching each to disk.

    Returns a long-format DataFrame with columns
    ``[date, symbol, low, high, close]``. Cached CSVs make reruns offline-safe
    and deterministic.
    """
    md_cfg = cfg["market_data"]
    cache_dir = PROJECT_ROOT / cfg["paths"]["market_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ticker in md_cfg["tickers"]:
        cache_file = _cache_path(cache_dir, ticker, md_cfg["period"], md_cfg["interval"])
        if cache_file.exists():
            logger.debug("Loading cached market data for %s", ticker)
            df = pd.read_csv(cache_file, parse_dates=["date"])
        else:
            logger.info("Fetching market data for %s", ticker)
            raw = yf.download(
                ticker,
                period=md_cfg["period"],
                interval=md_cfg["interval"],
                progress=False,
                auto_adjust=False,
            )
            if raw.empty:
                logger.warning("No data returned for %s -- skipping", ticker)
                continue
            # yfinance under newer versions returns a MultiIndex on columns even
            # for a single ticker (level 0 = field, level 1 = ticker). Flatten
            # by keeping the field name.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw.index),
                    "symbol": ticker,
                    "low": raw["Low"].to_numpy(dtype=float),
                    "high": raw["High"].to_numpy(dtype=float),
                    "close": raw["Close"].to_numpy(dtype=float),
                }
            )
            df.to_csv(cache_file, index=False)
        if "symbol" not in df.columns:
            df["symbol"] = ticker
        frames.append(df[["date", "symbol", "low", "high", "close"]])

    if not frames:
        raise RuntimeError("No market data could be fetched or loaded from cache.")

    market = pd.concat(frames, ignore_index=True)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    logger.info(
        "Loaded %d market bars across %d symbols",
        len(market),
        market["symbol"].nunique(),
    )
    return market


# ---------------------------------------------------------------------------
# Internal ledger generation
# ---------------------------------------------------------------------------


def generate_ledger(cfg: dict, market: pd.DataFrame, rng: np.random.Generator, logger: logging.Logger) -> pd.DataFrame:
    """Generate a synthetic internal ledger priced off the real market data."""
    led_cfg = cfg["ledger"]
    n = int(led_cfg["num_trades"])

    # Sample (date, symbol) pairs by drawing row indices from the market frame.
    idx = rng.integers(low=0, high=len(market), size=n)
    sampled = market.iloc[idx].reset_index(drop=True)

    # Fill price: uniform between the day's low and high, plus a small bps jitter
    # so prices don't look suspiciously clean.
    low = sampled["low"].to_numpy(dtype=float)
    high = sampled["high"].to_numpy(dtype=float)
    base_price = rng.uniform(low=low, high=high)
    jitter_bps = led_cfg["price_jitter_bps"]
    jitter = rng.uniform(-jitter_bps, jitter_bps, size=n) / 10_000.0
    price = np.round(base_price * (1.0 + jitter), 4)

    # Quantity: 100-share lots.
    qty_min = led_cfg["quantity_min"] // 100
    qty_max = led_cfg["quantity_max"] // 100
    quantity = rng.integers(qty_min, qty_max + 1, size=n) * 100

    # Side: B/S roughly even.
    side = rng.choice(["BUY", "SELL"], size=n)

    # Timestamps: place each trade at a random second during US market hours
    # (09:30--16:00 ET, treated as naive local time for simplicity).
    seconds_in_session = (16 - 9) * 3600 + (0 - 30) * 60  # 6.5h
    offsets = rng.integers(0, seconds_in_session, size=n)
    base_ts = pd.to_datetime(sampled["date"]) + pd.Timedelta(hours=9, minutes=30)
    timestamps = base_ts + pd.to_timedelta(offsets, unit="s")

    ledger = pd.DataFrame(
        {
            "trade_id": [f"T{idx:07d}" for idx in range(1, n + 1)],
            "timestamp": timestamps.values,
            "symbol": sampled["symbol"].values,
            "side": side,
            "quantity": quantity.astype(int),
            "price": price,
        }
    ).sort_values("timestamp").reset_index(drop=True)

    logger.info("Generated internal ledger with %d trades", len(ledger))
    return ledger


# ---------------------------------------------------------------------------
# Broker feed corruption
# ---------------------------------------------------------------------------


@dataclass
class DefectRecord:
    """One row of the ground-truth table."""

    trade_id: str
    defect_type: str
    notes: str = ""


def _sample_ids(ids: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a fraction of ``ids`` without replacement."""
    n = int(round(len(ids) * rate))
    if n <= 0:
        return np.array([], dtype=ids.dtype)
    return rng.choice(ids, size=n, replace=False)


def _typo_symbol(symbol: str, rng: np.random.Generator) -> str:
    """Replace one character with a different uppercase letter."""
    if not symbol:
        return symbol
    pos = int(rng.integers(0, len(symbol)))
    alphabet = string.ascii_uppercase.replace(symbol[pos], "")
    new_char = alphabet[int(rng.integers(0, len(alphabet)))]
    return symbol[:pos] + new_char + symbol[pos + 1 :]


def generate_broker_feed(
    cfg: dict,
    ledger: pd.DataFrame,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the broker feed by copying the ledger and injecting defects.

    Returns ``(broker_feed_df, ground_truth_df)``. Defect types in ground truth:

    * ``missing_from_broker`` -- ledger row absent from broker feed
    * ``missing_from_ledger`` -- broker row absent from ledger
    * ``duplicate``           -- broker reports the same fill twice
    * ``price_mismatch_within_tol`` / ``price_mismatch_breach``
    * ``quantity_mismatch``
    * ``timestamp_shift``
    * ``symbol_typo``
    """
    corr = cfg["corruption"]
    broker = ledger.copy()
    defects: list[DefectRecord] = []

    all_ids = ledger["trade_id"].to_numpy()

    # 1) Drop rows entirely (missing_from_broker). Do this first so subsequent
    #    corruption rates target the surviving population.
    drop_ids = set(_sample_ids(all_ids, corr["missing_from_broker_rate"], rng))
    if drop_ids:
        for tid in drop_ids:
            defects.append(DefectRecord(tid, "missing_from_broker"))
        broker = broker[~broker["trade_id"].isin(drop_ids)].reset_index(drop=True)
        logger.info("Dropped %d ledger rows from broker feed", len(drop_ids))

    survivor_ids = broker["trade_id"].to_numpy()

    # 2) Price within tolerance.
    within_ids = _sample_ids(survivor_ids, corr["price_within_tol_rate"], rng)
    if len(within_ids):
        mask = broker["trade_id"].isin(within_ids)
        bps = corr["price_within_tol_bps"]
        shift = rng.uniform(-bps, bps, size=mask.sum()) / 10_000.0
        # Force the shift to be non-trivial so the diff is actually detectable.
        shift = np.where(np.abs(shift) < 0.5 / 10_000.0, 0.5 / 10_000.0, shift)
        broker.loc[mask, "price"] = np.round(
            broker.loc[mask, "price"].to_numpy() * (1.0 + shift), 4
        )
        for tid in within_ids:
            defects.append(DefectRecord(tid, "price_mismatch_within_tol"))

    # 3) Price breach (beyond tolerance).
    breach_ids = _sample_ids(survivor_ids, corr["price_breach_rate"], rng)
    if len(breach_ids):
        mask = broker["trade_id"].isin(breach_ids)
        bps = corr["price_breach_bps"]
        # Random sign, magnitude at least the breach threshold.
        signs = rng.choice([-1.0, 1.0], size=mask.sum())
        shift = signs * (bps / 10_000.0)
        broker.loc[mask, "price"] = np.round(
            broker.loc[mask, "price"].to_numpy() * (1.0 + shift), 4
        )
        for tid in breach_ids:
            defects.append(DefectRecord(tid, "price_mismatch_breach"))

    # 4) Quantity mismatch.
    qty_ids = _sample_ids(survivor_ids, corr["quantity_mismatch_rate"], rng)
    if len(qty_ids):
        mask = broker["trade_id"].isin(qty_ids)
        pct = corr["quantity_mismatch_pct"]
        signs = rng.choice([-1.0, 1.0], size=mask.sum())
        # Round to the nearest 100 shares so it still looks like a real fill,
        # but guarantee the value actually moves.
        original = broker.loc[mask, "quantity"].to_numpy()
        delta = np.maximum(np.round(original * pct / 100.0).astype(int) * 100, 100)
        new_qty = (original + (signs.astype(int) * delta)).astype(int)
        new_qty = np.where(new_qty <= 0, original + 100, new_qty)
        broker.loc[mask, "quantity"] = new_qty
        for tid in qty_ids:
            defects.append(DefectRecord(tid, "quantity_mismatch"))

    # 5) Timestamp shift (simulates a TZ-handling bug on the broker side).
    ts_ids = _sample_ids(survivor_ids, corr["timestamp_shift_rate"], rng)
    if len(ts_ids):
        mask = broker["trade_id"].isin(ts_ids)
        hours = corr["timestamp_shift_hours"]
        signs = rng.choice([-1, 1], size=mask.sum())
        shift = pd.to_timedelta(signs * hours, unit="h")
        broker.loc[mask, "timestamp"] = (
            pd.to_datetime(broker.loc[mask, "timestamp"]).to_numpy() + shift.to_numpy()
        )
        for tid in ts_ids:
            defects.append(DefectRecord(tid, "timestamp_shift"))

    # 6) Symbol typo.
    sym_ids = _sample_ids(survivor_ids, corr["symbol_typo_rate"], rng)
    if len(sym_ids):
        for tid in sym_ids:
            row_idx = broker.index[broker["trade_id"] == tid][0]
            original = broker.at[row_idx, "symbol"]
            broker.at[row_idx, "symbol"] = _typo_symbol(original, rng)
            defects.append(DefectRecord(tid, "symbol_typo", notes=f"orig={original}"))

    # 7) Duplicate rows. Duplicates retain the same trade_id intentionally --
    #    the broker reported the same fill twice. We append after all other
    #    mutations so the duplicate reflects the (already corrupted) state.
    dup_ids = _sample_ids(survivor_ids, corr["duplicate_rate"], rng)
    if len(dup_ids):
        dup_rows = broker[broker["trade_id"].isin(dup_ids)].copy()
        broker = pd.concat([broker, dup_rows], ignore_index=True)
        for tid in dup_ids:
            defects.append(DefectRecord(tid, "duplicate"))

    # 8) Extra broker fills with no ledger counterpart (missing_from_ledger).
    extra_n = int(round(len(ledger) * corr["missing_from_ledger_rate"]))
    if extra_n > 0:
        seed_idx = rng.integers(0, len(ledger), size=extra_n)
        extras = ledger.iloc[seed_idx].copy().reset_index(drop=True)
        # Mint new trade_ids that can't collide with the ledger's namespace.
        extras["trade_id"] = [f"B{i:07d}" for i in range(1, extra_n + 1)]
        broker = pd.concat([broker, extras], ignore_index=True)
        for tid in extras["trade_id"]:
            defects.append(DefectRecord(tid, "missing_from_ledger"))

    # Final shuffle so the broker file order isn't a trivial superset of the
    # ledger order. Use a stable permutation drawn from the same RNG.
    perm = rng.permutation(len(broker))
    broker = broker.iloc[perm].reset_index(drop=True)

    ground_truth = pd.DataFrame(
        [{"trade_id": d.trade_id, "defect_type": d.defect_type, "notes": d.notes} for d in defects]
    ).sort_values(["defect_type", "trade_id"]).reset_index(drop=True)

    logger.info(
        "Broker feed built: %d rows, %d injected defects across %d unique trade_ids",
        len(broker),
        len(ground_truth),
        ground_truth["trade_id"].nunique() if not ground_truth.empty else 0,
    )
    return broker, ground_truth


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    """Run the full ingestion pipeline and return paths to the written files."""
    cfg = load_config(config_path)
    logger = configure_logging(cfg)

    seed = int(cfg["random_seed"])
    rng = np.random.default_rng(seed)
    random.seed(seed)

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    market = fetch_market_data(cfg, logger)
    ledger = generate_ledger(cfg, market, rng, logger)
    broker, ground_truth = generate_broker_feed(cfg, ledger, rng, logger)

    ledger_path = PROJECT_ROOT / cfg["paths"]["ledger_csv"]
    broker_path = PROJECT_ROOT / cfg["paths"]["broker_csv"]
    truth_path = PROJECT_ROOT / cfg["paths"]["ground_truth_csv"]

    ledger.to_csv(ledger_path, index=False)
    broker.to_csv(broker_path, index=False)
    ground_truth.to_csv(truth_path, index=False)

    logger.info("Wrote %s (%d rows)", ledger_path.relative_to(PROJECT_ROOT), len(ledger))
    logger.info("Wrote %s (%d rows)", broker_path.relative_to(PROJECT_ROOT), len(broker))
    logger.info(
        "Wrote %s (%d defect rows)", truth_path.relative_to(PROJECT_ROOT), len(ground_truth)
    )

    return {"ledger": ledger_path, "broker": broker_path, "ground_truth": truth_path}


def main(argv: Iterable[str] | None = None) -> None:
    run()


if __name__ == "__main__":
    main()
