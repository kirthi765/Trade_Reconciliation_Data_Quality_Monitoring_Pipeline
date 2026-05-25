# Trade Reconciliation & Data Quality Monitoring Pipeline — Build Plan

> This is a build plan for Claude Code. Work through it **phase by phase**. Do not skip ahead.
> After each phase, stop and confirm the acceptance criteria are met before moving on.
> Commit to git at the end of every phase with a meaningful message.

---

## Context for Claude Code

I'm building a portfolio project for a **Data Analyst / Operations Automation** role at a financial market-making firm. The job emphasizes: automating data review and reconciliation, building workflows across multiple databases and APIs, strong Python/Pandas + SQL, debugging/triaging data exceptions, and producing **maintainable, well-documented tooling** (not one-off notebooks).

**The scenario this project simulates:**
A trading firm executes trades through a broker. Two records of every trade exist — the firm's **internal ledger** (what the firm thinks happened) and the **broker confirmation feed** (what the broker reports). These should match exactly but don't. This tool ingests both feeds, automatically reconciles them, classifies every discrepancy, runs data-quality checks, and surfaces results in a dashboard. It replaces manual Excel review done by ops analysts.

**Hard requirements (these are how the project gets judged):**
- Real Python package structure under `src/`, not a single notebook.
- Reconciliation logic split deliberately between **SQL** (joins) and **Pandas** (tolerance/fuzzy comparison) to showcase both.
- Config-driven (no hardcoded paths/thresholds), with logging and unit tests.
- Because corruption is injected synthetically, the engine's accuracy is **measurable against ground truth** — this is a key selling point. Preserve a ground-truth label on every injected defect.

---

## Tech stack & conventions

- **Python 3.11+**, managed with a virtual environment.
- Core libs: `pandas`, `yfinance`, `pyyaml`, `streamlit`, `pytest`. DB via stdlib `sqlite3` (no ORM needed).
- Style: type hints on all functions, docstrings, `logging` module (never bare `print` in `src/`).
- All tunable values (tickers, corruption rates, price tolerance, file paths) live in `config.yaml`.
- Every defect the generator injects must be recorded in a ground-truth table so reconciliation accuracy can be scored.

---

## Target repo structure

```
trade-reconciliation/
├── README.md
├── requirements.txt
├── config.yaml
├── run_pipeline.py            # single entry point: ingest -> load -> reconcile -> checks -> report
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── ingest.py              # market data pull + ledger/broker feed generation
│   ├── database.py            # SQLite load/query helpers
│   ├── reconcile.py           # matching + classification engine
│   ├── quality_checks.py      # schema/null/range validation
│   └── report.py              # summary metrics + accuracy scoring
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── tests/
│   └── test_reconcile.py
└── logs/
    └── pipeline.log
```

---

## Phase 1 — Data foundation

**Goal:** Produce two trade feeds that should agree but don't, with a ground-truth record of every injected defect.

Tasks:
1. Scaffold the repo structure above. Set up venv + `requirements.txt`.
2. In `ingest.py`: pull ~6 months of daily price data for ~20 liquid tickers via `yfinance`. Cache to `data/raw/` so reruns don't refetch.
3. Generate a synthetic **internal ledger**: realistic trade records (trade_id, timestamp, symbol, side, quantity, price) priced off the real market data.
4. Generate a **broker feed** as a copy of the ledger with config-driven corruption injected: missing rows, duplicate fills, price errors (small within-tolerance + large beyond-tolerance), quantity mismatches, timezone-shifted timestamps, and symbol typos.
5. Write a **ground-truth table** logging every defect (trade_id, defect_type) for later accuracy scoring.

**Acceptance criteria:** Running `python -m src.ingest` produces `internal_ledger.csv`, `broker_feed.csv`, and `ground_truth.csv` in `data/raw/`. Corruption rates are read from `config.yaml`. Re-running is deterministic given a fixed random seed.

---

## Phase 2 — Database layer

**Goal:** Load both feeds into SQLite and prove the SQL half of reconciliation.

Tasks:
1. In `database.py`: helpers to create the DB, load a DataFrame to a table, and run a query returning a DataFrame.
2. Load both feeds into `trades.db`.
3. Write a SQL query using a **FULL OUTER JOIN** (emulate via LEFT/UNION in SQLite) on trade_id to find missing-from-broker and missing-from-ledger records.

**Acceptance criteria:** `database.py` loads both tables and the join query correctly returns all orphaned records on both sides. Verify counts against the ground-truth table.

---

## Phase 3 — Reconciliation engine

**Goal:** Classify every record and measure accuracy against ground truth.

Tasks:
1. In `reconcile.py`: combine the SQL join (missing/orphaned) with **Pandas** logic for matched records — compare price within a config tolerance, compare quantity, detect duplicates.
2. Classify each record: `matched`, `missing_from_broker`, `missing_from_ledger`, `duplicate`, `price_mismatch_within_tol`, `price_mismatch_breach`, `quantity_mismatch`.
3. Assign a **severity score** per exception type.
4. Score the engine against `ground_truth.csv`: precision/recall of defect detection.

**Acceptance criteria:** Produces a classified exceptions table in `data/processed/`. Prints a detection accuracy figure (target: catches >95% of injected defects). Classification logic is pure functions that are unit-testable.

---

## Phase 4 — Quality checks & automation polish

**Goal:** Make it a tool, not a script.

Tasks:
1. In `quality_checks.py`: schema validation, null checks, out-of-range price/quantity detection. Return structured results, don't just raise.
2. Build `run_pipeline.py` as a single entry point chaining ingest → load → reconcile → checks → report, driven by `config.yaml`, with `logging` to `logs/pipeline.log`.
3. In `tests/test_reconcile.py`: unit tests on the classification functions using small hand-built fixtures (one of each defect type).

**Acceptance criteria:** `python run_pipeline.py` runs end-to-end and logs each stage. `pytest` passes. No hardcoded paths or thresholds anywhere in `src/`.

---

## Phase 5 — Dashboard & documentation

**Goal:** Make results legible and document use cases.

Tasks:
1. In `dashboard/app.py`: Streamlit dashboard with reconciliation rate (headline metric), exceptions by type and severity (charts), a filterable drill-down table of flagged trades, and the detection-accuracy figure.
2. Write `README.md`: open with a plain-English **Problem** section, then architecture, how to run, and a **Use Cases** section documenting each workflow the tool supports.
3. Final pass: ensure logging, config, and tests are clean.

**Acceptance criteria:** `streamlit run dashboard/app.py` launches and displays all panels off the processed data. README reads like onboarding docs for a teammate.

---

## Git conventions

Commit at the end of each phase (or more granularly within phases). Use clear messages, e.g.:
- `feat(ingest): pull market data and generate corrupted broker feed`
- `feat(reconcile): add classification engine with severity scoring`
- `test: add unit tests for exception classification`

Incremental, meaningful commit history is part of the deliverable — it tells the "process owner" story.

---

## Out of scope (do not build)

- No live/real broker connections — all data is synthetic or public-market historical.
- No ORM, no cloud deployment, no auth. Keep it local and simple.
- No ML models — this is rules-based reconciliation by design.
