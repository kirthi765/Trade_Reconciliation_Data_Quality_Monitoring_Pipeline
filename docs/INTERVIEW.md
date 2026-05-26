# Interview talking points -- Trade Reconciliation Pipeline

A one-pager for prepping the "tell me about a project you built" question.

---

## 30-second pitch

"I built a trade-reconciliation pipeline that automates the kind of
manual Excel-based reconciliation an ops analyst at a market-making
firm would do every day. It ingests two trade feeds -- the firm's
internal ledger and the broker's confirmation feed -- finds every
mismatch, classifies the cause into one of eight categories, runs
structural data-quality checks, and surfaces results in a Streamlit
dashboard. Because the broker-side defects are injected synthetically
against a known ground truth, the engine's detection accuracy is
*measurable* -- not just a vibe. On the bundled dataset it hits 100%
precision and recall."

## The numbers to cite

| Metric                           | Value                              |
| -------------------------------- | ---------------------------------- |
| Internal ledger rows             | 5,000                              |
| Broker feed rows                 | 5,025 (50 dropped + 50 duped + 25 extras) |
| Defects injected across 8 types  | 399 (across 393 unique trade_ids)  |
| Reconciliation rate              | 92.64%                             |
| Detection precision / recall / F1| **1.000 / 1.000 / 1.000**          |
| Unit tests                       | 22, all green, run in CI           |
| Lines of code in `src/`          | ~1,100                             |
| Build phases / commits           | 5 phases, ~10 commits              |

## Design decisions worth defending

1. **SQL for the join, Pandas for the comparison.** SQLite handles
   orphan detection via a `FULL OUTER JOIN` (emulated with two
   `LEFT JOIN`s `UNION ALL`-ed for SQLite portability). Pandas handles
   tolerance-based comparison on the matched pairs, vectorised. Each
   tool used for what it's actually good at, deliberately.
2. **Pure-function classifiers separated from orchestration.**
   `classify_price_diff`, `classify_quantity_diff`,
   `classify_symbol_diff`, `classify_timestamp_diff`, `severity_score`
   are scalar-in / scalar-out with no I/O. That's the unit-test
   surface. Orchestration sits in `reconcile()` which composes them
   vectorised across the joined frame.
3. **Synthetic data with ground truth makes accuracy measurable.**
   Every injected defect is logged to `ground_truth.csv` keyed on
   `(trade_id, defect_type)`. The engine's classified output is
   compared as a set against ground truth to produce real
   precision/recall numbers, including per-class breakdown.
4. **Config-driven, not coded.** Every tunable -- tickers, corruption
   rates, price tolerance, timestamp threshold, severity weights, file
   paths -- lives in `config.yaml`. Grep-audited: zero hardcoded paths
   or thresholds in `src/`. Lets ops tune behaviour without code
   changes.
5. **Quality checks return data, never raise.** `CheckResult`
   dataclass per check with severity (`info` / `warn` / `error`). The
   pipeline never halts on bad input -- it tells you what's wrong and
   keeps going so you still get partial output to triage.
6. **One entry point, structured logging.** `run_pipeline.py` chains
   all five stages behind per-stage banners written to
   `logs/pipeline.log`. Shares one SQLite connection across stages so
   the DB is loaded once.

## Anticipated questions and answers

**"Why SQLite and not Postgres / DuckDB?"** Local-first analyst tool,
zero ops cost, comes with Python's stdlib. Picked SQLite specifically
to *show* the `FULL OUTER JOIN` emulation pattern -- that's an
interview-friendly piece of SQL craft that newer engines hide behind
native support.

**"Why no ORM?"** Two tables, three queries. SQLAlchemy would add a
layer of abstraction with no payoff at this scale. `pd.read_sql_query`
+ raw SQL is more readable.

**"How would you handle 100M trades?"** Two changes. (1) The Pandas
comparison stage would either chunk the inner join or move it back
into SQL with windowed `LAG` / `LEAD` for duplicate detection. (2)
SQLite would become Postgres or DuckDB so the orphan query can use a
real `FULL OUTER JOIN` and indexed hash joins. The classifier
functions stay pure either way.

**"What if the broker feed schema changes?"** That's exactly what the
`check_schema` step is for. Type mismatch and missing-column errors
surface in the QC panel rather than crashing the pipeline. The
schema dict lives at the top of `quality_checks.py` so updating it is
a one-line change.

**"What didn't you test?"** `quality_checks.py` and `report.py`. The
plan called for unit tests only on `reconcile.py` (the most logic-heavy
module). Real-world I would add coverage for both as the next
iteration -- they're written with the same pure-function discipline so
testing them is mechanical.

**"What was harder than you expected?"** Two things. (1) `yfinance`
under newer versions returns a `MultiIndex` on columns even for a
single ticker -- the first run crashed on a `KeyError: ['date']`.
Fixed by flattening defensively and reading column names off level 0.
(2) SQLite roundtripping `timestamp` columns as strings broke the
type-check stage with a `expected datetime, got string` warning. Both
caught by the quality-check stage, which is the point of the
quality-check stage existing.

**"How does this relate to a market-making firm's actual workflow?"**
Real ops teams spend hours daily reconciling clearing-broker reports
against internal positions. The problems are identical to the ones
injected here -- TZ bugs, fat-finger symbol typos, dropped fills,
late-arriving duplicates. The tool collapses that workflow into:
review dashboard, sort by severity 3, work top to bottom.

## What to show on a screen-share

In this order:

1. The dashboard (most visual). Open `streamlit run dashboard/app.py`,
   point at the headline metrics, then drag a severity filter.
2. `config.yaml`. "Every tunable lives here. Watch what happens if I
   tighten `price_tolerance_bps`." (Rerun the pipeline, refresh the
   dashboard, point at the changed per-class recall.)
3. `tests/test_reconcile.py` -- the mini end-to-end fixture with one
   trade per defect type. "This is how I unit-test classification
   logic that operates on data."
4. The orphan-finder SQL in `src/database.py`. "Two `LEFT JOIN`s
   `UNION ALL`-ed because SQLite predates native `FULL OUTER JOIN`."
5. Git log. "Five commits, one per phase, plus polish. Incremental
   process is part of the deliverable."

## Things to NOT say

- Don't oversell the ML angle -- there isn't one, on purpose. Say so.
- Don't claim "production-ready" -- it's a portfolio project with
  synthetic data. Say "production-shaped" or "structured like a real
  ops tool."
- Don't say "I used AI to build it" without specifics -- describe
  *what* you decided and *why*. The interviewer cares about your
  judgment, not the keyboard mechanics.
