"""Streamlit dashboard for the reconciliation pipeline.

Reads the artefacts produced by ``run_pipeline.py``:

* ``data/processed/exceptions.csv``    -- one row per detected defect
* ``data/processed/summary.json``      -- headline metrics + per-class accuracy
* ``data/processed/quality_checks.csv`` -- structural-DQ results

Launch from the project root::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ingest import load_config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_artifacts(
    exceptions_path: str,
    summary_path: str,
    quality_path: str,
    cache_buster: float,  # noqa: ARG001  -- used purely to invalidate cache
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    exceptions = pd.read_csv(exceptions_path, parse_dates=["ledger_timestamp", "broker_timestamp"])
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    quality = pd.read_csv(quality_path)
    return exceptions, summary, quality


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Trade Reconciliation",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_config(PROJECT_DIR / "config.yaml")
exceptions_path = PROJECT_DIR / cfg["reconciliation"]["exceptions_csv"]
summary_path = PROJECT_DIR / cfg["report"]["summary_json"]
quality_path = PROJECT_DIR / cfg["quality_checks"]["results_csv"]

# Friendly halt if the pipeline hasn't been run yet rather than a stack trace.
missing = [p for p in (exceptions_path, summary_path, quality_path) if not p.exists()]
if missing:
    st.title("Trade Reconciliation -- Pipeline outputs not found")
    st.error(
        "The pipeline hasn't produced its outputs yet. Run "
        "`python run_pipeline.py` from the project root, then refresh."
    )
    st.write("Missing files:")
    for p in missing:
        st.write(f"- `{p.relative_to(PROJECT_DIR)}`")
    st.stop()

cache_key = max(_file_mtime(p) for p in (exceptions_path, summary_path, quality_path))
exceptions, summary, quality = _load_artifacts(
    str(exceptions_path), str(summary_path), str(quality_path), cache_key
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Trade Reconciliation & Data Quality Monitoring")
st.caption(
    "Daily reconciliation between the firm's internal ledger and the broker's "
    "confirmation feed. Every defect classified, scored, and triageable."
)

last_run = datetime.fromtimestamp(cache_key).strftime("%Y-%m-%d %H:%M:%S")
top_bar_left, top_bar_right = st.columns([4, 1])
top_bar_left.caption(f"Last pipeline run: **{last_run}**")
if top_bar_right.button("Refresh data", width="stretch"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

acc = summary["detection_accuracy"]
trades = summary["trades"]
excs = summary["exceptions"]

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Reconciliation rate",
    f"{summary['reconciliation_rate'] * 100:.2f}%",
    help="Share of ledger trades with no exceptions raised.",
)
m2.metric(
    "Exceptions raised",
    f"{excs['total_rows']:,}",
    delta=f"{excs['unique_trade_ids']:,} unique trade_ids",
    delta_color="off",
    help="One trade can collect multiple exceptions (e.g. price + timestamp).",
)
m3.metric(
    "Detection precision",
    f"{acc['precision']:.3f}",
    help="Of flagged defects, share that match an injected (ground-truth) defect.",
)
m4.metric(
    "Detection recall",
    f"{acc['recall']:.3f}",
    help="Of injected defects, share the engine successfully caught.",
)

st.divider()

# ---------------------------------------------------------------------------
# Charts: exceptions by type + severity
# ---------------------------------------------------------------------------

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Exceptions by type")
    by_type = exceptions["classification"].value_counts().sort_values(ascending=False)
    st.bar_chart(by_type, height=320)

with chart_right:
    st.subheader("Exceptions by severity")
    severity_labels = {0: "0 - clean", 1: "1 - info", 2: "2 - warn", 3: "3 - critical"}
    by_sev = (
        exceptions["severity"]
        .value_counts()
        .sort_index()
        .rename(index=severity_labels)
    )
    st.bar_chart(by_sev, height=320)

# ---------------------------------------------------------------------------
# Quality checks panel
# ---------------------------------------------------------------------------

st.subheader("Data quality checks")
qc_failed = quality[~quality["passed"]]
if qc_failed.empty:
    st.success(f"All {len(quality)} structural checks passed.")
else:
    st.warning(
        f"{len(qc_failed)} of {len(quality)} checks failed -- review below."
    )
    st.dataframe(
        qc_failed[["table", "check", "severity", "rows_failed", "detail"]],
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Per-class accuracy detail
# ---------------------------------------------------------------------------

with st.expander("Per-class detection accuracy"):
    per_class = acc.get("per_class", {})
    if not per_class:
        st.write("No per-class metrics in the current summary.")
    else:
        per_class_df = (
            pd.DataFrame(per_class)
            .T.rename_axis("classification")
            .reset_index()
            [["classification", "tp", "fp", "fn", "precision", "recall"]]
        )
        st.dataframe(
            per_class_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Drill-down filters")
all_classes = sorted(exceptions["classification"].dropna().unique())
all_severities = sorted(exceptions["severity"].dropna().unique().astype(int))
all_symbols = sorted(exceptions["symbol"].dropna().unique())

sel_classes = st.sidebar.multiselect("Classification", all_classes, default=all_classes)
sel_severities = st.sidebar.multiselect("Severity", all_severities, default=all_severities)
sel_symbols = st.sidebar.multiselect(
    "Symbol (leave blank for all)", all_symbols, default=[]
)
search_id = st.sidebar.text_input("Search trade_id contains")

# ---------------------------------------------------------------------------
# Drill-down table
# ---------------------------------------------------------------------------

st.subheader("Flagged trades")
mask = (
    exceptions["classification"].isin(sel_classes)
    & exceptions["severity"].isin(sel_severities)
)
if sel_symbols:
    mask &= exceptions["symbol"].isin(sel_symbols)
if search_id:
    mask &= exceptions["trade_id"].astype(str).str.contains(search_id, case=False, na=False)

filtered = exceptions.loc[mask].sort_values(
    ["severity", "classification", "trade_id"], ascending=[False, True, True]
)
st.caption(
    f"Showing **{len(filtered):,}** of {len(exceptions):,} exception rows "
    f"across {filtered['trade_id'].nunique():,} unique trade_ids."
)
st.dataframe(filtered, hide_index=True, width="stretch", height=520)

st.download_button(
    "Download filtered rows as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="exceptions_filtered.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Footer -- raw input counts for context
# ---------------------------------------------------------------------------

st.divider()
foot_l, foot_m, foot_r = st.columns(3)
foot_l.caption(f"Ledger rows: **{trades['ledger_rows']:,}**")
foot_m.caption(f"Broker rows: **{trades['broker_rows']:,}**")
foot_r.caption(f"Broker-only trade_ids: **{trades['broker_only_trade_ids']:,}**")
