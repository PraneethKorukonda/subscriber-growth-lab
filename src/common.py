"""Shared paths, settings and helpers."""
import json
import os
from datetime import date
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("SGL_WORK", ROOT / "work"))
RESULTS = Path(os.environ.get("SGL_RESULTS", ROOT / "results"))
FIGS = RESULTS / "figures"

# The KKBox data ends on March 31, 2017. A churn is only observed once 30 days have
# passed after a membership expires, so anything closer to the end is censored.
DATA_START = date(2015, 1, 1)
DATA_END = date(2017, 3, 31)
GRACE_DAYS = 30
# Transactions start in Jan 2015, so early "new" subscribers are really people who
# were already subscribed. Growth metrics are reported from this month on.
REPORT_START = date(2016, 1, 1)


def connect(memory_limit=None, threads=None):
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WORK / "kkbox.duckdb"))
    con.execute(f"SET temp_directory='{WORK / 'duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    # Never let DuckDB fill the disk: stop with a clear error before temp files pass this size
    con.execute(f"SET max_temp_directory_size='{os.environ.get('SGL_MAX_TEMP', '4GB')}'")
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    return con


def save_metrics(section, values):
    """Merge one section of results into results/metrics.json."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "metrics.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = values
    path.write_text(json.dumps(data, indent=2, default=str))


def load_metrics():
    path = RESULTS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def figure_path(name):
    FIGS.mkdir(parents=True, exist_ok=True)
    return FIGS / name
