"""Load the raw KKBox CSVs into DuckDB and build the analysis tables.

Tables created in work/kkbox.duckdb:
  txn         cleaned transactions (payments and cancellations)
  pay         paid transactions, with coverage cut short by any later cancellation
  spells      subscription spells: paid coverage where gaps of 30 days or less are merged
              (the KKBox churn rule: no new subscription within 30 days of expiry = churn)
  sub_months  one row per spell per calendar month the subscriber was covered
  members     demographics
  logs_month  listening per user per month, aggregated from the daily user logs.
              Keyed by uid = hash(msno), a compact number, so the big aggregation fits in a
              laptop's memory. Join it with: logs_month.uid = hash(<table>.msno)
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from common import DATA_END, DATA_START, GRACE_DAYS, connect, save_metrics

TXN_COLS = {
    "msno": "VARCHAR", "payment_method_id": "INTEGER", "payment_plan_days": "INTEGER",
    "plan_list_price": "DOUBLE", "actual_amount_paid": "DOUBLE", "is_auto_renew": "INTEGER",
    "transaction_date": "VARCHAR", "membership_expire_date": "VARCHAR", "is_cancel": "INTEGER",
}
LOG_COLS = {
    "msno": "VARCHAR", "date": "VARCHAR", "num_25": "DOUBLE", "num_50": "DOUBLE", "num_75": "DOUBLE",
    "num_985": "DOUBLE", "num_100": "DOUBLE", "num_unq": "DOUBLE", "total_secs": "DOUBLE",
}
MEMBER_COLS = {
    "msno": "VARCHAR", "city": "INTEGER", "bd": "INTEGER", "gender": "VARCHAR",
    "registered_via": "INTEGER", "registration_init_time": "VARCHAR",
}


def find(data_dir, names):
    """Find raw files anywhere under data_dir (7z archives often extract into subfolders)."""
    hits = []
    for n in names:
        hits += [p for p in Path(data_dir).rglob(n) if p.is_file()]
    return sorted({str(p) for p in hits})


def seven_zip():
    for exe in ("7z", "7zz", "7za"):
        if shutil.which(exe):
            return exe
    return None


def open_7z_pipe(archive):
    """Decompress one .7z archive into a named pipe so DuckDB can read it without the CSV
    ever touching the disk (user_logs.csv is ~30 GB unzipped). Returns (pipe path, process)."""
    exe = seven_zip()
    if not exe:
        raise SystemExit("Found .7z files but no 7-Zip. Install it: conda install -c conda-forge p7zip")
    pipe = Path(tempfile.mkdtemp(prefix="sgl_pipe_")) / "stream.csv"
    os.mkfifo(pipe)
    proc = subprocess.Popen(f'"{exe}" x -so "{archive}" > "{pipe}"', shell=True, stderr=subprocess.DEVNULL)
    return str(pipe), proc


def csv_source(files, cols):
    file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    col_spec = "{" + ", ".join(f"'{k}': '{v}'" for k, v in cols.items()) + "}"
    return f"read_csv({file_list}, header=true, columns={col_spec}, ignore_errors=true)"


def build(data_dir, sample_pct=100, memory_limit=None, threads=None):
    t0 = time.time()
    con = connect(memory_limit, threads)
    sample = f"hash(msno) % 100 < {int(sample_pct)}"
    end, start = f"DATE '{DATA_END}'", f"DATE '{DATA_START}'"

    txn_files = find(data_dir, ["transactions.csv", "transactions_v2.csv"])
    log_files = find(data_dir, ["user_logs.csv", "user_logs_v2.csv"])
    # If a logs CSV isn't extracted but its .7z is there, stream it from the archive
    have = {Path(f).name for f in log_files}
    log_7z = [a for a in find(data_dir, ["user_logs.csv.7z", "user_logs_v2.csv.7z"])
              if Path(a).name[:-3] not in have]
    mem_files = find(data_dir, ["members_v3.csv"])  # members.csv (v1) has an extra column; use v3
    if not txn_files:
        raise SystemExit(f"No transactions.csv or transactions_v2.csv found under {data_dir}")
    print("transactions:", txn_files, "\nuser logs:", log_files or "none",
          "\nuser logs streamed from .7z:", log_7z or "none", "\nmembers:", mem_files or "none")

    # ---- transactions ----
    print("Loading transactions ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE txn_raw AS
        SELECT * FROM {csv_source(txn_files, TXN_COLS)} WHERE {sample}""")
    con.execute(f"""
        CREATE OR REPLACE TABLE txn AS
        SELECT DISTINCT msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid,
               is_auto_renew, is_cancel,
               try_strptime(transaction_date, '%Y%m%d')::DATE AS txn_date,
               try_strptime(membership_expire_date, '%Y%m%d')::DATE AS expire_date
        FROM txn_raw""")
    con.execute(f"""
        DELETE FROM txn WHERE txn_date IS NULL OR expire_date IS NULL
           OR txn_date < {start} OR txn_date > {end}
           OR expire_date < {start} OR expire_date > DATE '2020-12-31'""")
    n_raw = con.execute("SELECT count(*) FROM txn_raw").fetchone()[0]
    n_txn = con.execute("SELECT count(*) FROM txn").fetchone()[0]
    con.execute("DROP TABLE txn_raw")

    # ---- paid coverage, cut short by cancellations ----
    # A cancellation row (is_cancel = 1) ends auto-renew. Its expire date is when coverage
    # stops, so it cuts short the latest payment made on or before the cancellation.
    print("Building subscription spells ...")
    con.execute("""
        CREATE OR REPLACE TABLE cancels AS
        SELECT msno, txn_date AS cancel_date, expire_date AS cancel_expire FROM txn WHERE is_cancel = 1""")
    con.execute("""
        CREATE OR REPLACE TABLE pay AS
        SELECT p.*,
               CASE WHEN c.cancel_date IS NOT NULL AND c.cancel_date <= p.expire_date
                    THEN greatest(p.txn_date, least(p.expire_date, c.cancel_expire))
                    ELSE p.expire_date END AS end_date,
               (c.cancel_date IS NOT NULL AND c.cancel_date <= p.expire_date)::INTEGER AS was_cancelled
        FROM (SELECT * FROM txn WHERE is_cancel = 0 AND expire_date > txn_date) p
        ASOF LEFT JOIN cancels c ON p.msno = c.msno AND c.cancel_date >= p.txn_date""")

    # ---- spells: merge coverage when the gap is 30 days or less ----
    con.execute(f"""
        CREATE OR REPLACE TABLE pay AS
        WITH o AS (
            SELECT *, max(end_date) OVER (PARTITION BY msno ORDER BY txn_date, end_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_end
            FROM pay)
        SELECT *, sum(CASE WHEN prev_max_end IS NULL OR txn_date > prev_max_end + {GRACE_DAYS} THEN 1 ELSE 0 END)
                  OVER (PARTITION BY msno ORDER BY txn_date, end_date ROWS UNBOUNDED PRECEDING) AS spell_no
        FROM o""")
    # covered_through = how far paid coverage reaches as of each payment (running max within the spell)
    con.execute("""
        CREATE OR REPLACE TABLE pay AS
        SELECT * EXCLUDE (prev_max_end),
               max(end_date) OVER (PARTITION BY msno, spell_no ORDER BY txn_date, end_date
                                   ROWS UNBOUNDED PRECEDING) AS covered_through,
               -- same, but ignoring cancellations: what was scheduled at the time of payment
               max(expire_date) OVER (PARTITION BY msno, spell_no ORDER BY txn_date, end_date
                                      ROWS UNBOUNDED PRECEDING) AS expire_through
        FROM pay""")
    con.execute(f"""
        CREATE OR REPLACE TABLE spells AS
        SELECT msno, spell_no,
               min(txn_date) AS spell_start, max(end_date) AS spell_end,
               arg_min(is_auto_renew, txn_date) AS start_auto_renew,
               arg_min(payment_plan_days, txn_date) AS start_plan_days,
               arg_min((actual_amount_paid < plan_list_price)::INTEGER, txn_date) AS start_discount,
               count(*) AS n_payments,
               sum(actual_amount_paid) AS revenue
        FROM pay GROUP BY 1, 2""")
    con.execute(f"""
        CREATE OR REPLACE TABLE spells AS
        SELECT *, (spell_end + {GRACE_DAYS} <= {end})::INTEGER AS churned,
               datediff('day', spell_start, spell_end) AS duration_days
        FROM spells""")

    # ---- one row per spell per covered month ----
    con.execute(f"""
        CREATE OR REPLACE TABLE sub_months AS
        SELECT msno, spell_no, month, spell_start, spell_end, churned
        FROM (SELECT *, unnest(generate_series(date_trunc('month', spell_start), date_trunc('month', spell_end),
                                               INTERVAL 1 MONTH))::DATE AS month FROM spells)
        WHERE month <= date_trunc('month', {end})""")

    # ---- members ----
    if mem_files:
        con.execute(f"""
            CREATE OR REPLACE TABLE members AS
            SELECT DISTINCT ON (msno) msno, city,
                   CASE WHEN bd BETWEEN 10 AND 80 THEN bd END AS age,
                   coalesce(nullif(gender, ''), 'unknown') AS gender, registered_via,
                   try_strptime(registration_init_time, '%Y%m%d')::DATE AS reg_date
            FROM {csv_source(mem_files, MEMBER_COLS)} WHERE {sample}""")
    else:
        con.execute("""CREATE OR REPLACE TABLE members (msno VARCHAR, city INTEGER, age INTEGER,
                       gender VARCHAR, registered_via INTEGER, reg_date DATE)""")

    # ---- listening logs, per user per month ----
    # Two steps so it fits in a laptop's memory:
    #  1. Stream each logs source once (straight from .7z if needed) into a compact Parquet file:
    #     only sampled users, numeric user id, parsed date. No aggregation, so memory stays low.
    #  2. Aggregate to user-month in N chunks of users (uid % N), each small enough for memory.
    from common import WORK
    stage = WORK / "logs_stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    sources = [(f, None) for f in log_files] + [(a, "7z") for a in log_7z]
    for i, (path, kind) in enumerate(sources):
        proc, label = None, Path(path).name
        if kind == "7z":
            path, proc = open_7z_pipe(path)
            label += " (streamed, not unzipped)"
        print(f"Reading user logs from {label}. The full user_logs is ~30 GB, so this can take 20-60 minutes ...",
              flush=True)
        con.execute(f"""
            COPY (
                SELECT hash(msno) AS uid, d, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs
                FROM (SELECT *, try_strptime(date, '%Y%m%d')::DATE AS d
                      FROM {csv_source([path], LOG_COLS)} WHERE {sample})
                WHERE d BETWEEN {start} AND {end}
            ) TO '{stage / f"part_{i}.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
        if proc:
            proc.wait()
    con.execute("""CREATE OR REPLACE TABLE logs_month (uid UBIGINT, month DATE, active_days BIGINT,
                   hours DOUBLE, plays DOUBLE, full_plays DOUBLE, unique_songs DOUBLE)""")
    parts = int(os.environ.get("SGL_LOG_CHUNKS", 8))
    if sources:
        for k in range(parts):
            print(f"  summarizing listening by user and month: chunk {k + 1}/{parts}", flush=True)
            con.execute(f"""
                INSERT INTO logs_month
                SELECT uid, date_trunc('month', d)::DATE AS month,
                       -- one log row = one user-day, so count(*) gives listening days
                       least(count(*), 31) AS active_days,
                       sum(least(greatest(total_secs, 0), 86400)) / 3600.0 AS hours,
                       sum(num_25 + num_50 + num_75 + num_985 + num_100) AS plays,
                       sum(num_985 + num_100) AS full_plays,
                       sum(num_unq) AS unique_songs
                FROM read_parquet('{stage}/*.parquet')
                WHERE uid % {parts} = {k}
                GROUP BY 1, 2""")
    shutil.rmtree(stage, ignore_errors=True)

    q = lambda s: con.execute(s).fetchone()[0]
    stats = {
        "sample_pct": sample_pct,
        "transaction_rows_raw": n_raw,
        "transaction_rows_clean": n_txn,
        "paid_transactions": q("SELECT count(*) FROM pay"),
        "cancellations": q("SELECT count(*) FROM cancels"),
        "users": q("SELECT count(DISTINCT msno) FROM spells"),
        "spells": q("SELECT count(*) FROM spells"),
        "subscriber_months": q("SELECT count(*) FROM sub_months"),
        "log_user_months": q("SELECT count(*) FROM logs_month"),
        "log_months_covered": q("SELECT count(DISTINCT month) FROM logs_month"),
        "share_discounted_payments": q("SELECT avg((actual_amount_paid < plan_list_price)::INTEGER) FROM pay WHERE plan_list_price > 0"),
        "share_auto_renew_payments": q("SELECT avg(is_auto_renew) FROM pay"),
        "build_seconds": round(time.time() - t0, 1),
    }
    save_metrics("data", stats)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--sample-pct", type=int, default=100, help="keep this %% of users (hash-based)")
    ap.add_argument("--memory-limit", default=None, help="e.g. 8GB")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    build(a.data_dir, a.sample_pct, a.memory_limit, a.threads)
