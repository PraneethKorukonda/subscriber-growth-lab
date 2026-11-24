"""Run the whole project: build tables, run every analysis, write results/RESULTS.md.

  python run_all.py --data-dir data/raw                 # real KKBox data
  python run_all.py --data-dir data/raw --sample-pct 20 # 20% of users, much faster
  python run_all.py --synthetic                         # fake data with known answers
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(args):
    print("\n$ python " + " ".join(args), flush=True)
    t = time.time()
    subprocess.run([sys.executable] + args, cwd=ROOT, check=True)
    print(f"  done in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--sample-pct", type=int, default=100)
    ap.add_argument("--memory-limit", default=None, help="DuckDB memory cap, e.g. 6GB")
    ap.add_argument("--treatment", default="discount", choices=["discount", "auto_renew"])
    ap.add_argument("--synthetic", action="store_true", help="generate and use synthetic data")
    ap.add_argument("--skip-build", action="store_true", help="reuse tables from a previous run")
    a = ap.parse_args()

    data_dir = a.data_dir
    if a.synthetic:
        data_dir = "data/synthetic"
        if not (ROOT / data_dir / "transactions.csv").exists():
            run(["scripts/make_synthetic_data.py", "--out", data_dir])
    if not a.skip_build:
        build = ["src/build.py", "--data-dir", data_dir, "--sample-pct", str(a.sample_pct)]
        if a.memory_limit:
            build += ["--memory-limit", a.memory_limit]
        run(build)
    run(["src/growth.py"])
    run(["src/survival.py"])
    run(["src/engagement.py"])
    run(["src/causal.py", "--treatment", a.treatment])
    run(["src/report.py"] + (["--synthetic"] if a.synthetic else []))
    print("\nResults: results/RESULTS.md and results/figures/")


if __name__ == "__main__":
    main()
