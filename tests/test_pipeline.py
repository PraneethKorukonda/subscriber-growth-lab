"""End-to-end check on synthetic data with a known answer.

  python tests/test_pipeline.py      (or: pytest tests/)

Takes about 2-4 minutes. It runs the whole pipeline in a temporary folder, then checks
that the numbers add up and that AIPW recovers the planted +5 point discount effect
while the naive comparison does not.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_synthetic_data import TAU  # noqa: E402


def run_pipeline(tmp):
    env = dict(os.environ, SGL_WORK=str(tmp / "work"), SGL_RESULTS=str(tmp / "results"))
    data = tmp / "data"
    py = sys.executable
    subprocess.run([py, "scripts/make_synthetic_data.py", "--out", str(data), "--users", "15000", "--seed", "7"],
                   cwd=ROOT, env=env, check=True)
    for step in (["src/build.py", "--data-dir", str(data)], ["src/growth.py"], ["src/survival.py"],
                 ["src/engagement.py"], ["src/causal.py"], ["src/report.py", "--synthetic"]):
        subprocess.run([py] + step, cwd=ROOT, env=env, check=True)
    return json.loads((tmp / "results" / "metrics.json").read_text())


def check(m):
    r = m["causal"]["renewed"]
    se = (r["aipw_ci"][1] - r["aipw_ci"][0]) / (2 * 1.96)
    assert r["naive"] < 0, "naive estimate should be pulled negative by confounding"
    assert abs(r["aipw"] - TAU) < 3 * se, f"AIPW {r['aipw']:.3f} too far from truth {TAU}"
    assert r["aipw_ci"][0] > r["naive"], "adjusted estimate should sit above the biased naive one"
    p = m["causal"]["placebo_active_days_2m_before"]
    assert abs(p["aipw"]) < abs(p["naive"]) / 3, "placebo should shrink toward zero after adjustment"
    assert m["causal"]["balance"]["max_abs_smd_weighted"] < m["causal"]["balance"]["max_abs_smd_raw"]
    hz = m["survival"]["hazard"]
    assert 0.6 < hz["row_level"]["auc"] <= 1
    assert all(v == v for v in hz["mean_abs_pct_error"].values() if v is not None)
    assert m["growth"]["active_end"] > 0
    th = m["engagement"]["aha_threshold"]
    assert th["retained_above"] > th["retained_below"], "more early listening should go with more retention"
    print(f"OK  naive {r['naive']:+.3f}  AIPW {r['aipw']:+.3f} (truth {TAU:+.3f}, se {se:.3f})  "
          f"placebo naive {p['naive']:+.2f} -> AIPW {p['aipw']:+.2f} days")


def test_pipeline():
    with tempfile.TemporaryDirectory() as d:
        check(run_pipeline(Path(d)))


if __name__ == "__main__":
    test_pipeline()
