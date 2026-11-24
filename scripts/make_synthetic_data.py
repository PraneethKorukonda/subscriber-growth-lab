"""Generate fake data in the exact KKBox file format, with known answers planted in it.

Use it to check the whole pipeline in a few minutes before running on the real 30 GB data,
and to confirm the causal estimators recover a known effect:

  * Discounts go mostly to users who listened little last month (confounding).
  * Low listening lowers renewal, so a naive comparison makes discounts look bad.
  * Last month's listening is fully visible in the logs, so adjusting for it should work.
  * The true effect of a discount on renewal is +TAU (5 points) for everyone.

Usage: python scripts/make_synthetic_data.py --out data/synthetic --users 12000
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TAU = 0.05
END = date(2017, 3, 31)
BASE = date(2014, 12, 1)


def midx(d):
    return (d.year - BASE.year) * 12 + d.month - BASE.month


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n_users, seed):
    rng = np.random.default_rng(seed)
    n_months = midx(END) + 1
    members, txns, logs = [], [], []
    for u in range(n_users):
        msno = f"u{u:07d}"
        reg = BASE + timedelta(days=int(rng.integers(0, (date(2017, 1, 15) - BASE).days)))
        e_u = rng.normal()
        a = np.zeros(n_months)
        for m in range(1, n_months):
            a[m] = 0.6 * a[m - 1] + rng.normal(0, 0.7)
        e = e_u + a                                   # monthly engagement
        auto = rng.random() < sigmoid(0.8 + 0.6 * e_u)
        method = 41 if auto else int(rng.choice([36, 37, 38, 39, 40]))
        members.append((msno, int(rng.integers(1, 23)), int(rng.integers(18, 55)) if rng.random() < 0.6 else 0,
                        rng.choice(["male", "female", ""]), int(rng.choice([3, 4, 7, 9])), reg.strftime("%Y%m%d")))

        # Daily listening from registration on. Generated first, so the pipeline sees exactly
        # the signal that drives discounts and renewals: no hidden confounder.
        days = np.array([(reg + timedelta(days=k)) for k in range((END - reg).days + 1)])
        mi = np.array([midx(d) for d in days])
        on = rng.random(len(days)) < sigmoid(-0.2 + 1.2 * e[mi])
        active = np.bincount(mi[on], minlength=n_months)       # listening days per month
        z = (active - 15) / 8                                   # what the pipeline measures

        start = max(reg + timedelta(days=int(rng.integers(35, 120))), date(2015, 1, 1))
        while start <= END:
            t = start
            while t <= END:
                plan = 30 if rng.random() < 0.9 else 90
                price = 149 * plan // 30
                m_prev = max(midx(t) - 1, 0)
                disc = rng.random() < sigmoid(-1.8 - 1.0 * z[m_prev])
                paid = round(price * 0.66) if disc else price
                exp = t + timedelta(days=plan)
                txns.append((msno, method, plan, price, paid, int(auto), t.strftime("%Y%m%d"), exp.strftime("%Y%m%d"), 0))
                p_renew = 0.66 + 0.15 * np.tanh(z[m_prev]) + 0.10 * auto + TAU * disc
                if rng.random() < p_renew:
                    t = exp if auto else exp + timedelta(days=int(rng.integers(0, 10)))
                    continue
                if auto and rng.random() < 0.5:      # user turns off auto-renew during the period
                    c = t + timedelta(days=int(rng.integers(1, plan)))
                    txns.append((msno, method, plan, price, paid, 0, c.strftime("%Y%m%d"), exp.strftime("%Y%m%d"), 1))
                break
            else:
                break
            if rng.random() < 0.35:                  # comes back later = resurrected
                start = exp + timedelta(days=int(rng.integers(45, 240)))
            else:
                break

        if not on.any():
            continue
        d_on, e_on = days[on], e[mi[on]]
        plays = rng.poisson(12 * np.exp(0.3 * np.clip(e_on, -2, 2))) + 1
        full = rng.binomial(plays, sigmoid(0.5 + 0.4 * e_on))
        rest = plays - full
        n25 = rng.binomial(rest, 0.5); n50 = rng.binomial(rest - n25, 0.4); n75 = rng.binomial(rest - n25 - n50, 0.5)
        n985 = rest - n25 - n50 - n75
        secs = full * 230 + n985 * 210 + n75 * 150 + n50 * 90 + n25 * 30 + rng.normal(0, 20, len(plays))
        logs.append(pd.DataFrame({"msno": msno, "date": [d.strftime("%Y%m%d") for d in d_on],
                                  "num_25": n25, "num_50": n50, "num_75": n75, "num_985": n985,
                                  "num_100": full, "num_unq": np.maximum(1, (plays * 0.7).astype(int)),
                                  "total_secs": secs.round(3)}))
    mem = pd.DataFrame(members, columns=["msno", "city", "bd", "gender", "registered_via", "registration_init_time"])
    tx = pd.DataFrame(txns, columns=["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                                     "actual_amount_paid", "is_auto_renew", "transaction_date",
                                     "membership_expire_date", "is_cancel"])
    lg = pd.concat(logs, ignore_index=True)
    return mem, tx, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--users", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mem, tx, lg = simulate(a.users, a.seed)
    mem.to_csv(out / "members_v3.csv", index=False)
    # Same split as Kaggle: March 2017 is in the *_v2 files
    v2 = tx["transaction_date"] >= "20170301"
    tx[~v2].to_csv(out / "transactions.csv", index=False)
    tx[v2].to_csv(out / "transactions_v2.csv", index=False)
    lv2 = lg["date"] >= "20170301"
    lg[~lv2].to_csv(out / "user_logs.csv", index=False)
    lg[lv2].to_csv(out / "user_logs_v2.csv", index=False)
    print(f"members {len(mem):,}  transactions {len(tx):,}  log rows {len(lg):,}  -> {out}")
    print(f"planted effect of a discount on renewal: +{TAU:.2f}")


if __name__ == "__main__":
    main()
