"""Duolingo-style growth accounting for paid subscribers.

Every month each user who has ever subscribed is in one state:
  new          first month ever as a paying subscriber
  retained     paid last month and this month
  resurrected  paid this month, not last month, but paid at some point before
  churned      paid last month, not this month
  dormant      paid before, not last month and not this month

Net growth = new + resurrected - churned. The model then asks which lever moves the
subscriber base most: +1 point of monthly retention, +1 point of resurrection, or +10% new.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_END, REPORT_START, RESULTS, connect, figure_path, save_metrics


def monthly_states(con):
    df = con.execute("""
        WITH um AS (SELECT DISTINCT msno, month FROM sub_months),
        f AS (SELECT msno, min(month) AS fm FROM um GROUP BY 1),
        l AS (SELECT um.msno, um.month, f.fm,
                     lag(um.month) OVER (PARTITION BY um.msno ORDER BY um.month) AS prev
              FROM um JOIN f USING (msno))
        SELECT month,
               count(*) AS active,
               count(*) FILTER (WHERE month = fm) AS new,
               count(*) FILTER (WHERE prev = (month - INTERVAL 1 MONTH)::DATE) AS retained,
               count(*) FILTER (WHERE month <> fm AND (prev IS NULL OR prev < (month - INTERVAL 1 MONTH)::DATE)) AS resurrected
        FROM l GROUP BY 1 ORDER BY 1""").df()
    df["month"] = pd.to_datetime(df["month"])
    df = df.set_index("month").asfreq("MS", fill_value=0)
    df["prev_active"] = df["active"].shift(1)
    df["churned"] = df["prev_active"] - df["retained"]
    df["ever_active"] = df["new"].cumsum()
    df["dormant_start"] = (df["ever_active"] - df["active"]).shift(1)
    df["retention_rate"] = df["retained"] / df["prev_active"]
    df["resurrection_rate"] = df["resurrected"] / df["dormant_start"]
    df["net_growth"] = df["new"] + df["resurrected"] - df["churned"]
    return df


def listening_risk(con):
    """Among paying subscribers, do the ones who stopped listening churn more next month?"""
    df = con.execute(f"""
        WITH um AS (SELECT DISTINCT msno, month FROM sub_months),
        nxt AS (SELECT a.msno, a.month, (b.msno IS NOT NULL)::INTEGER AS active_next
                FROM um a LEFT JOIN um b ON a.msno = b.msno AND b.month = (a.month + INTERVAL 1 MONTH)::DATE
                WHERE a.month BETWEEN DATE '{REPORT_START}' AND (date_trunc('month', DATE '{DATA_END}') - INTERVAL 1 MONTH)::DATE)
        SELECT CASE WHEN coalesce(l.active_days, 0) = 0 THEN 'paid, did not listen'
                    WHEN l.active_days < 10 THEN 'listened 1-9 days'
                    ELSE 'listened 10+ days' END AS listening,
               count(*) AS subscriber_months,
               1 - avg(active_next) AS churn_next_month
        FROM nxt LEFT JOIN logs_month l ON l.uid = hash(nxt.msno) AND l.month = nxt.month
        GROUP BY 1 ORDER BY 1""").df()
    return df


def project(df, months=12, window=6):
    """Simple two-pool (active, dormant) projection from the recent average rates."""
    recent = df.loc[df.index >= pd.Timestamp(REPORT_START)].dropna(subset=["retention_rate"]).tail(window)
    r = recent["retention_rate"].mean()
    s = recent["resurrection_rate"].mean()
    n = recent["new"].mean()
    a0 = df["active"].iloc[-1]
    d0 = df["ever_active"].iloc[-1] - a0

    def run(r, s, n):
        a, d = a0, d0
        for _ in range(months):
            a, d = a * r + d * s + n, d * (1 - s) + a * (1 - r)
        return a

    base = run(r, s, n)
    scen = {
        "base": base,
        "retention +1 pt": run(min(r + 0.01, 1), s, n),
        "resurrection +1 pt": run(r, s + 0.01, n),
        "new subscribers +10%": run(r, s, n * 1.10),
    }
    return {
        "inputs": {"monthly_retention": r, "monthly_resurrection": s, "new_per_month": n,
                   "active_now": float(a0), "dormant_now": float(d0), "months_ahead": months},
        "subscribers_after_12_months": scen,
        "lift_vs_base_pct": {k: 100 * (v / base - 1) for k, v in scen.items() if k != "base"},
    }


def plot(df):
    d = df.loc[df.index >= pd.Timestamp(REPORT_START)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = d.index
    ax1.bar(x, d["new"], width=20, label="new", color="#2f6fb3")
    ax1.bar(x, d["resurrected"], width=20, bottom=d["new"], label="resurrected", color="#7fb3e6")
    ax1.bar(x, -d["churned"], width=20, label="churned", color="#d9534f")
    ax1.plot(x, d["net_growth"], color="black", marker="o", ms=3, label="net growth")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_title("Monthly subscriber flows")
    ax1.legend(ncol=4, fontsize=8, loc="upper left")
    ax2.plot(x, 100 * d["retention_rate"], marker="o", ms=3, color="#2f6fb3")
    ax2.set_ylabel("monthly retention %")
    fig.tight_layout()
    fig.savefig(figure_path("growth_accounting.png"), dpi=150)
    plt.close(fig)


def main():
    con = connect()
    df = monthly_states(con)
    # Identity check: active_m = active_{m-1} - churned + new + resurrected
    chk = df.dropna(subset=["prev_active"])
    gap = (chk["prev_active"] - chk["churned"] + chk["new"] + chk["resurrected"] - chk["active"]).abs().max()
    assert gap == 0, f"growth accounting does not add up (max gap {gap})"

    risk = listening_risk(con)
    proj = project(df)
    plot(df)
    rep = df.loc[df.index >= pd.Timestamp(REPORT_START)]
    df.reset_index().to_csv(RESULTS / "growth_monthly.csv", index=False)
    save_metrics("growth", {
        "months_reported": [str(rep.index.min().date()), str(rep.index.max().date())],
        "active_start": int(rep["active"].iloc[0]), "active_end": int(rep["active"].iloc[-1]),
        "avg_monthly_retention": float(rep["retention_rate"].mean()),
        "avg_monthly_resurrection": float(rep["resurrection_rate"].mean()),
        "share_of_gross_adds_from_resurrection": float(rep["resurrected"].sum() / (rep["resurrected"].sum() + rep["new"].sum())),
        "listening_vs_next_month_churn": risk.to_dict(orient="records"),
        "projection": proj,
    })
    print(rep[["active", "new", "resurrected", "churned", "retention_rate", "resurrection_rate"]].tail(8))
    print(risk)
    print(proj["lift_vs_base_pct"])


if __name__ == "__main__":
    main()
