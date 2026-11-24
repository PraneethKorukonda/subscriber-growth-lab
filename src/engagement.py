"""Does early listening predict who is still subscribed six months later?

For each new subscriber (first spell only), measure listening in the first full calendar
month after they subscribed, then check whether that same subscription is still running
180 days after it started. This is a leading indicator, not a causal effect: people who
were always going to stay may simply listen more.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_END, GRACE_DAYS, connect, figure_path, save_metrics

BUCKETS = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 31)]


def load(con):
    return con.execute(f"""
        WITH s AS (
            SELECT msno, spell_start, spell_end,
                   (date_trunc('month', spell_start) + INTERVAL 1 MONTH)::DATE AS m1
            FROM spells
            WHERE spell_no = 1 AND spell_start >= DATE '2015-04-01'
              AND spell_start + 180 + {GRACE_DAYS} <= DATE '{DATA_END}')
        SELECT s.msno, s.spell_start,
               coalesce(l.active_days, 0) AS active_days, coalesce(l.hours, 0) AS hours,
               CASE WHEN l.plays > 0 THEN l.full_plays / l.plays END AS full_play_rate,
               (s.spell_end >= s.spell_start + 180)::INTEGER AS retained_180d
        FROM s LEFT JOIN logs_month l ON l.uid = hash(s.msno) AND l.month = s.m1
        -- only subscribers still covered through the end of that first full month
        WHERE s.spell_end >= (s.m1 + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE""").df()


def main():
    con = connect()
    if con.execute("SELECT count(*) FROM logs_month").fetchone()[0] == 0:
        print("No user logs loaded, skipping engagement analysis.")
        save_metrics("engagement", {"skipped": "no user logs"})
        return
    df = load(con)
    rows = []
    for lo, hi in BUCKETS:
        g = df[(df["active_days"] >= lo) & (df["active_days"] <= hi)]
        rows.append({"active_days": f"{lo}" if lo == hi else f"{lo}-{hi}", "subscribers": int(len(g)),
                     "retained_180d": float(g["retained_180d"].mean()) if len(g) else None})
    tab = pd.DataFrame(rows)

    # "Aha" threshold: the number of active days that best separates who stays (Youden's J)
    best = None
    for k in range(1, 31):
        above = df["active_days"] >= k
        if above.mean() < 0.05 or above.mean() > 0.95:
            continue
        tpr = above[df["retained_180d"] == 1].mean()
        fpr = above[df["retained_180d"] == 0].mean()
        j = tpr - fpr
        if best is None or j > best["youden_j"]:
            best = {"threshold_days": k, "youden_j": float(j), "share_above": float(above.mean()),
                    "retained_above": float(df.loc[above, "retained_180d"].mean()),
                    "retained_below": float(df.loc[~above, "retained_180d"].mean())}
    if best:
        best["ratio"] = best["retained_above"] / max(best["retained_below"], 1e-9)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tab["active_days"], 100 * tab["retained_180d"].fillna(0), color="#2f6fb3")
    ax.set_xlabel("days listened in first full month"); ax.set_ylabel("% still subscribed at day 180")
    ax.set_title("Early listening vs. 6-month retention (new subscribers)")
    fig.tight_layout(); fig.savefig(figure_path("early_listening.png"), dpi=150); plt.close(fig)

    save_metrics("engagement", {
        "new_subscribers": int(len(df)),
        "overall_retained_180d": float(df["retained_180d"].mean()),
        "by_active_days": tab.to_dict(orient="records"),
        "aha_threshold": best,
        "corr_hours_retained": float(np.corrcoef(np.log1p(df["hours"]), df["retained_180d"])[0, 1]),
    })
    print(tab)
    print("threshold:", best)


if __name__ == "__main__":
    main()
