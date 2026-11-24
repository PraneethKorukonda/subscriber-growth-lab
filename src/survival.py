"""When do subscribers leave, and can a hazard model forecast monthly churn better than
the usual trailing-average rate?

1. Kaplan-Meier survival curves for subscription spells, split by how the spell started
   (auto-renew vs. manual, 30-day vs. longer plans, discounted vs. full price).
   Spells still running at the end of the data are censored, not counted as retained or lost.
2. A discrete-time hazard model: one row per subscriber per month, predicting whether the
   subscription ends that month from what is known at the start of the month.
3. A backtest on the last three fully observed months against two baselines:
   trailing 3-month churn rate and same-month-last-year churn rate.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import DATA_END, GRACE_DAYS, connect, figure_path, save_metrics

TRAIN_START = "2015-07-01"
# Churn in month m is only fully observed once the 30-day grace period has passed,
# so the last usable month is the one whose end + 30 days falls before the data ends.
LAST_OBSERVED = (pd.Timestamp(DATA_END) - pd.Timedelta(days=GRACE_DAYS)).to_period("M").to_timestamp()
if (LAST_OBSERVED + pd.offsets.MonthEnd(0) + pd.Timedelta(days=GRACE_DAYS)) > pd.Timestamp(DATA_END):
    LAST_OBSERVED -= pd.offsets.MonthBegin(1)
HOLDOUT = [LAST_OBSERVED - pd.offsets.MonthBegin(k) for k in (2, 1, 0)]

FEATURES = ["tenure_months", "month_of_year", "spell_no", "days_left", "is_auto_renew", "plan_days",
            "list_price", "discount", "cancel_known", "payment_method_id", "prev_active_days",
            "prev_hours", "prev_full_play_rate", "age", "gender_code", "city", "registered_via"]
CATEGORICAL = ["month_of_year", "payment_method_id", "gender_code", "city", "registered_via"]


# ---------------------------------------------------------------- Kaplan-Meier
def kaplan_meier(durations, events):
    d = pd.DataFrame({"t": durations, "e": events}).groupby("t")["e"].agg(["sum", "count"])
    at_risk = d["count"][::-1].cumsum()[::-1]
    surv = (1 - d["sum"] / at_risk).cumprod()
    return pd.concat([pd.Series([1.0], index=[0]), surv[surv.index > 0]])


def survival_at(surv, day):
    s = surv[surv.index <= day]
    return float(s.iloc[-1]) if len(s) else 1.0


def km_section(con):
    sp = con.execute("""
        SELECT duration_days, churned, start_auto_renew, start_plan_days, start_discount
        FROM spells WHERE spell_start >= DATE '2015-04-01'""").df()
    groups = {
        "auto-renew start": sp["start_auto_renew"] == 1,
        "manual start": sp["start_auto_renew"] == 0,
        "30-day plan": sp["start_plan_days"] == 30,
        "other plan length": sp["start_plan_days"] != 30,
        "discounted start": sp["start_discount"] == 1,
        "full-price start": sp["start_discount"] == 0,
    }
    out, curves = {}, {}
    for name, mask in groups.items():
        g = sp[mask]
        if len(g) < 50:
            continue
        s = kaplan_meier(g["duration_days"].values, g["churned"].values)
        curves[name] = s
        below = s[s <= 0.5]
        out[name] = {"spells": int(len(g)), "share_censored": float(1 - g["churned"].mean()),
                     "median_days": float(below.index[0]) if len(below) else None,
                     "survival_90d": survival_at(s, 90), "survival_180d": survival_at(s, 180),
                     "survival_365d": survival_at(s, 365)}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["auto-renew start", "manual start", "discounted start", "full-price start"]:
        if name in curves:
            s = curves[name]
            ax.step(s.index, s.values, where="post", label=name)
    ax.set_xlim(0, 730); ax.set_ylim(0, 1.02)
    ax.set_xlabel("days since subscription started"); ax.set_ylabel("share still subscribed")
    ax.set_title("Kaplan-Meier survival of subscription spells"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_path("survival_curves.png"), dpi=150); plt.close(fig)
    return out


# ---------------------------------------------------------------- hazard model
def build_panel(con):
    con.execute(f"""
        CREATE OR REPLACE TABLE hazard_panel AS
        WITH base AS (
            SELECT msno, spell_no, month, spell_start, spell_end, churned,
                   greatest(month, spell_start) AS asof_date
            FROM sub_months
            WHERE month BETWEEN DATE '{TRAIN_START}' AND DATE '{LAST_OBSERVED.date()}')
        SELECT b.msno, b.month,
               (b.churned = 1 AND date_trunc('month', b.spell_end) = b.month)::INTEGER AS event,
               datediff('month', date_trunc('month', b.spell_start), b.month) AS tenure_months,
               month(b.month) AS month_of_year,
               b.spell_no,
               datediff('day', b.month, p.expire_through) AS days_left,
               p.is_auto_renew, p.payment_plan_days AS plan_days, p.plan_list_price AS list_price,
               (p.actual_amount_paid < p.plan_list_price)::INTEGER AS discount,
               (c.cancel_date IS NOT NULL AND c.cancel_date >= p.txn_date)::INTEGER AS cancel_known,
               p.payment_method_id,
               coalesce(l.active_days, 0) AS prev_active_days, coalesce(l.hours, 0) AS prev_hours,
               CASE WHEN l.plays > 0 THEN l.full_plays / l.plays END AS prev_full_play_rate,
               m.age, CASE m.gender WHEN 'male' THEN 1 WHEN 'female' THEN 2 ELSE 0 END AS gender_code,
               m.city, m.registered_via
        FROM base b
        ASOF LEFT JOIN pay p ON b.msno = p.msno AND b.spell_no = p.spell_no AND p.txn_date <= b.asof_date
        -- only cancellations that already happened by the start of the month
        ASOF LEFT JOIN cancels c ON b.msno = c.msno AND c.cancel_date <= b.asof_date
        LEFT JOIN logs_month l ON l.uid = hash(b.msno) AND l.month = (b.month - INTERVAL 1 MONTH)::DATE
        LEFT JOIN members m ON m.msno = b.msno""")


def fetch(con, where, max_rows):
    n = con.execute(f"SELECT count(*) FROM hazard_panel WHERE {where}").fetchone()[0]
    frac = min(1.0, max_rows / max(n, 1))
    sample = f"AND hash(msno || month::VARCHAR) % 1000000 < {int(frac * 1_000_000)}" if frac < 1 else ""
    df = con.execute(f"SELECT * FROM hazard_panel WHERE {where} {sample}").df()
    return df, frac


def to_xy(df):
    X = df[FEATURES].astype("float64")
    return X, df["event"].values


def hazard_section(con, max_train_rows, max_test_rows):
    build_panel(con)
    hold_start = HOLDOUT[0].date()
    train, f_tr = fetch(con, f"month < DATE '{hold_start}'", max_train_rows)
    test, f_te = fetch(con, f"month >= DATE '{hold_start}'", max_test_rows)
    print(f"hazard rows: train {len(train):,} (sampled {f_tr:.0%}), test {len(test):,} (sampled {f_te:.0%})")
    Xtr, ytr = to_xy(train)
    Xte, yte = to_xy(test)
    cat_mask = [c in CATEGORICAL for c in FEATURES]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=48,
                                           l2_regularization=1.0, categorical_features=cat_mask,
                                           early_stopping=True, validation_fraction=0.1, random_state=7)
    model.fit(Xtr, ytr)
    test["p"] = model.predict_proba(Xte)[:, 1]

    # Monthly churn rate history (on the same sampled rows) for the baselines
    hist = pd.concat([train[["month", "event"]], test[["month", "event"]]])
    hist["month"] = pd.to_datetime(hist["month"])
    rate = hist.groupby("month")["event"].mean()

    rows = []
    for m in HOLDOUT:
        t = test[pd.to_datetime(test["month"]) == m]
        actual = t["event"].sum()
        trailing = rate[(rate.index < m) & (rate.index >= m - pd.offsets.MonthBegin(3))].mean() * len(t)
        ly = m - pd.DateOffset(years=1)
        seasonal = rate.get(ly, np.nan) * len(t)
        rows.append({"month": str(m.date()), "subscribers": len(t), "actual_churn": int(actual),
                     "hazard_model": float(t["p"].sum()), "trailing_3m_rate": float(trailing),
                     "same_month_last_year": float(seasonal)})
    bt = pd.DataFrame(rows)
    ape = {c: float((100 * (bt[c] - bt["actual_churn"]).abs() / bt["actual_churn"]).mean())
           for c in ["hazard_model", "trailing_3m_rate", "same_month_last_year"]}

    sub = test.sample(min(50_000, len(test)), random_state=1)
    imp = permutation_importance(model, sub[FEATURES].astype("float64"), sub["event"].values,
                                 scoring="roc_auc", n_repeats=3, random_state=1)
    importance = (pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).head(8))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bt))
    for i, (c, lab) in enumerate([("actual_churn", "actual"), ("hazard_model", "hazard model"),
                                  ("trailing_3m_rate", "trailing 3-month rate"),
                                  ("same_month_last_year", "same month last year")]):
        ax.bar(x + (i - 1.5) * 0.2, bt[c], width=0.2, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(bt["month"]); ax.set_ylabel("churned subscribers (sample)")
    ax.set_title("Monthly churn forecast backtest"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figure_path("churn_backtest.png"), dpi=150); plt.close(fig)

    return {
        "train_months": [TRAIN_START, str((HOLDOUT[0] - pd.offsets.MonthBegin(1)).date())],
        "holdout_months": [str(m.date()) for m in HOLDOUT],
        "row_level": {"auc": float(roc_auc_score(yte, test["p"])), "log_loss": float(log_loss(yte, test["p"])),
                      "brier": float(brier_score_loss(yte, test["p"])), "base_rate": float(yte.mean())},
        "backtest": bt.to_dict(orient="records"),
        "mean_abs_pct_error": ape,
        "top_features_auc_drop": importance.round(4).to_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-rows", type=int, default=1_500_000)
    ap.add_argument("--max-test-rows", type=int, default=1_500_000)
    a = ap.parse_args()
    con = connect()
    km = km_section(con)
    hz = hazard_section(con, a.max_train_rows, a.max_test_rows)
    save_metrics("survival", {"kaplan_meier": km, "hazard": hz})
    print(pd.DataFrame(km).T)
    print(pd.DataFrame(hz["backtest"]))
    print("mean abs % error:", hz["mean_abs_pct_error"])


if __name__ == "__main__":
    main()
