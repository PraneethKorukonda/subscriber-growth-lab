"""Do discounted subscription prices keep subscribers, or just attract people who would
leave anyway?

Unit: one randomly chosen paid transaction per user (list price > 0).
Treatment (default): the user paid less than list price for that period.
Outcomes:
  renewed      the subscription continued past the paid period (renewed within 30 days)
  active_180d  the user was a paying subscriber 180 days after the payment

Discounts are not random. If KKBox gave them to people who were already drifting away,
a naive comparison makes discounts look harmful. We adjust for what was known at the time
of payment (tenure, plan, auto-renew, payment method, last month's listening, demographics)
with three estimators: outcome regression, inverse propensity weighting (IPW) and
cross-fitted augmented IPW (AIPW, doubly robust).

Checks: covariate balance before and after weighting, propensity overlap, and a placebo
test on listening two months *before* the payment, which a discount cannot have changed.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from common import DATA_END, GRACE_DAYS, connect, figure_path, save_metrics

COVARIATES = ["plan_days", "log_list_price", "is_auto_renew", "payment_method_id", "tenure_days",
              "spell_tenure_days", "spell_no", "prior_cancels", "prev_active_days", "prev_hours",
              "prev_full_play_rate", "prev_unique_songs", "age", "gender_code", "city", "registered_via",
              "days_since_registration", "month_of_year"]
CATEGORICAL = ["payment_method_id", "gender_code", "city", "registered_via", "month_of_year"]


def load(con, treatment, max_units):
    treat_sql = {"discount": "(p.actual_amount_paid < p.plan_list_price)::INTEGER",
                 "auto_renew": "p.is_auto_renew"}[treatment]
    df = con.execute(f"""
        WITH firsts AS (SELECT msno, min(txn_date) AS first_txn FROM pay GROUP BY 1),
        elig AS (
            SELECT p.*, s.spell_start, s.spell_end, {treat_sql} AS t,
                   (s.spell_end > p.covered_through)::INTEGER AS renewed,
                   date_trunc('month', p.txn_date)::DATE AS m0
            FROM pay p JOIN spells s USING (msno, spell_no)
            WHERE p.plan_list_price > 0 AND p.actual_amount_paid <= p.plan_list_price
              AND p.txn_date >= DATE '2015-04-01'
              AND p.covered_through + {GRACE_DAYS} <= DATE '{DATA_END}'),
        one AS (SELECT * FROM elig
                QUALIFY row_number() OVER (PARTITION BY msno ORDER BY hash(msno || txn_date::VARCHAR)) = 1)
        SELECT o.msno, o.t, o.renewed,
               CASE WHEN o.txn_date + 180 + {GRACE_DAYS} <= DATE '{DATA_END}' THEN
                    (EXISTS (SELECT 1 FROM spells s2 WHERE s2.msno = o.msno
                             AND o.txn_date + 180 BETWEEN s2.spell_start AND s2.spell_end))::INTEGER END AS active_180d,
               o.payment_plan_days AS plan_days, ln(o.plan_list_price) AS log_list_price, o.is_auto_renew,
               o.payment_method_id,
               datediff('day', f.first_txn, o.txn_date) AS tenure_days,
               datediff('day', o.spell_start, o.txn_date) AS spell_tenure_days,
               o.spell_no,
               (SELECT count(*) FROM cancels c WHERE c.msno = o.msno AND c.cancel_date < o.txn_date) AS prior_cancels,
               coalesce(l1.active_days, 0) AS prev_active_days, coalesce(l1.hours, 0) AS prev_hours,
               CASE WHEN l1.plays > 0 THEN l1.full_plays / l1.plays END AS prev_full_play_rate,
               coalesce(l1.unique_songs, 0) AS prev_unique_songs,
               coalesce(l2.active_days, 0) AS placebo_active_days_2m_before,
               m.age, CASE m.gender WHEN 'male' THEN 1 WHEN 'female' THEN 2 ELSE 0 END AS gender_code,
               m.city, m.registered_via, datediff('day', m.reg_date, o.txn_date) AS days_since_registration,
               month(o.txn_date) AS month_of_year
        FROM one o JOIN firsts f USING (msno)
        LEFT JOIN logs_month l1 ON l1.uid = hash(o.msno) AND l1.month = (o.m0 - INTERVAL 1 MONTH)::DATE
        LEFT JOIN logs_month l2 ON l2.uid = hash(o.msno) AND l2.month = (o.m0 - INTERVAL 2 MONTH)::DATE
        LEFT JOIN members m ON m.msno = o.msno""").df()
    if len(df) > max_units:
        df = df.sample(max_units, random_state=11)
    return df.reset_index(drop=True)


# Strong regularization and early stopping: overfit propensity scores near 0 or 1 make
# IPW and AIPW blow up, so these models are deliberately kept smooth.
GBM = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=5.0,
           early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=3)


def _clf(cat_mask):
    return HistGradientBoostingClassifier(categorical_features=cat_mask, **GBM)


def _reg(cat_mask):
    return HistGradientBoostingRegressor(categorical_features=cat_mask, **GBM)


def cross_fit(X, t, y, binary, folds=5, seed=0):
    """Out-of-fold propensity e(x) and outcome predictions mu1(x), mu0(x) (T-learner)."""
    cat = [c in CATEGORICAL for c in X.columns]
    e, mu1, mu0 = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        ps = _clf(cat).fit(X.iloc[tr], t[tr])
        e[te] = ps.predict_proba(X.iloc[te])[:, 1]
        for arm, out in ((1, mu1), (0, mu0)):
            idx = tr[t[tr] == arm]
            m = (_clf(cat) if binary else _reg(cat)).fit(X.iloc[idx], y[idx])
            out[te] = m.predict_proba(X.iloc[te])[:, 1] if binary else m.predict(X.iloc[te])
    return e, mu1, mu0


def estimate(X, t, y, binary, clip=0.02):
    e, mu1, mu0 = cross_fit(X, t, y, binary)
    e_c = np.clip(e, clip, 1 - clip)
    n = len(y)
    naive = y[t == 1].mean() - y[t == 0].mean()
    naive_se = np.sqrt(y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum())
    w1, w0 = t / e_c, (1 - t) / (1 - e_c)
    ipw = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
    phi = mu1 - mu0 + t * (y - mu1) / e_c - (1 - t) * (y - mu0) / (1 - e_c)
    keep = (e > 0.05) & (e < 0.95)
    return {
        "n": int(n), "treated": int(t.sum()),
        "naive": float(naive), "naive_ci": [float(naive - 1.96 * naive_se), float(naive + 1.96 * naive_se)],
        "outcome_regression": float((mu1 - mu0).mean()),
        "ipw": float(ipw),
        "aipw": float(phi.mean()),
        "aipw_ci": [float(phi.mean() - 1.96 * phi.std(ddof=1) / np.sqrt(n)),
                    float(phi.mean() + 1.96 * phi.std(ddof=1) / np.sqrt(n))],
        "aipw_trimmed_05_95": float(phi[keep].mean()),
        "share_outside_05_95": float(1 - keep.mean()),
    }, e_c


def smd(x, t, w=None):
    w = np.ones(len(x)) if w is None else w
    a, b = x[t == 1], x[t == 0]
    wa, wb = w[t == 1], w[t == 0]
    ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
    va, vb = np.average((a - ma) ** 2, weights=wa), np.average((b - mb) ** 2, weights=wb)
    return (ma - mb) / np.sqrt((va + vb) / 2 + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", choices=["discount", "auto_renew"], default="discount")
    ap.add_argument("--max-units", type=int, default=400_000)
    a = ap.parse_args()
    con = connect()
    df = load(con, a.treatment, a.max_units)
    covs = [c for c in COVARIATES if not (a.treatment == "auto_renew" and c == "is_auto_renew")]
    X = df[covs].astype("float64")
    t = df["t"].values.astype(int)
    print(f"units: {len(df):,}  treated: {t.sum():,} ({t.mean():.1%})")
    if t.sum() < 300 or (1 - t).sum() < 300:
        raise SystemExit("Too few treated or control units. Try --treatment auto_renew.")

    res = {"treatment": a.treatment}
    res["renewed"], e = estimate(X, t, df["renewed"].values.astype(float), binary=True)
    sub = df["active_180d"].notna().values
    if sub.sum() > 1000:
        res["active_180d"], _ = estimate(X[sub].reset_index(drop=True), t[sub],
                                         df.loc[sub, "active_180d"].values.astype(float), binary=True)
    res["placebo_active_days_2m_before"], _ = estimate(X, t, df["placebo_active_days_2m_before"].values.astype(float),
                                                       binary=False)

    # Balance before vs. after IPW weighting
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    bal = []
    for c in covs:
        x = X[c].fillna(X[c].median()).values
        if np.nanstd(x) == 0:
            continue
        bal.append({"covariate": c, "smd_raw": float(smd(x, t)), "smd_weighted": float(smd(x, t, w))})
    bal = pd.DataFrame(bal).sort_values("smd_raw", key=np.abs, ascending=False)
    res["balance"] = {"max_abs_smd_raw": float(bal["smd_raw"].abs().max()),
                      "max_abs_smd_weighted": float(bal["smd_weighted"].abs().max()),
                      "table": bal.round(3).to_dict(orient="records")}
    res["treated_vs_control_prev_active_days"] = [float(df.loc[t == 1, "prev_active_days"].mean()),
                                                  float(df.loc[t == 0, "prev_active_days"].mean())]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    yy = np.arange(len(bal))
    ax1.scatter(bal["smd_raw"].abs(), yy, label="before weighting", color="#d9534f")
    ax1.scatter(bal["smd_weighted"].abs(), yy, label="after IPW", color="#2f6fb3")
    ax1.axvline(0.1, ls="--", color="grey"); ax1.set_yticks(yy); ax1.set_yticklabels(bal["covariate"], fontsize=7)
    ax1.set_xlabel("|standardized mean difference|"); ax1.set_title("Covariate balance"); ax1.legend(fontsize=8)
    r = res["renewed"]
    labels = ["naive", "outcome\nregression", "IPW", "AIPW"]
    vals = [r["naive"], r["outcome_regression"], r["ipw"], r["aipw"]]
    ax2.bar(labels, [100 * v for v in vals], color=["#999999", "#7fb3e6", "#7fb3e6", "#2f6fb3"])
    ax2.errorbar([3], [100 * r["aipw"]], yerr=[[100 * (r["aipw"] - r["aipw_ci"][0])], [100 * (r["aipw_ci"][1] - r["aipw"])]],
                 color="black", capsize=4)
    ax2.axhline(0, color="black", lw=0.8); ax2.set_ylabel("effect on renewal (points)")
    ax2.set_title(f"Effect of {a.treatment.replace('_', '-')} on renewal")
    fig.tight_layout(); fig.savefig(figure_path("causal_effect.png"), dpi=150); plt.close(fig)

    save_metrics("causal", res)
    for k in ["renewed", "active_180d", "placebo_active_days_2m_before"]:
        if k in res:
            print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res[k].items()})
    print(bal.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
