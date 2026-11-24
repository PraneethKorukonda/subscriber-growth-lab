"""Turn results/metrics.json into results/RESULTS.md, with draft resume bullets."""
import argparse

from common import RESULTS, load_metrics


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def pts(x, d=1):
    return "n/a" if x is None else f"{100 * x:+.{d}f} pts"


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.get(c, str)(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    m = load_metrics()
    d, g, s, e, c = (m.get(k, {}) for k in ["data", "growth", "survival", "engagement", "causal"])
    L = ["# Subscriber Growth Lab: results", ""]
    if a.synthetic:
        L += ["> **SYNTHETIC DATA.** These numbers come from fake data with planted answers. "
              "Do not put them on a resume. Rerun on the real KKBox files.", ""]

    L += ["## Data", "",
          f"- Users: {d.get('users', 0):,} ({d.get('sample_pct')}% sample) · paid transactions: {d.get('paid_transactions', 0):,} "
          f"· cancellations: {d.get('cancellations', 0):,}",
          f"- Subscription spells: {d.get('spells', 0):,} · subscriber-months: {d.get('subscriber_months', 0):,} "
          f"· listening user-months: {d.get('log_user_months', 0):,}",
          f"- Discounted payments: {pct(d.get('share_discounted_payments'))} · auto-renew payments: "
          f"{pct(d.get('share_auto_renew_payments'))}", ""]

    if g:
        p = g["projection"]
        L += ["## 1. Growth accounting (Duolingo-style user states)", "",
              f"Months {g['months_reported'][0]} to {g['months_reported'][1]}: active subscribers went from "
              f"{g['active_start']:,} to {g['active_end']:,}. Average monthly retention {pct(g['avg_monthly_retention'])}, "
              f"monthly resurrection of lapsed subscribers {pct(g['avg_monthly_resurrection'], 2)}. "
              f"{pct(g['share_of_gross_adds_from_resurrection'])} of gross adds were returning subscribers.", "",
              "**Which lever matters most?** Subscribers after 12 months vs. the base projection:", "",
              table([{"lever": k, "lift": v} for k, v in p["lift_vs_base_pct"].items()], ["lever", "lift"],
                    {"lift": lambda v: f"{v:+.1f}%"}), "",
              "**Paying but not listening = at risk.** Churn next month by listening this month:", "",
              table(g["listening_vs_next_month_churn"], ["listening", "subscriber_months", "churn_next_month"],
                    {"subscriber_months": lambda v: f"{v:,}", "churn_next_month": pct}), "",
              "![growth](figures/growth_accounting.png)", ""]

    if s:
        km, hz = s["kaplan_meier"], s["hazard"]
        L += ["## 2. Survival and churn forecasting", "",
              table([{"group": k, **v} for k, v in km.items()],
                    ["group", "spells", "median_days", "survival_90d", "survival_180d", "survival_365d", "share_censored"],
                    {"spells": lambda v: f"{v:,}", "median_days": lambda v: "not reached" if v is None else f"{v:.0f}",
                     "survival_90d": pct, "survival_180d": pct, "survival_365d": pct, "share_censored": pct}), "",
              "![survival](figures/survival_curves.png)", "",
              f"**Discrete-time hazard model**, trained {hz['train_months'][0]} to {hz['train_months'][1]}, "
              f"tested on {', '.join(hz['holdout_months'])}. Row-level AUC {hz['row_level']['auc']:.3f}, "
              f"log loss {hz['row_level']['log_loss']:.4f}.", "",
              table(hz["backtest"], ["month", "subscribers", "actual_churn", "hazard_model", "trailing_3m_rate",
                                     "same_month_last_year"],
                    {"subscribers": lambda v: f"{v:,}", "actual_churn": lambda v: f"{v:,}",
                     "hazard_model": lambda v: f"{v:,.0f}", "trailing_3m_rate": lambda v: f"{v:,.0f}",
                     "same_month_last_year": lambda v: "n/a" if v != v else f"{v:,.0f}"}), "",
              "Mean absolute % error of the monthly churn count: " +
              ", ".join(f"{k} {v:.1f}%" for k, v in hz["mean_abs_pct_error"].items()), "",
              "Top features (drop in AUC when shuffled): " +
              ", ".join(f"{k} ({v:.3f})" for k, v in hz["top_features_auc_drop"].items()), "",
              "![backtest](figures/churn_backtest.png)", ""]

    if e and "skipped" not in e:
        th = e.get("aha_threshold") or {}
        L += ["## 3. Early listening as a leading indicator", "",
              f"{e['new_subscribers']:,} new subscribers; {pct(e['overall_retained_180d'])} still subscribed at day 180.", "",
              table(e["by_active_days"], ["active_days", "subscribers", "retained_180d"],
                    {"subscribers": lambda v: f"{v:,}", "retained_180d": pct}), ""]
        if th:
            L += [f"Best split: **{th['threshold_days']}+ listening days** in the first full month "
                  f"({pct(th['share_above'])} of new subscribers). They kept {pct(th['retained_above'])} at day 180 vs. "
                  f"{pct(th['retained_below'])} for the rest ({th['ratio']:.1f}x). Correlation, not causation.", ""]
        L += ["![early listening](figures/early_listening.png)", ""]

    if c:
        tr = c["treatment"].replace("_", "-")
        rows = []
        for k in ["renewed", "active_180d", "placebo_active_days_2m_before"]:
            if k in c:
                r = c[k]
                scale = (lambda v: f"{v:+.2f} days") if k.startswith("placebo") else pts
                rows.append({"outcome": k, "n": f"{r['n']:,}", "treated": f"{r['treated']:,}",
                             "naive": scale(r["naive"]), "outcome reg.": scale(r["outcome_regression"]),
                             "IPW": scale(r["ipw"]), "AIPW": scale(r["aipw"]),
                             "AIPW 95% CI": f"[{scale(r['aipw_ci'][0])}, {scale(r['aipw_ci'][1])}]"})
        b = c["balance"]
        L += [f"## 4. Causal effect of {tr} on retention", "",
              table(rows, ["outcome", "n", "treated", "naive", "outcome reg.", "IPW", "AIPW", "AIPW 95% CI"]), "",
              f"- Balance: largest standardized mean difference {b['max_abs_smd_raw']:.2f} before weighting, "
              f"{b['max_abs_smd_weighted']:.2f} after (under 0.1 is the usual bar).",
              f"- Overlap: {pct(c['renewed']['share_outside_05_95'])} of units have propensity outside 0.05-0.95; "
              f"AIPW on the trimmed sample is {pts(c['renewed']['aipw_trimmed_05_95'])}.",
              f"- Listening last month: treated {c['treated_vs_control_prev_active_days'][0]:.1f} days vs. control "
              f"{c['treated_vs_control_prev_active_days'][1]:.1f} days.",
              "- Placebo: the treatment cannot change listening that happened before it. An adjusted estimate near "
              "zero means the adjustment is working; a large one means hidden confounding.", "",
              "![causal](figures/causal_effect.png)", ""]

    # ---- draft resume bullets ----
    L += ["## Draft resume bullets (check every number against the tables above)", ""]
    if g:
        lifts = g["projection"]["lift_vs_base_pct"]
        top = max(lifts, key=lifts.get)
        L.append(f"- Built a Duolingo-style growth model for {d.get('users', 0):,} subscribers of a music streaming "
                 f"service, splitting monthly growth into new, retained, resurrected and churned users. It showed "
                 f"{top} grows the base most ({lifts[top]:+.1f}% in 12 months).")
    if s:
        ape = s["hazard"]["mean_abs_pct_error"]
        best_base = min(v for k, v in ape.items() if k != "hazard_model" and v == v)
        if ape["hazard_model"] < best_base:
            L.append(f"- Built a discrete-time hazard model on subscriber-months that forecast monthly churn with "
                     f"{ape['hazard_model']:.1f}% error vs. {best_base:.1f}% for the best simple baseline.")
        else:
            L.append(f"- (Hazard model did not beat the simple baselines: {ape['hazard_model']:.1f}% vs. "
                     f"{best_base:.1f}% error. Lead with the survival curves or AUC instead, or say so honestly.)")
    if c and "renewed" in c:
        r = c["renewed"]
        L.append(f"- Estimated the effect of {tr}s on renewal with cross-fitted AIPW: naive comparison "
                 f"{pts(r['naive'])}, adjusted {pts(r['aipw'])}, with balance and placebo checks.")
    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"wrote {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
