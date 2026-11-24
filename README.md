# Subscriber Growth Lab

How does a streaming subscription service grow, who leaves, and do discounts actually keep people?

This project answers those questions on **KKBox**, a music streaming subscription service, using
the public data from the WSDM 2018 churn challenge: 21M+ payment and cancellation records and
~30 GB of daily listening logs from 2015 to March 2017. These are the same questions a subscription
product like YouTube TV, NFL Sunday Ticket or Spotify Premium faces.

| Question | Method | Output |
|---|---|---|
| Where does subscriber growth come from, and which lever matters most? | Duolingo-style user-state growth accounting (new, retained, resurrected, churned, dormant) + a 12-month projection | `growth_accounting.png` |
| Are subscribers who pay but stop listening at risk? | Next-month churn by this month's listening | table in `RESULTS.md` |
| When do subscribers leave? | Kaplan-Meier survival curves, with still-active subscribers treated as censored | `survival_curves.png` |
| Can we forecast monthly churn better than a trailing average? | Discrete-time hazard model (gradient boosting on subscriber-months), backtested on the last 3 observed months | `churn_backtest.png` |
| Does early listening predict who stays? | Retention at day 180 by days listened in the first full month, best threshold by Youden's J | `early_listening.png` |
| Do discounts keep subscribers or attract people who would leave anyway? | Propensity score methods: IPW and cross-fitted AIPW (doubly robust), with balance, overlap and placebo checks | `causal_effect.png` |

Everything runs on a laptop with DuckDB. There's no Spark or cloud.

---

## 1. Set up

```bash
git clone https://github.com/<you>/subscriber-growth-lab.git
cd subscriber-growth-lab
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## 2. Check that it works (2 minutes, no download)

```bash
python run_all.py --synthetic
```

This generates fake data in the exact KKBox format with known answers planted in it, then runs
everything. The end-to-end test checks that the causal estimate recovers the planted effect:

```bash
python tests/test_pipeline.py
# OK  naive -0.063  AIPW +0.069 (truth +0.050, se 0.014)  placebo naive -4.21 -> AIPW -0.21 days
```

## 3. Get the real data

1. Sign in to Kaggle and accept the rules at
   https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules
2. Get an API token (Kaggle → Settings → API → Create New Token) and save it as `~/.kaggle/kaggle.json`.
3. Download and extract:

```bash
pip install kaggle
kaggle competitions download -c kkbox-churn-prediction-challenge -p data/raw
cd data/raw && unzip -o '*.zip'
7z x transactions.csv.7z; 7z x transactions_v2.csv.7z; 7z x members_v3.csv.7z   # brew/conda install p7zip
cd ../..
```

You need these files. They can be in any subfolder of `data/raw`:

- `transactions.csv`, `transactions_v2.csv`
- `user_logs.csv` (about 30 GB unzipped), `user_logs_v2.csv`
- `members_v3.csv`

**Low on disk space?** Don't extract `user_logs.csv.7z`. If the pipeline finds a `.7z` with no matching
CSV, it streams it through 7-Zip straight into DuckDB, so the ~30 GB CSV never lands on disk. Then you
only need about 15 GB free: the archives (~9 GB) plus the small CSVs and DuckDB's working files.

## 4. Run it

Start with a 20% sample of users. It builds in a few minutes and gives you a feel for the results:

```bash
python run_all.py --data-dir data/raw --sample-pct 20 --memory-limit 6GB
```

Then the full run:

```bash
python run_all.py --data-dir data/raw --memory-limit 8GB
```

| Laptop RAM | Suggested command |
|---|---|
| 8 GB | `--sample-pct 25 --memory-limit 4GB` |
| 16 GB | `--memory-limit 8GB` (full data) |
| 32 GB+ | `--memory-limit 20GB` |

The slow step is reading the 30 GB user logs once, which takes roughly 10–30 minutes depending on your disk.
The tables are saved in `work/kkbox.duckdb`, so after one build you can rerun the analyses alone:

```bash
python run_all.py --skip-build
python src/causal.py --treatment auto_renew      # a second causal question
```

Results: **`results/RESULTS.md`** (tables, charts and draft resume bullets) plus `results/figures/`.

## 5. Definitions

- **Churn** follows the KKBox rule: no new paid subscription within 30 days after the membership
  expires. A cancellation row (`is_cancel = 1`) ends auto-renew and cuts coverage short.
- **Spell**: one stretch of paid coverage, where gaps of 30 days or less are merged.
- **Censoring**: the data ends on 2017-03-31, so any subscription that expires within 30 days of that
  date has an unknown outcome. Those spells are censored, not counted as lost or retained. The hazard
  model only trains and tests on months whose outcome is fully observed (through Feb 2017).
- **Monthly user states** count a user as active in a month if a spell covers any day of it.
  Transactions start in Jan 2015, so growth metrics are reported from Jan 2016. Before that, the "new"
  subscribers are partly people who were already subscribed.
- **Causal unit**: one randomly chosen paid transaction per user. Treatment: paid below list price.
  Covariates come from before the payment: plan, price, auto-renew, payment method, tenure, prior
  cancellations, last month's listening and demographics.
- **Placebo**: the effect of a discount on listening *two months before* the payment. It has to be zero.
  If the adjusted estimate isn't close to zero, there is confounding the covariates don't capture.

## 6. Caveats to know before an interview

- The discount analysis is observational. AIPW removes bias from what we measured, not from what we
  didn't measure (for example, a user emailing support to complain). The placebo test and balance
  table show how much to trust it.
- Early listening vs. retention is a correlation. It suggests where to look, for example an onboarding
  test to get new subscribers listening, but it doesn't prove that more listening causes retention.
- The hazard model uses `days_left` (when the paid period ends). That is known at the start of each
  month, so it isn't leakage. Cancellations only count if they happened before that month started.
- KKBox prices are in New Taiwan dollars.

## 7. Project layout

```
run_all.py                     one command for everything
src/common.py                  paths, dates, metrics file
src/build.py                   raw CSVs -> DuckDB tables (txn, pay, spells, sub_months, members, logs_month)
src/growth.py                  user-state growth accounting, listening risk, lever projection
src/survival.py                Kaplan-Meier, discrete-time hazard model, churn backtest
src/engagement.py              early listening vs. 6-month retention
src/causal.py                  discount effect: naive, outcome regression, IPW, AIPW, balance, placebo
src/report.py                  results/metrics.json -> results/RESULTS.md
scripts/make_synthetic_data.py fake KKBox-format data with a planted effect
tests/test_pipeline.py         end-to-end check that the estimators recover the planted effect
```

## Data source

WSDM - KKBox's Churn Prediction Challenge (Kaggle, 2017). Data provided by KKBOX for research use under
the competition rules. The data is not included in this repository.
