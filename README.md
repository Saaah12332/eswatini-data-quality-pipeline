# Bank early-warning risk score

A one-quarter-ahead distress score for 20 institutions over 12 quarters, built only from capital, liquidity and asset-quality ratios. Two models — logistic regression and a random forest — are compared in a time-based back-test. The logistic score is the one that goes into the one-page supervisory summary.

Python, scikit-learn.

## What this project is doing

A supervisor already receives a quarterly return. The question is whether those ratios say anything about *next* quarter, in time to look at the institution before it is already in distress.

The score is the model's estimated probability that the institution is in distress next quarter. It is a review queue, not a capital decision. A high score means "the ratios look like institutions that were in distress a quarter later." A low score is not a clean bill of health.

## The data

`data/bank_panel.csv` is a panel of 20 institutions and 12 quarters, 2022Q1 through 2024Q4.

Real institution-level supervisory returns cannot be published. This file is a synthetic panel with the same shape as the exercise, so the code, the back-test and the one-page summary can be re-run by anyone. The names (Aldermere Bank, Redwillow Trust, and so on) are fictional.

Each institution has a latent condition that persists from quarter to quarter. The six reported ratios are noisy readings of that condition:

| Ratio | What it measures |
|---|---|
| `cet1_ratio` | Common equity tier 1, as a percent of risk-weighted assets |
| `leverage_ratio` | Capital as a percent of total exposure |
| `lcr` | Liquidity coverage ratio, percent |
| `loan_to_deposit` | Loans as a percent of deposits |
| `npl_ratio` | Non-performing loans as a percent of gross loans |
| `provision_coverage` | Provisions as a percent of non-performing loans |

Distress is not a hard cut on those six numbers. An institution is in distress in a quarter when its condition, plus a shock the ratios do not observe, crosses a breach line. Because the condition persists, this quarter's ratios still lead next quarter's breach. Because of the shock, they do not determine it. About 17% of institution-quarters are in distress.

The column the models are allowed to learn from is `distress_next_quarter`. The last quarter (2024Q4) has ratios but no label, because the next quarter has not happened. That is the quarter the watchlist is scoring.

## What the script does, in order

`analysis.py` does the whole project in one run.

1. **Build the panel** with a fixed random seed, and write it to `data/bank_panel.csv`. The seed is in the script, so the file does not change between machines.
2. **Walk forward through time.** The first scored quarter is 2022Q4. For that quarter the model is trained only on 2022Q1–2022Q3. It is then refit for every later quarter, always using only earlier quarters. It never sees the quarter it is scoring, and it never sees the future. There are no random train/test splits. A random split would let 2024 teach the model about 2022, which is not a forecast.
3. **Fit both models** on each origin. Logistic regression is fit on standardised ratios, with class weights, so a rare distress flag is not ignored. The random forest is a modest forest (depth limited, leaves not tiny) for the same reason: 20 institutions is a small panel, and a deep forest would memorise it.
4. **Choose the alert threshold on 2022Q4–2023Q1 only.** The target is a hit rate close to 85% of distress cases, without flagging more than 40% of institutions. The cutoff that gets closest to that target is frozen.
5. **Score the holdout, 2023Q2–2024Q3, with that frozen cutoff.** This window is the result quoted below. The threshold was not adjusted to it.
6. **Score 2024Q4** with a model trained on all labeled history, and write the watchlist.
7. **Write** `reports/metrics.json`, `reports/one_page_supervisory_summary.md`, and the figures under `reports/figures/`.

## Result

On the holdout the logistic regression flagged **18 of 24 distress cases (75%)** one quarter ahead. AUC was **0.95**. Of the names sent for review, 72% were in distress the next quarter. The model alerted on 21% of institution-quarters.

| Model | Validation AUC | Holdout AUC | Hit rate | Precision | Alerts |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.833 | 0.949 | 18/24 (75%) | 72% | 21% |
| Random forest | 0.799 | 0.930 | 17/24 (71%) | 61% | 23% |

The logistic model is the selected score because it had the higher AUC on the validation window. The holdout was not used to pick the winner.

The 85% figure is the *target* the threshold was aimed at on the validation window. The holdout did not reach it: 6 of the 24 distress cases were missed. That is the number this repository supports. The misses are visible by quarter in `reports/figures/hits_by_quarter.png`.

![Distress cases caught one quarter ahead](reports/figures/hits_by_quarter.png)

![ROC on the time-based holdout](reports/figures/roc_backtest.png)

### How to read the logistic score

Ratios are standardised before the logistic regression, so the coefficients can be compared with each other. A positive coefficient raises the probability of distress.

| Ratio | Coefficient |
|---|---:|
| Leverage ratio | −0.98 |
| NPL ratio | +0.88 |
| Provision coverage | −0.58 |
| Loan-to-deposit | +0.34 |
| CET1 ratio | −0.05 |
| LCR | +0.05 |

Leverage, asset quality and coverage point the way a supervisor would expect: thinner leverage and higher non-performing loans raise the score; higher coverage lowers it. CET1 and the liquidity coverage ratio barely move the linear score. They travel with leverage and non-performing loans, so once those two are in the model the linear equation has little left to assign to capital and liquidity. That is collinearity, not a finding that liquidity does not matter. The forest still uses them: LCR is about 14% of its importance and CET1 about 14% (`reports/figures/feature_importance.png`).

![Which ratios the forest uses](reports/figures/feature_importance.png)

## Watchlist for the quarter after 2024Q4

These four institutions are above the frozen threshold on the latest ratios. Scores are probabilities from the logistic model.

| Institution | Score | CET1 % | LCR | NPL % | Leverage % |
|---|---:|---:|---:|---:|---:|
| Redwillow Trust | 0.97 | 11.1 | 98 | 10.2 | 4.2 |
| Southmere Building Society | 0.97 | 9.8 | 116 | 11.5 | 4.3 |
| Northglass Bank | 0.94 | 10.0 | 106 | 10.2 | 3.8 |
| Cedar & Holm | 0.86 | 10.5 | 112 | 8.7 | 4.6 |

The full one-page note for a supervisor is `reports/one_page_supervisory_summary.md`. Every institution, flagged or not, is in `reports/metrics.json` under `latest`.

## What the score is not

It does not use market prices, governance, or anything a supervisor learned in a meeting. It will miss a distress event that comes from a shock the ratios do not contain — that shock is built into the data on purpose, and it is why the hit rate is not 100% even though the panel is synthetic. On real returns the ratios are noisier than they are here, and a 0.95 AUC should not be expected to carry over.

## Run it

```bash
pip install -r requirements.txt
python analysis.py
```

That regenerates the panel, the back-test, the figures, the metrics file and the one-page summary. Nothing in `reports/` was typed in by hand.
