# One-page supervisory summary

**As-of quarter:** 2024Q4  
**Score:** probability of distress next quarter, from logistic regression.

## Question

Which of the 20 institutions look likely to be in distress next quarter, using only the capital, liquidity and asset-quality ratios already on the return?

## Sample

A balanced panel of 20 institutions and 12 quarters (2022Q1–2024Q4). Six ratios: CET1, leverage, liquidity coverage (LCR), loan-to-deposit, non-performing loans (NPL) and provision coverage. The label is distress in the *following* quarter, so every score is an early warning, not a description of the quarter just reported.

## How the back-test works

The models are refit every quarter. Training uses only earlier quarters — never the quarter being scored and never the future. The alert threshold is the cutoff whose hit rate on the first 2 out-of-sample quarters was closest to 85%, without alerting on more than 40% of institutions. It was then frozen. The result below is the later holdout (2023Q2–2024Q3), which the threshold did not see.

## Result

**18 of 24 distress cases (75%) were flagged one quarter ahead** by the logistic regression (AUC 0.95). Of the names sent for review, 72% were subsequently in distress. The alert rate was 21% of institution-quarters.

Logistic regression is reported beside the forest so a linear, inspectable score is always available. The forest is used for the watchlist when it wins the validation AUC; otherwise the logistic score is used.

## Watchlist going into the next quarter

Scores below use the latest filed ratios (2024Q4) and the model trained only on history before that quarter. They are a queue for review, not a capital decision.

| Institution | Score | CET1 % | LCR % | NPL % | Leverage % |
|---|---:|---:|---:|---:|---:|
| Redwillow Trust | 0.97 | 11.1 | 98 | 10.2 | 4.2 |
| Southmere Building Society | 0.97 | 9.8 | 116 | 11.5 | 4.3 |
| Northglass Bank | 0.94 | 10.0 | 106 | 10.2 | 3.8 |
| Cedar & Holm | 0.86 | 10.5 | 112 | 8.7 | 4.6 |

## What this score is not

It does not use market prices, governance findings or qualitative supervision. A high score means the ratios resemble institutions that were in distress a quarter later. A low score is not a clean bill of health.

## Caveat on the published data

Institution-level supervisory returns cannot be published. The panel in `data/bank_panel.csv` is a synthetic panel with the same design: 20 institutions, 12 quarters, the six ratios, and a next-quarter distress event that is correlated with those ratios but not a hard threshold on them. Every number in this note is computed by `python analysis.py`.
