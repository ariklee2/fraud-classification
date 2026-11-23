# Credit Card Fraud Detection

Predicting whether a card transaction is fraudulent, on 100,000 transactions
from 2026 (about 2% fraud), and measuring what the model would mean for a fraud
review team.

Everything is in [`fraud.ipynb`](fraud.ipynb).

## Approach

- **Date-based split.** Train on Jan–Sep, test once on Oct–Dec, the way a
  fraud model is used in practice. Inside the training months, models are fit
  on Jan–Jul and the decision threshold and hyperparameters are chosen on
  Aug–Sep.
- **No test-set peeking in the EDA.** Every EDA finding that becomes a feature
  cutoff (night hours, 100+ km from home, 3+ transactions in the past hour,
  risky merchant countries) is computed on Jan–Sep only.
- **Feature engineering.** 11 features from the EDA: binary flags, an
  "impossible travel" speed (km per day since the last transaction), the
  amount's z-score against the card's 30-day average, and a combined
  suspicion score whose weights are each flag's log lift in the training
  months.
- **Models.** LightGBM, random forest and logistic regression, each with:
  default settings, an ablation without the engineered features, SMOTENC
  oversampling, and a random hyperparameter search scored by validation
  PR-AUC.
- **Metrics.** F1 at the validation-tuned threshold, plus **PR-AUC**, which is
  the more honest metric at 2% fraud (a random model scores 0.022). The final
  model is picked by validation PR-AUC, never by test score.
- **Explainability and impact.** SHAP values for the final model, and fraud
  dollars caught vs. false alarms at different alert budgets.

## Results (test set: Oct–Dec, 25,337 transactions, 550 fraud)

| Setup | Model | F1 | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- |
| Default | LightGBM | 0.206 | 0.135 | 0.836 |
| Default | Random forest | 0.217 | 0.141 | 0.829 |
| Default | Logistic regression | 0.233 | 0.174 | 0.853 |
| SMOTENC | LightGBM | 0.234 | 0.152 | 0.841 |
| Tuned | LightGBM | 0.205 | 0.163 | 0.853 |
| **Tuned (final)** | **Logistic regression** | **0.239** | **0.174** | **0.853** |

**Business impact of the final model:**

| Transactions flagged | False alarms per 1,000 | Precision | Fraud cases caught | Fraud $ caught |
| --- | --- | --- | --- | --- |
| Top 1% | 6.9 | 31% | 79 / 550 | 11.5% |
| Top 2% (≈ F1 threshold) | 15.0 | 25% | 127 / 550 | 31.2% |
| Top 5% | 41.7 | 17% | 210 / 550 | 50.3% |
| Top 10% | 88.2 | 12% | 300 / 550 | 67.4% |

## Key findings

- **Distance and foreign activity drive fraud.** Transactions 100+ km from
  home are about 9x as likely to be fraud as nearby ones, and
  `international_transaction` is the largest SHAP driver.
- **Night hours (1–4 am), new accounts (< 90 days) and bursts of 3+
  transactions in an hour** each roughly double to quadruple the fraud rate.
- **Engineered features help the linear model** (PR-AUC 0.138 → 0.174) but not
  the tree models, which learn those cutoffs on their own.
- **Oversampling isn't a reliable fix.** SMOTENC helped LightGBM but hurt
  logistic regression; tuning the threshold already handles most of the
  imbalance.
- The model **catches the larger frauds first**: flagging 2% of transactions
  catches 23% of fraud cases but 31% of fraud dollars.

## Limitations

- F1 around 0.24 means most fraud still gets through at the chosen threshold;
  each feature is only weakly related to fraud on its own.
- The dataset is synthetic.
- Next steps: features comparing each transaction with that card's own
  history, a threshold based on real review costs, and probability
  calibration.

## Running it

Python 3 with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`,
`lightgbm`, `imbalanced-learn`, `shap`, `scipy` and `kagglehub`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm imbalanced-learn shap scipy kagglehub
```

The first cell downloads the dataset from Kaggle
(`mohankrishnathalla/credit-card-fraud-detection-dataset-2026`) with
`kagglehub`. A full run takes about 3 minutes on a laptop.
