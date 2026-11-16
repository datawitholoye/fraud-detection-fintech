On the held-out test month, the cost-tuned LightGBM model caught **57.2% of fraud by value** (rules: 20.9%) while blocking **3.6% of genuine transactions** (rules: 2.6%). Net, it cost **£158,916 less per 30 days** than a hand-written rules engine whose block threshold was tuned on the same validation month.

**Test month, scaled to 30 days** (thresholds chosen on validation, applied once to test):

| Policy | Fraud caught (count) | Fraud caught (£ value) | Precision | Genuine txns blocked | Alerts / 1k txns | Total cost (30 days) |
|---|---|---|---|---|---|---|
| Rules engine (k = 2 rules) | 4.9% | 20.9% | 6.3% | 2.6% | 27.0 | £451,417 |
| LightGBM @ 0.50 (default) | 48.6% | 44.1% | 53.5% | 1.5% | 31.3 | £303,648 |
| LightGBM @ 0.31 (cost-tuned on validation) | 58.3% | 57.2% | 36.8% | 3.6% | 54.5 | £292,502 |

**Ranking quality on the test month:**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.826 | 0.172 |
| LightGBM (all test transactions) | 0.894 | 0.525 |
| LightGBM (cards seen in training) | 0.893 | 0.523 |
| LightGBM (cards **never** seen in training, 5.8% of test) | 0.903 | 0.569 |

**Does the conclusion survive different cost assumptions?** Saving vs rules per 30 days (both systems re-tuned on validation for each scenario):

| Chargeback fee \ Cost per blocked customer | £10 | £30 | £60 | £100 |
|---|---|---|---|---|
| £0 | £168,419 | £135,630 | £119,976 | £92,238 |
| £15 | £187,740 | £158,916 | £140,811 | £102,603 |
| £30 | £208,973 | £182,201 | £152,713 | £123,020 |

The model is cheaper in **12 of 12** cost scenarios.

**Drift check (PSI):** model scores validation → test = 0.003 (stable); transaction amount train → test = 0.006 (stable).

**A/B test sizing** (rules as champion, model as challenger, detect a 10.0% relative change, α = 0.05, power = 0.8, 50/50 split): **126,396** transactions per arm (≈ 90 days) for fraud £ loss; **55,577** per arm (≈ 39 days) for the genuine-block-rate guardrail. See [docs/rollout_plan.md](docs/rollout_plan.md).
