# Your rules-based fraud system is costing you £5.6M a year

That’s not a hypothetical. That’s what this project found by replacing a threshold-based rules engine with a cost-optimised LightGBM model on 590,540 real payment transactions from the IEEE-CIS dataset. The rules-based system blocked 15% of legitimate customers. The ML model cut that to 1.06% while saving an estimated £471,475 over 30 days.

This isn’t a Kaggle notebook that stops at AUC. It goes all the way: data loading, EDA, feature engineering, a time-based train/test split (not random — more on why that matters), baseline vs champion model comparison, cost-sensitive threshold optimisation, SHAP explainability for FCA compliance, and a 30-day deployment simulation with sensitivity analysis on the cost assumptions.

**Author:** Sherifdeen Oloye    
**GitHub:** [@datawitholoye](https://github.com/datawitholoye)

-----

## The Problem

Rules-based fraud systems fail in two directions at once: they miss fraud that doesn’t fit the rules, and they block real customers whose transactions look suspicious on paper. A £2,000 transaction from a new device is flagged whether it’s a fraudster or someone buying a laptop on their first day with a new card.

The question this project answers: can a machine learning model do better on both counts simultaneously, and what’s the actual £ cost difference over a month of live traffic?

-----

## Dataset

**IEEE-CIS Fraud Detection** (Vesta Corporation via Kaggle, 2019)

Two tables joined on `TransactionID`:

|Table                |Rows    |Description                                            |
|---------------------|--------|-------------------------------------------------------|
|train_transaction.csv|590,540 |Transaction amount, card info, time, 339 V-features    |
|train_identity.csv   |~144,000|Device, browser, email signals (subset of transactions)|

|Property            |Value           |
|--------------------|----------------|
|Total transactions  |590,540         |
|Legitimate (label 0)|569,877 (96.50%)|
|Fraud (label 1)     |20,663 (3.50%)  |
|Imbalance ratio     |~27:1           |
|Feature columns used|367             |

**Download from Kaggle:**  
<https://www.kaggle.com/c/ieee-fraud-detection/data>

> Both CSV files are too large for GitHub. Download `train_transaction.csv` and `train_identity.csv` from the link above and place them in `data/` before running the notebook.

-----

## Models and Results

Train/test split: time-based at the 80th percentile of `TransactionDT` (not random — see Key Findings below for why this matters).

|Model                         |ROC-AUC|PR-AUC|
|------------------------------|-------|------|
|Logistic Regression (baseline)|0.8253 |0.1761|
|LightGBM (champion)           |0.9049 |0.5137|

LightGBM lift over baseline: +9.6% ROC-AUC, +191.7% PR-AUC.

**At optimal threshold (0.83):**

|Metric                           |Value     |
|---------------------------------|----------|
|Fraud caught (recall)            |43.4%     |
|Precision                        |59.4%     |
|Legitimate customers blocked     |1.06%     |
|30-day cost saving vs rules-based|£471,475  |
|Annualised saving                |£5,657,700|

-----

## Key Findings

**Random train/test splits are wrong for financial data.** If you shuffle rows and take 80% for training, the model sees future transactions during training. Performance looks better than it is; in production it degrades fast. This notebook splits by time: the first 80% of `TransactionDT` is training data, the last 20% is the test set. That’s how the model will actually be used.

**Default threshold 0.5 costs £159,580 more per month than it needs to.** The model outputs a fraud probability, but the decision to block or approve still requires a threshold. At 0.5, the model over-blocks. By sweeping thresholds from 0.01 to 0.99 and computing business cost at each point (false negative = £85, false positive = £30), the optimal threshold landed at 0.83. That single change saved £159,580 against the default.

**The rules-based system blocked 14x more legitimate customers.** At 15% false positive rate vs 1.06%, the old system was generating enormous customer friction. The ML model’s precision of 59.4% means nearly 6 in 10 blocks are genuine fraud; under the rules system that ratio was far worse.

**SHAP makes the model explainable to a compliance officer.** Under GDPR Article 22 and FCA expectations, firms need to explain automated decisions. The notebook includes both global feature importance (SHAP summary plots) and individual transaction explanations — showing exactly which features pushed a specific transaction toward a fraud score of 0.9998.

**The saving holds under stress-tested cost assumptions.** The sensitivity analysis tested combinations of false negative costs from £50 to £150 and false positive costs from £10 to £100. The ML model saves money over the rules-based system in every scenario.

-----

## How to Run Locally

**Requirements:** Python 3.9+, pip, Git. Tested on macOS with Apple Silicon.

```bash
# 1. Clone the repo
git clone https://github.com/datawitholoye/fraud-detection-fintech.git
cd fraud-detection-fintech

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the data folder and add the Kaggle CSVs
mkdir data
# Place train_transaction.csv and train_identity.csv inside data/

# 5. Launch the notebook
jupyter notebook fraud_detection_project.ipynb
```

**Runtime note:** LightGBM training takes 2-5 minutes on ~470k rows. SHAP value computation uses a 2,000-transaction sample to keep it manageable. Total notebook runtime is roughly 15-20 minutes end to end on Apple Silicon.

-----

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
shap
jupyter
```

Full pinned versions in `requirements.txt`.

-----

