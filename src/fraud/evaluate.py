"""Business-cost evaluation: every decision is priced, missed fraud at its real amount."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def decision_summary(y, amount, block, fp_cost: float, fn_fee: float) -> dict:
    """Price a set of block/approve decisions.

    Missed fraud costs its transaction amount plus a fixed chargeback/ops fee.
    A blocked genuine transaction costs a fixed friction amount.
    """
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype=float)
    block = np.asarray(block).astype(bool)

    tp, fn = (y & block).sum(), (y & ~block).sum()
    fp, tn = (~y & block).sum(), (~y & ~block).sum()
    fraud_value = amount[y].sum()
    caught_value = amount[y & block].sum()
    cost_fn = (amount[y & ~block] + fn_fee).sum()
    cost_fp = fp * fp_cost
    n = len(y)
    return {
        "transactions": int(n),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": tp / max(tp + fn, 1),
        "value_detection_rate": caught_value / fraud_value if fraud_value else 0.0,
        "precision": tp / max(tp + fp, 1),
        "legit_block_rate": fp / max(fp + tn, 1),
        "alerts_per_1k": 1000 * (tp + fp) / max(n, 1),
        "fraud_value_total": float(fraud_value),
        "fraud_value_caught": float(caught_value),
        "cost_missed_fraud": float(cost_fn),
        "cost_blocked_customers": float(cost_fp),
        "total_cost": float(cost_fn + cost_fp),
    }


def sweep(y, amount, scores, thresholds, fp_cost: float, fn_fee: float) -> pd.DataFrame:
    """Total cost at each threshold (block if score >= threshold)."""
    scores = np.asarray(scores)
    rows = []
    for t in thresholds:
        s = decision_summary(y, amount, scores >= t, fp_cost, fn_fee)
        rows.append({"threshold": float(t), **s})
    return pd.DataFrame(rows)


def pick_threshold(y, amount, scores, thresholds, fp_cost: float, fn_fee: float) -> tuple[float, pd.DataFrame]:
    """Cheapest threshold. Call this on VALIDATION data only."""
    table = sweep(y, amount, scores, thresholds, fp_cost, fn_fee)
    best = table.loc[table["total_cost"].idxmin(), "threshold"]
    return float(best), table


def ranking_metrics(y, scores) -> dict:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores))}


def scale_to_days(summary: dict, period_days: float, days: float = 30.0) -> dict:
    """Normalise count/£ figures from a test period to a fixed window (e.g. 30 days)."""
    f = days / period_days
    keys = ["tp", "fp", "fn", "tn", "transactions", "fraud_value_total", "fraud_value_caught",
            "cost_missed_fraud", "cost_blocked_customers", "total_cost"]
    return {k: (v * f if k in keys else v) for k, v in summary.items()}
