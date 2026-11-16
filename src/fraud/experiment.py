"""Sizing a champion (rules) vs challenger (model) A/B test before going live."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def n_per_arm_means(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-sample comparison of means (two-sided)."""
    if delta <= 0:
        return math.inf
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * (z * sd / delta) ** 2)


def n_per_arm_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-proportion z-test (two-sided)."""
    if p1 == p2:
        return math.inf
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def loss_per_transaction(y, amount, block, fn_fee: float) -> np.ndarray:
    """£ lost on each transaction under a decision policy (0 unless fraud got through)."""
    y, block = np.asarray(y).astype(bool), np.asarray(block).astype(bool)
    return np.where(y & ~block, np.asarray(amount, float) + fn_fee, 0.0)


def plan_test(y, amount, champion_block, fn_fee: float, txns_per_day: float,
              alpha=0.05, power=0.8, relative_mde=0.10, share_per_arm=0.5) -> dict:
    """Sample size and duration to detect a `relative_mde` change in
    (a) fraud £ lost per transaction  — primary metric
    (b) genuine-customer block rate   — guardrail
    using the champion's behaviour on historical data as the baseline."""
    loss = loss_per_transaction(y, amount, champion_block, fn_fee)
    mean, sd = loss.mean(), loss.std(ddof=1)
    n_loss = n_per_arm_means(sd, relative_mde * mean, alpha, power)

    y_b, b = np.asarray(y).astype(bool), np.asarray(champion_block).astype(bool)
    p_block = b[~y_b].mean()
    n_block = n_per_arm_proportions(p_block, p_block * (1 - relative_mde), alpha, power)

    per_arm_per_day = txns_per_day * share_per_arm
    return {
        "baseline_loss_per_txn": float(mean),
        "baseline_legit_block_rate": float(p_block),
        "relative_mde": relative_mde,
        "n_per_arm_loss": int(n_loss),
        "n_per_arm_block_rate": int(n_block),
        "days_needed_loss": n_loss / per_arm_per_day,
        "days_needed_block_rate": n_block / per_arm_per_day,
    }
