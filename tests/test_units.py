import math

import numpy as np
import pandas as pd
import pytest

from fraud.data import card_key, time_split
from fraud.evaluate import decision_summary, pick_threshold
from fraud.experiment import n_per_arm_proportions
from fraud.features import FeatureBuilder
from fraud.monitoring import psi
from fraud.rules import RulesEngine


def test_time_split_is_ordered_and_disjoint():
    df = pd.DataFrame({"TransactionDT": np.arange(1000), "isFraud": 0})
    tr, va, te = time_split(df, 0.7, 0.8)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.TransactionDT.max() < va.TransactionDT.min()
    assert va.TransactionDT.max() < te.TransactionDT.min()


def test_time_split_rejects_bad_quantiles():
    with pytest.raises(ValueError):
        time_split(pd.DataFrame({"TransactionDT": [1, 2]}), 0.8, 0.7)


def test_missed_fraud_costs_its_amount_plus_fee():
    y = [1, 1, 0, 0]
    amount = [100.0, 500.0, 50.0, 20.0]
    block = [True, False, True, False]  # catch the 100, miss the 500, block one genuine
    s = decision_summary(y, amount, block, fp_cost=30, fn_fee=15)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (1, 1, 1, 1)
    assert s["cost_missed_fraud"] == 515
    assert s["cost_blocked_customers"] == 30
    assert s["total_cost"] == 545
    assert s["value_detection_rate"] == pytest.approx(100 / 600)


def test_pick_threshold_finds_cheapest():
    y = np.array([1, 1, 0, 0, 0])
    amount = np.array([1000, 1000, 10, 10, 10.0])
    scores = np.array([0.9, 0.6, 0.5, 0.2, 0.1])
    t, table = pick_threshold(y, amount, scores, [0.3, 0.55, 0.95], fp_cost=30, fn_fee=0)
    assert t == 0.55  # catches both frauds without blocking any genuine customer
    assert table.total_cost.min() == 0


def test_rules_engine_conditions():
    rules = [
        {"name": "big", "conditions": [{"col": "TransactionAmt", "op": "gt", "value": 1000}]},
        {"name": "mismatch", "conditions": [
            {"col": "P_emaildomain", "op": "notnull"},
            {"col": "P_emaildomain", "op": "ne_col", "value": "R_emaildomain"}]},
        {"name": "missing_col", "conditions": [{"col": "nope", "op": "gt", "value": 0}]},
        {"name": "missing_is_null", "conditions": [{"col": "addr1", "op": "isnull"}]},
    ]
    df = pd.DataFrame({
        "TransactionAmt": [2000, 10, 10],
        "P_emaildomain": ["a.com", "a.com", None],
        "R_emaildomain": ["a.com", "b.com", "b.com"],
    })
    hits = RulesEngine(rules).evaluate(df)
    assert hits["big"].tolist() == [True, False, False]
    assert hits["mismatch"].tolist() == [False, True, False]
    assert not hits["missing_col"].any()
    assert hits["missing_is_null"].all()  # field not supplied = unknown address


def test_feature_builder_learns_from_train_only():
    train = pd.DataFrame({"TransactionDT": [1, 2, 3], "TransactionAmt": [10.0, 20.0, 30.0],
                          "card4": ["visa", "visa", "mastercard"], "mostly_missing": [np.nan, np.nan, 1.0],
                          "isFraud": [0, 1, 0]})
    fb = FeatureBuilder(drop_missing_above=0.5).fit(train)
    assert "mostly_missing" not in fb.feature_names
    assert "isFraud" not in fb.feature_names
    test = pd.DataFrame({"TransactionDT": [4], "TransactionAmt": [5.0], "card4": ["amex"]})
    X = fb.transform(test)
    assert list(X.columns) == fb.feature_names
    assert np.isnan(X["card4"].iloc[0])  # unseen category -> unknown, not a new code


def test_card_key_handles_missing():
    df = pd.DataFrame({"card1": [1, 1], "card2": [np.nan, 2.0], "card3": [3, 3], "card5": [5, 5], "addr1": [9, 9]})
    k = card_key(df)
    assert k.iloc[0] != k.iloc[1]


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) == pytest.approx(0, abs=1e-9)
    assert psi(a, a + 2) > 0.25


def test_sample_size_matches_textbook():
    # 10% vs 12%, alpha .05 two-sided, power .8 -> ~3,841 per arm
    assert abs(n_per_arm_proportions(0.10, 0.12) - 3841) < 15
    assert math.isinf(n_per_arm_proportions(0.1, 0.1))
