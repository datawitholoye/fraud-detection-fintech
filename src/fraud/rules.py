"""A small, config-driven rules engine: the baseline the model has to beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_derived


def _cond(df: pd.DataFrame, col: str, op: str, value=None) -> pd.Series:
    # A field that wasn't supplied is unknown: it never satisfies a comparison,
    # but it does count as null (e.g. "no billing address").
    s = df[col] if col in df else pd.Series(np.nan, index=df.index, dtype=object)
    num = lambda x: pd.to_numeric(x, errors="coerce")  # noqa: E731
    if op == "isnull":
        return s.isna()
    if op == "notnull":
        return s.notna()
    if op == "in":
        return s.isin(value)
    if op == "eq":
        return s == value
    if op == "ne_col":
        other = df[value] if value in df else pd.Series(np.nan, index=df.index)
        return s.notna() & other.notna() & (s != other)
    ops = {"gt": np.greater, "ge": np.greater_equal, "lt": np.less, "le": np.less_equal}
    if op in ops:
        return pd.Series(ops[op](num(s), value), index=df.index).fillna(False).astype(bool)
    raise ValueError(f"Unknown operator '{op}'")


class RulesEngine:
    def __init__(self, rules: list[dict]):
        if not rules:
            raise ValueError("RulesEngine needs at least one rule")
        self.rules = rules
        self.names = [r["name"] for r in rules]

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """One boolean column per rule: did it fire on this transaction?"""
        df = add_derived(df)
        hits = {}
        for r in self.rules:
            fired = pd.Series(True, index=df.index)
            for c in r["conditions"]:
                fired &= _cond(df, c["col"], c["op"], c.get("value"))
            hits[r["name"]] = fired.astype(bool)
        return pd.DataFrame(hits, index=df.index)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Number of rules fired. Block if score >= k (k chosen on validation)."""
        return self.evaluate(df).sum(axis=1).to_numpy()

    def rule_report(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Hit rate and precision of each rule: how an analyst would audit a rule set."""
        hits = self.evaluate(df)
        y = np.asarray(y)
        rows = []
        for r in self.rules:
            h = hits[r["name"]].to_numpy()
            rows.append({
                "rule": r["name"],
                "description": r.get("description", ""),
                "hit_rate": h.mean(),
                "precision": y[h].mean() if h.any() else np.nan,
                "fraud_caught_share": h[y == 1].mean() if (y == 1).any() else np.nan,
            })
        return pd.DataFrame(rows)
