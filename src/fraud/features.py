"""Feature engineering that is fit on training data only and reused at serving time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AMOUNT, TARGET, TIME_COL

NON_FEATURES = {"TransactionID", TIME_COL, TARGET}
IDENTITY_FLAG_COLS = ["id_01", "id_02", "id_03", "id_04", "id_05", "DeviceType", "DeviceInfo"]
ENGINEERED = ["log_amount", "amount_cents", "hour_of_day", "day_of_week", "email_match"] + [
    f"{c}_missing" for c in IDENTITY_FLAG_COLS
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns derived row-by-row (no learned state), shared with the rules engine."""
    out = df.copy()
    if AMOUNT in out:
        amt = pd.to_numeric(out[AMOUNT], errors="coerce")
        out["log_amount"] = np.log1p(amt)
        out["amount_cents"] = (amt - np.floor(amt)).round(3)
    if TIME_COL in out:
        t = pd.to_numeric(out[TIME_COL], errors="coerce")
        out["hour_of_day"] = (t % 86400) // 3600
        out["day_of_week"] = (t // 86400) % 7
    if {"P_emaildomain", "R_emaildomain"} <= set(out.columns):
        both = out["P_emaildomain"].notna() & out["R_emaildomain"].notna()
        out["email_match"] = np.where(both, (out["P_emaildomain"] == out["R_emaildomain"]).astype(float), np.nan)
    for c in IDENTITY_FLAG_COLS:
        out[f"{c}_missing"] = out[c].isna().astype(np.int8) if c in out else 1
    return out


class FeatureBuilder:
    """fit() learns which columns to keep and the category vocabularies from TRAIN
    only; transform() applies exactly that to any later data (validation, test, API)."""

    def __init__(self, drop_missing_above: float = 0.8):
        self.drop_missing_above = drop_missing_above
        self.feature_names: list[str] = []
        self.categories: dict[str, list[str]] = {}

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        df = add_derived(train)
        candidates = [c for c in df.columns if c not in NON_FEATURES]
        missing = df[candidates].isna().mean()
        keep = [c for c in candidates if missing[c] <= self.drop_missing_above or c in ENGINEERED]
        self.categories = {
            c: sorted(df[c].dropna().astype(str).unique().tolist())
            for c in keep if not pd.api.types.is_numeric_dtype(df[c])
        }
        self.feature_names = keep
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            raise RuntimeError("FeatureBuilder must be fit before transform")
        df = add_derived(df)
        cols = {}
        for c in self.feature_names:
            s = df[c] if c in df else pd.Series(np.nan, index=df.index)
            if c in self.categories:
                s = s.astype("string")
                s = s.where(s.isin(self.categories[c]))  # unseen categories -> unknown
                codes = pd.Categorical(s, categories=self.categories[c]).codes
                cols[c] = np.where(codes < 0, np.nan, codes).astype(np.float32)  # unseen/missing -> NaN
            else:
                cols[c] = pd.to_numeric(s, errors="coerce").astype(np.float32)
        return pd.DataFrame(cols, index=df.index)[self.feature_names]
