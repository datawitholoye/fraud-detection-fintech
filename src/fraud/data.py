"""Loading the IEEE-CIS tables and splitting them by time."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIME_COL = "TransactionDT"
TARGET = "isFraud"
AMOUNT = "TransactionAmt"
CARD_KEY_COLS = ["card1", "card2", "card3", "card5", "addr1"]


def load_raw(raw_dir: str | Path, transaction_file: str, identity_file: str,
             nrows: int | None = None) -> pd.DataFrame:
    """Left-join transactions to identity so every transaction is kept."""
    raw_dir = Path(raw_dir)
    tx_path, id_path = raw_dir / transaction_file, raw_dir / identity_file
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Download the IEEE-CIS data from "
            "https://www.kaggle.com/c/ieee-fraud-detection/data and put "
            f"{transaction_file} and {identity_file} in {raw_dir}/"
        )
    tx = pd.read_csv(tx_path, nrows=nrows)
    ident = pd.read_csv(id_path) if id_path.exists() else pd.DataFrame({"TransactionID": []})
    df = tx.merge(ident, on="TransactionID", how="left")
    if len(df) != len(tx):
        raise ValueError("Row count changed after merge: duplicate TransactionIDs in identity table")
    return _downcast(df)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 roughly halves memory on the 434-column table."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def time_split(df: pd.DataFrame, train_end_q: float, val_end_q: float
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on the timeline: train < validation < test, no overlap, no shuffling."""
    if not 0 < train_end_q < val_end_q < 1:
        raise ValueError("Need 0 < train_end_quantile < val_end_quantile < 1")
    t1, t2 = df[TIME_COL].quantile([train_end_q, val_end_q])
    train = df[df[TIME_COL] <= t1].copy()
    val = df[(df[TIME_COL] > t1) & (df[TIME_COL] <= t2)].copy()
    test = df[df[TIME_COL] > t2].copy()
    return train, val, test


def card_key(df: pd.DataFrame) -> pd.Series:
    """Approximate card identity. IEEE-CIS labels are card-level, so the same card
    appears in train and test; evaluating on unseen cards shows how much of the
    score is the model recognising cards rather than learning fraud patterns."""
    cols = [c for c in CARD_KEY_COLS if c in df.columns]
    part = lambda c: df[c].astype("string").fillna("nan")  # noqa: E731
    key = part(cols[0])
    for c in cols[1:]:
        key = key + "|" + part(c)
    return key


def period_days(df: pd.DataFrame) -> float:
    return float((df[TIME_COL].max() - df[TIME_COL].min()) / 86400) or 1.0
