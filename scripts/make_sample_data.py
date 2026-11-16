"""Generate a small synthetic dataset with the IEEE-CIS schema.

Used by the tests and CI so the pipeline, API and Docker image can be checked
without the 1.3 GB Kaggle download. Numbers produced on this data mean nothing;
the real results come from running the pipeline on the Kaggle files.

    python scripts/make_sample_data.py --out data/sample --rows 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com",
           "anonymous.com", "protonmail.com", "mail.com"]


def make(rows: int = 20000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = rows
    dt = np.sort(rng.integers(86400, 86400 * 182, n))
    product = rng.choice(list("WCRHS"), n, p=[0.74, 0.12, 0.06, 0.05, 0.03])
    n_cards = max(n // 6, 50)
    card_pool = rng.integers(1000, 18000, n_cards)
    card_risk = rng.normal(0, 1, n_cards)
    card_idx = rng.integers(0, n_cards, n)
    p_email = rng.choice(DOMAINS, n, p=[.45, .17, .12, .08, .06, .05, .03, .02, .02]).astype(object)
    p_email[rng.random(n) < 0.15] = None
    r_email = np.where(rng.random(n) < 0.25, rng.choice(DOMAINS[:6], n), None).astype(object)
    hour = (dt % 86400) // 3600
    c1 = np.round(rng.lognormal(0.3, 1.0, n)).clip(0, 500)
    d1 = np.where(rng.random(n) < 0.2, 0, rng.integers(0, 640, n)).astype(float)
    addr_missing = rng.random(n) < 0.1
    v = rng.normal(0, 1, (n, 30))

    logit = (-4.3 + 0.9 * card_risk[card_idx] + 1.0 * (product == "C") + 0.6 * np.isin(p_email, DOMAINS[6:])
             + 0.5 * (hour <= 5) + 0.35 * np.log1p(c1) + 0.6 * (d1 <= 1) + 0.6 * addr_missing
             + 0.8 * v[:, 0] - 0.5 * v[:, 1])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    amt = np.round(rng.lognormal(4.0 + 0.5 * y, 1.0, n), 2)

    tx = pd.DataFrame({
        "TransactionID": np.arange(2987000, 2987000 + n), "isFraud": y, "TransactionDT": dt,
        "TransactionAmt": amt, "ProductCD": product, "card1": card_pool[card_idx],
        "card2": rng.choice([111.0, 321.0, 555.0, np.nan], n), "card3": 150.0,
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n, p=[.65, .3, .03, .02]),
        "card5": rng.choice([226.0, 224.0, 166.0], n), "card6": rng.choice(["debit", "credit"], n, p=[.75, .25]),
        "addr1": np.where(addr_missing, np.nan, rng.integers(100, 540, n)).astype(float),
        "addr2": np.where(addr_missing, np.nan, 87.0),
        "dist1": np.where(rng.random(n) < 0.6, np.nan, rng.exponential(50, n)),
        "P_emaildomain": p_email, "R_emaildomain": r_email,
    })
    for i in range(1, 15):
        tx[f"C{i}"] = c1 if i == 1 else np.round(rng.lognormal(0.2, 0.9, n))
    for i in range(1, 16):
        tx[f"D{i}"] = d1 if i == 1 else np.where(rng.random(n) < 0.4, np.nan, rng.integers(0, 600, n))
    for i in range(1, 10):
        tx[f"M{i}"] = np.where(rng.random(n) < 0.3, None, rng.choice(["T", "F"], n)).astype(object)
    for i in range(30):
        tx[f"V{i + 1}"] = np.round(v[:, i], 3)

    has_id = rng.random(n) < (0.2 + 0.3 * y)
    ids = tx.loc[has_id, ["TransactionID"]].copy()
    m = len(ids)
    ids["id_01"] = rng.choice([-5.0, 0.0, -10.0], m)
    ids["id_02"] = rng.integers(1, 600000, m).astype(float)
    ids["id_05"] = rng.integers(-10, 10, m).astype(float)
    ids["id_12"] = rng.choice(["Found", "NotFound"], m)
    ids["id_31"] = rng.choice(["chrome 63.0", "mobile safari 11.0", "edge 16.0"], m)
    ids["DeviceType"] = rng.choice(["desktop", "mobile"], m)
    ids["DeviceInfo"] = np.where(rng.random(m) < 0.3, None, rng.choice(["Windows", "iOS Device", "MacOS"], m))
    return tx, ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx, ids = make(a.rows, a.seed)
    tx.to_csv(out / "train_transaction.csv", index=False)
    ids.to_csv(out / "train_identity.csv", index=False)
    print(f"Wrote {len(tx):,} transactions ({tx.isFraud.mean():.1%} fraud) to {out}/")


if __name__ == "__main__":
    main()
