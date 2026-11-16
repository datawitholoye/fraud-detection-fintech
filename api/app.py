"""Real-time scoring API.

    uvicorn api.app:app --reload
    curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
         -d '{"transaction": {"TransactionAmt": 1250.0, "ProductCD": "C", "P_emaildomain": "protonmail.com"}}'

Returns the model decision, the rules-engine decision (for shadow-mode comparison)
and the top reasons behind the score, so an analyst or a declined-payment notice
can say *why*.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud.model import reason_codes
from fraud.rules import RulesEngine

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[1] / "artefacts" / "model.joblib"))

app = FastAPI(title="Fraud scoring API", version="2.0.0")


class ScoreRequest(BaseModel):
    transaction: dict[str, Any] = Field(
        ..., description="Raw IEEE-CIS fields. Anything missing is treated as unknown.",
        examples=[{"TransactionAmt": 1250.0, "ProductCD": "C", "card4": "visa",
                   "P_emaildomain": "protonmail.com", "TransactionDT": 7200}],
    )


class Reason(BaseModel):
    feature: str
    value: float | str | None
    contribution: float


class ScoreResponse(BaseModel):
    fraud_score: float
    threshold: float
    decision: str
    rules_fired: list[str]
    rules_decision: str
    top_reasons: list[Reason]
    model_version: str
    latency_ms: float


@lru_cache(maxsize=1)
def load_artefact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run `make train` first.")
    art = joblib.load(MODEL_PATH)
    art["engine"] = RulesEngine(art["rules"])
    return art


@app.get("/health")
def health() -> dict:
    try:
        art = load_artefact()
        return {"status": "ok", "model_version": art["version"], "trained_at": art["trained_at"]}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    start = time.perf_counter()
    try:
        art = load_artefact()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    raw = pd.DataFrame([req.transaction])
    X = art["feature_builder"].transform(raw)
    p = float(art["model"].predict_proba(X)[0, 1])
    hits = art["engine"].evaluate(raw).iloc[0]
    fired = [name for name, hit in hits.items() if hit]
    return ScoreResponse(
        fraud_score=round(p, 6),
        threshold=art["threshold"],
        decision="BLOCK" if p >= art["threshold"] else "APPROVE",
        rules_fired=fired,
        rules_decision="BLOCK" if len(fired) >= art["rules_k"] else "APPROVE",
        top_reasons=reason_codes(art["model"], X, raw)[0],
        model_version=art["version"],
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
