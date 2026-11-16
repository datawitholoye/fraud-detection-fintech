"""Model training: LightGBM champion, logistic regression baseline."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_lightgbm(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
                   params: dict, seed: int = 42) -> lgb.LGBMClassifier:
    """Early stopping on the validation month (by average precision), never the test month."""
    params = dict(params)
    early = params.pop("early_stopping_rounds", 150)
    y_train = np.asarray(y_train)
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        **params, metric="average_precision", scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1,
    )
    with warnings.catch_warnings():
        # eval_set is deprecated in the newest LightGBM but is the only form older versions accept
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, np.asarray(y_val))],
            callbacks=[lgb.early_stopping(early, first_metric_only=True, verbose=False)],
        )
    return model


def train_logistic(X_train: pd.DataFrame, y_train, seed: int = 42) -> Pipeline:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # medians learnt on train only
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", C=0.1, max_iter=300,
                                     solver="saga", random_state=seed)),
    ])
    pipe.fit(X_train, np.asarray(y_train))
    return pipe


def reason_codes(model: lgb.LGBMClassifier, X: pd.DataFrame, raw: pd.DataFrame | None = None,
                 top_n: int = 3) -> list[list[dict]]:
    """Per-transaction top drivers using LightGBM's built-in SHAP contributions
    (no shap dependency needed at serving time). If `raw` is given, values are shown
    as the customer would see them (e.g. "C", not its category code)."""
    contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]  # last column = bias
    out = []
    for i in range(len(X)):
        idx = np.argsort(-contrib[i])[:top_n]
        out.append([
            {"feature": X.columns[j], "value": _display_value(X, raw, i, j),
             "contribution": round(float(contrib[i, j]), 4)}
            for j in idx if contrib[i, j] > 0
        ])
    return out


def _display_value(X: pd.DataFrame, raw: pd.DataFrame | None, i: int, j: int):
    col = X.columns[j]
    if raw is not None and col in raw.columns:
        v = raw[col].iloc[i]
    else:
        v = X.iat[i, j]
    if pd.isna(v):
        return None
    return v if isinstance(v, str) else float(v)
