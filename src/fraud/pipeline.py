"""End-to-end training and evaluation.

    python -m fraud.pipeline                 # full run on data/
    python -m fraud.pipeline --skip-logistic # faster
    python -m fraud.pipeline --update-readme # write results into README.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from . import __version__
from .config import PROJECT_ROOT, load_config, resolve
from .data import AMOUNT, TARGET, card_key, load_raw, period_days, time_split
from .evaluate import decision_summary, pick_threshold, ranking_metrics, scale_to_days
from .experiment import plan_test
from .features import FeatureBuilder
from .model import train_lightgbm, train_logistic
from .monitoring import psi, psi_status
from .plots import plot_cost_curve, plot_shap_summary
from .report import inject, render
from .rules import RulesEngine


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(config_path=None, data_dir=None, nrows=None, skip_logistic=False,
        update_readme=False, shap_sample=2000, output_dir=None) -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    costs = cfg["costs"]
    fp_cost, fn_fee = costs["fp_cost"], costs["fn_fee"]
    if output_dir:  # keep dry runs away from the real reports/ and artefacts/
        reports, artefacts = resolve(output_dir) / "reports", resolve(output_dir) / "artefacts"
    else:
        reports, artefacts = resolve(cfg["paths"]["reports_dir"]), resolve(cfg["paths"]["artefacts_dir"])
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    artefacts.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    log("Loading data")
    d = cfg["data"]
    df = load_raw(resolve(data_dir or d["raw_dir"]), d["transaction_file"], d["identity_file"], nrows)
    train, val, test = time_split(df, cfg["split"]["train_end_quantile"], cfg["split"]["val_end_quantile"])
    del df
    log(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")
    y_tr, y_val, y_te = train[TARGET].to_numpy(), val[TARGET].to_numpy(), test[TARGET].to_numpy()
    amt_val, amt_te = val[AMOUNT].to_numpy(float), test[AMOUNT].to_numpy(float)

    # ── Features (fit on train only) ────────────────────────────────────────
    fb = FeatureBuilder(cfg["features"]["drop_missing_above"]).fit(train)
    X_tr, X_val, X_te = fb.transform(train), fb.transform(val), fb.transform(test)
    log(f"Features: {len(fb.feature_names)}")

    # ── Rules baseline ──────────────────────────────────────────────────────
    engine = RulesEngine(cfg["rules"])
    rule_val, rule_te = engine.score(val), engine.score(test)
    k_grid = np.arange(1, len(engine.rules) + 1)
    rules_k, _ = pick_threshold(y_val, amt_val, rule_val, k_grid, fp_cost, fn_fee)
    rules_k = int(rules_k)
    rule_table = engine.rule_report(train, y_tr)
    log(f"Rules: block when >= {rules_k} rule(s) fire (chosen on validation)")

    # ── Model ───────────────────────────────────────────────────────────────
    log("Training LightGBM (early stopping on validation)")
    model = train_lightgbm(X_tr, y_tr, X_val, y_val, cfg["model"], seed)
    p_val, p_te = model.predict_proba(X_val)[:, 1], model.predict_proba(X_te)[:, 1]
    tcfg = cfg["thresholds"]
    grid = np.round(np.arange(tcfg["grid_start"], tcfg["grid_stop"] + 1e-9, tcfg["grid_step"]), 4)
    threshold, val_curve = pick_threshold(y_val, amt_val, p_val, grid, fp_cost, fn_fee)
    log(f"Model: {model.best_iteration_} trees, threshold {threshold:.2f} (chosen on validation)")

    # ── Test month: evaluated once ──────────────────────────────────────────
    days = period_days(test)
    test_raw = {
        "rules": decision_summary(y_te, amt_te, rule_te >= rules_k, fp_cost, fn_fee),
        "model_default": decision_summary(y_te, amt_te, p_te >= 0.5, fp_cost, fn_fee),
        "model_tuned": decision_summary(y_te, amt_te, p_te >= threshold, fp_cost, fn_fee),
    }
    test_30d = {k: scale_to_days(v, days) for k, v in test_raw.items()}

    seen = card_key(test).isin(set(card_key(train))).to_numpy()
    ranking = {
        "model_all": ranking_metrics(y_te, p_te),
        "model_seen_cards": ranking_metrics(y_te[seen], p_te[seen]),
        "model_unseen_cards": ranking_metrics(y_te[~seen], p_te[~seen]),
    }
    if not skip_logistic:
        log("Training logistic regression baseline")
        lr = train_logistic(X_tr, y_tr, seed)
        ranking["logistic"] = ranking_metrics(y_te, lr.predict_proba(X_te)[:, 1])

    # ── Sensitivity: re-tune both systems on validation per scenario ────────
    sens = []
    for fpc in costs["sensitivity"]["fp_cost"]:
        for fee in costs["sensitivity"]["fn_fee"]:
            k_s, _ = pick_threshold(y_val, amt_val, rule_val, k_grid, fpc, fee)
            t_s, _ = pick_threshold(y_val, amt_val, p_val, grid, fpc, fee)
            r_cost = decision_summary(y_te, amt_te, rule_te >= k_s, fpc, fee)["total_cost"]
            m_cost = decision_summary(y_te, amt_te, p_te >= t_s, fpc, fee)["total_cost"]
            sens.append({"fp_cost": fpc, "fn_fee": fee, "rules_k": int(k_s), "model_threshold": t_s,
                         "saving_30d": (r_cost - m_cost) * 30 / days})

    # ── Drift + experiment sizing ───────────────────────────────────────────
    score_psi = psi(p_val, p_te)
    amount_psi = psi(train[AMOUNT], test[AMOUNT])
    e = cfg["experiment"]
    experiment = plan_test(y_te, amt_te, rule_te >= rules_k, fn_fee, len(test) / days,
                           e["alpha"], e["power"], e["relative_mde"], e["traffic_share_per_arm"])

    metrics = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_period_days": days,
        "costs": {"fp_cost": fp_cost, "fn_fee": fn_fee},
        "n_features": len(fb.feature_names),
        "n_trees": int(model.best_iteration_),
        "rules_k": rules_k,
        "model_threshold": threshold,
        "test_raw": test_raw,
        "test_30d": test_30d,
        "ranking": ranking,
        "unseen_card_share": float((~seen).mean()),
        "rules_on_train": rule_table.to_dict(orient="records"),
        "sensitivity": sens,
        "drift": {"score_psi": score_psi, "score_status": psi_status(score_psi),
                  "amount_psi": amount_psi, "amount_status": psi_status(amount_psi)},
        "experiment": experiment,
    }

    # ── Outputs ─────────────────────────────────────────────────────────────
    plot_cost_curve(val_curve, threshold, figures / "cost_vs_threshold.png")
    if shap_sample:
        n = min(shap_sample, len(X_te))
        plot_shap_summary(model, X_te.sample(n, random_state=seed), figures / "shap_summary.png")

    joblib.dump({
        "model": model, "feature_builder": fb, "threshold": threshold,
        "rules": cfg["rules"], "rules_k": rules_k, "version": __version__,
        "trained_at": metrics["trained_at"],
    }, artefacts / "model.joblib")
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    rule_table.to_csv(reports / "rules_on_train.csv", index=False)
    block = render(metrics)
    (reports / "results.md").write_text(block + "\n")
    if update_readme and inject(PROJECT_ROOT / "README.md", block):
        log("README results section updated")
    log(f"Done. Reports in {reports}, model in {artefacts / 'model.joblib'}")
    print("\n" + block)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--data-dir")
    p.add_argument("--nrows", type=int)
    p.add_argument("--skip-logistic", action="store_true")
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--shap-sample", type=int, default=2000, help="0 to skip SHAP plot")
    p.add_argument("--output-dir", help="write artefacts/ and reports/ here instead (for dry runs)")
    a = p.parse_args()
    if a.update_readme and a.output_dir:
        p.error("--update-readme only makes sense for a real run")
    run(a.config, a.data_dir, a.nrows, a.skip_logistic, a.update_readme, a.shap_sample, a.output_dir)


if __name__ == "__main__":
    main()
