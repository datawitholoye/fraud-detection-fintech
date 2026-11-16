"""Turn metrics.json into the markdown results block used in the README."""
from __future__ import annotations

import re
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x):
    return f"{100 * x:.1f}%"


def _gbp(x):
    return f"£{x:,.0f}"


def _f(x):
    return "n/a" if x != x else f"{x:.3f}"  # NaN-safe


def headline(m: dict) -> str:
    r, ml = m["test_30d"]["rules"], m["test_30d"]["model_tuned"]
    saving = r["total_cost"] - ml["total_cost"]
    direction = "less" if saving >= 0 else "more"
    return (
        f"On the held-out test month, the cost-tuned LightGBM model caught "
        f"**{_pct(ml['value_detection_rate'])} of fraud by value** (rules: {_pct(r['value_detection_rate'])}) "
        f"while blocking **{_pct(ml['legit_block_rate'])} of genuine transactions** "
        f"(rules: {_pct(r['legit_block_rate'])}). Net, it cost **{_gbp(abs(saving))} {direction} per 30 days** "
        f"than a hand-written rules engine whose block threshold was tuned on the same validation month."
    )


def render(m: dict) -> str:
    t = m["test_30d"]
    rows = [("Rules engine (k = %d rules)" % m["rules_k"], t["rules"]),
            ("LightGBM @ 0.50 (default)", t["model_default"]),
            ("LightGBM @ %.2f (cost-tuned on validation)" % m["model_threshold"], t["model_tuned"])]
    out = [headline(m), "",
           "**Test month, scaled to 30 days** (thresholds chosen on validation, applied once to test):", "",
           "| Policy | Fraud caught (count) | Fraud caught (£ value) | Precision | Genuine txns blocked | Alerts / 1k txns | Total cost (30 days) |",
           "|---|---|---|---|---|---|---|"]
    for name, s in rows:
        out.append(f"| {name} | {_pct(s['recall'])} | {_pct(s['value_detection_rate'])} | {_pct(s['precision'])} | "
                   f"{_pct(s['legit_block_rate'])} | {s['alerts_per_1k']:.1f} | {_gbp(s['total_cost'])} |")

    rk = m["ranking"]
    out += ["", "**Ranking quality on the test month:**", "",
            "| Model | ROC-AUC | PR-AUC |", "|---|---|---|"]
    if "logistic" in rk:
        out.append(f"| Logistic regression | {_f(rk['logistic']['roc_auc'])} | {_f(rk['logistic']['pr_auc'])} |")
    out.append(f"| LightGBM (all test transactions) | {_f(rk['model_all']['roc_auc'])} | {_f(rk['model_all']['pr_auc'])} |")
    out.append(f"| LightGBM (cards seen in training) | {_f(rk['model_seen_cards']['roc_auc'])} | {_f(rk['model_seen_cards']['pr_auc'])} |")
    out.append(f"| LightGBM (cards **never** seen in training, {_pct(m['unseen_card_share'])} of test) | "
               f"{_f(rk['model_unseen_cards']['roc_auc'])} | {_f(rk['model_unseen_cards']['pr_auc'])} |")

    out += ["", "**Does the conclusion survive different cost assumptions?** "
            "Saving vs rules per 30 days (both systems re-tuned on validation for each scenario):", ""]
    fps = sorted({s["fp_cost"] for s in m["sensitivity"]})
    fns = sorted({s["fn_fee"] for s in m["sensitivity"]})
    out.append("| Chargeback fee \\ Cost per blocked customer | " + " | ".join(_gbp(f) for f in fps) + " |")
    out.append("|---" * (len(fps) + 1) + "|")
    for fn in fns:
        cells = [next(s for s in m["sensitivity"] if s["fp_cost"] == fp and s["fn_fee"] == fn)["saving_30d"] for fp in fps]
        out.append(f"| {_gbp(fn)} | " + " | ".join(_gbp(c) for c in cells) + " |")
    wins = sum(s["saving_30d"] > 0 for s in m["sensitivity"])
    out += ["", f"The model is cheaper in **{wins} of {len(m['sensitivity'])}** cost scenarios."]

    d = m["drift"]
    out += ["", f"**Drift check (PSI):** model scores validation → test = {d['score_psi']:.3f} ({d['score_status']}); "
            f"transaction amount train → test = {d['amount_psi']:.3f} ({d['amount_status']})."]

    e = m["experiment"]
    out += ["", f"**A/B test sizing** (rules as champion, model as challenger, detect a "
            f"{_pct(e['relative_mde'])} relative change, α = 0.05, power = 0.8, 50/50 split): "
            f"**{e['n_per_arm_loss']:,}** transactions per arm (≈ {e['days_needed_loss']:.0f} days) for fraud £ loss; "
            f"**{e['n_per_arm_block_rate']:,}** per arm (≈ {e['days_needed_block_rate']:.0f} days) for the "
            "genuine-block-rate guardrail. See [docs/rollout_plan.md](docs/rollout_plan.md)."]
    return "\n".join(out)


def inject(readme_path: str | Path, block: str) -> bool:
    """Replace the text between the RESULTS markers in the README."""
    path = Path(readme_path)
    text = path.read_text()
    if START not in text or END not in text:
        return False
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), f"{START}\n{block}\n{END}", text, flags=re.S)
    path.write_text(new)
    return True
