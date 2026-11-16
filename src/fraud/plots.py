"""Report figures (static PNGs for the README)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID, SERIES = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_cost_curve(curve, chosen: float, path: Path) -> None:
    """Total validation cost by threshold, with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    _style(ax)
    ax.plot(curve["threshold"], curve["total_cost"] / 1000, color=SERIES, linewidth=2)
    best = curve.loc[(curve["threshold"] - chosen).abs().idxmin()]
    ax.scatter([chosen], [best["total_cost"] / 1000], s=50, color=SERIES, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"chosen: {chosen:.2f}", (chosen, best["total_cost"] / 1000), xytext=(8, 12),
                textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("Block threshold (model score)", color=MUTED, fontsize=9)
    ax.set_ylabel("Total cost, validation month (£k)", color=MUTED, fontsize=9)
    ax.set_title("Cost of missed fraud + blocked customers by threshold", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shap_summary(model, X_sample, path: Path, top_n: int = 15) -> None:
    """Global importance: mean |SHAP| per feature (LightGBM native contributions)."""
    contrib = model.booster_.predict(X_sample, pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.barh(np.array(X_sample.columns)[order], imp[order], color=SERIES, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (impact on log-odds of fraud)", color=MUTED, fontsize=9)
    ax.set_title(f"Top {top_n} drivers of the fraud score", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
