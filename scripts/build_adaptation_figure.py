"""Show both spam detection and ordinary-message errors on the new holdouts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/adaptation/evaluation"


def main():
    results = pd.read_csv(DIRECTORY / "test_metrics.csv")
    sources = ["financial", "sms_history"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, metric, title in zip(axes, ["recall", "fpr"],
                                    ["Spam detected (higher is better)", "Ham falsely flagged (lower is better)"]):
        for offset, name, label, color in [(-.18, "baseline", "Original", "#246c76"),
                                            (.18, "candidate", "Adapted", "#b65a3a")]:
            selected = results[results.model.eq(name)].set_index("source").loc[sources]
            values = 100 * selected[metric].to_numpy()
            bars = axis.bar(np.arange(2) + offset, values, width=.34, label=label, color=color)
            axis.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=9)
        axis.set(xticks=[0, 1], xticklabels=["Financial holdout\n46 spam / 32 ham", "SMS history\n699 spam / 55 PSAs"],
                 ylim=(0, 100), title=title)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.savefig(DIRECTORY / "comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
