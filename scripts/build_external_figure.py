"""Draw the frozen-model recall comparison from saved tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/external/imc25/evaluation/recall.png"


def main():
    external = pd.read_csv(OUT.parent / "metrics.csv")
    old = pd.read_csv(ROOT / "results/full/confidence_intervals.csv")
    old = old[(old.metric == "recall") & (old.threshold_mode == "validation")].iloc[0]
    rows = external[(external.view == "published") & (external.threshold_mode == "validation")]
    values = [old.estimate, *rows.recall]
    low = [old.lower, *rows.recall_lower]
    high = [old.upper, *rows.recall_upper]
    labels = ["Historical test · 130 spam", "External 2023–2024 · 372 reports", "External all English · 15,531 reports"]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    positions = np.arange(3)
    ax.barh(positions, values, height=.48, color=["#246c76", "#b65a3a", "#b65a3a"], alpha=.9)
    ax.errorbar(values, positions, xerr=[np.array(values)-low, np.array(high)-values],
                fmt="none", color="#263238", capsize=4, linewidth=1.3)
    for y, value in zip(positions, values):
        ax.text(.02, y, f"{value:.2%}", va="center", color="white", fontweight="bold", fontsize=11)
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 1.02), xlabel="Recall on positive messages",
           title="Frozen validation threshold · conditional 95% group intervals")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(OUT, dpi=220)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
