"""Draw the paper's focused comparison from committed aggregate results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({"font.size": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    data = pd.read_csv(ROOT / "results/full/robustness_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.45), sharey=True)
    models = [("linear_svc_word_gentle", "Word", "#24618A", "o"),
              ("linear_svc_char_gentle", "Character", "#B35425", "s"),
              ("linear_svc_word_char_gentle", "Word + character", "#368064", "^")]
    for ax, attack, title in zip(axes, ["leet", "separators", "unicode"], ["Leet", "Separators", "Unicode lookalikes"]):
        for model, label, color, marker in models:
            rows = data[(data.model == model) & (data.threshold_mode == "validation") &
                        (data.attack_type == attack)].sort_values("intensity")
            ax.plot(rows.intensity, rows.recall_mean, label=label, color=color, marker=marker, markersize=4)
            ax.fill_between(rows.intensity, rows.recall_mean - rows.recall_std,
                            rows.recall_mean + rows.recall_std, color=color, alpha=.12)
        ax.set(title=title, xlabel="Edit probability", ylim=(0.3, 1.01), xlim=(0, .5))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Spam recall")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    out = ROOT / "paper/figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "robustness.png", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
