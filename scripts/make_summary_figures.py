"""Recréer la figure de l'ablation depuis les CSV, sans entraînement."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import J_STAR


def main():
    folder = Path("results/1_premiere_campagne/hybride/frequencies_ablation")
    rows = list(csv.DictReader((folder / "learning_curves.csv").open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    labels = {"logistic": "Hybride : régression", "frequencies": "Hybride : fréquences",
              "exact": "Hybride : probabilités exactes", "standard": "Q-learning standard"}
    for method, label in labels.items():
        selected = [r for r in rows if r["method"] == method]
        checkpoints = sorted({int(r["episodes"]) for r in selected})
        values = [np.array([J_STAR-float(r["exact_expected_reward"]) for r in selected
                            if int(r["episodes"]) == n]) for n in checkpoints]
        mean = np.array([v.mean() for v in values]); sd = np.array([v.std(ddof=1) for v in values])
        ax.plot(checkpoints, mean, label=label)
        ax.fill_between(checkpoints, np.maximum(mean-sd, 1e-8), mean+sd, alpha=.12)
    ax.set(xlabel="Épisodes RL (coût des données supervisées exclu)",
           ylabel="Écart au gain optimal (récompense / partie)", yscale="log",
           title="Ablation : moyenne ± 1 écart-type, 20 graines RL")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(folder / "learning_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
