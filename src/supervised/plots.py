"""Figures exportables pour l'analyse du premier modèle."""

import os
from pathlib import Path
import tempfile

import numpy as np

from .constants import OUTCOME_NAMES
from .values import build_stand_table


def make_plots(learned, baseline, reference, counts, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blackjack-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cards = np.arange(1, 11)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for k, ax in enumerate(axes.flat):
        ax.plot(cards, reference[:, k], color="black", label="Référence exacte", linewidth=1.5)
        ax.plot(cards, learned[:, k], color="#176a96", label="Régression logistique", linestyle="--", marker="o", markersize=3)
        ax.plot(cards, baseline[:, k], color="#c37529", label="Fréquences d'entraînement", linestyle=":")
        valid = counts.sum(axis=1) > 0
        n = counts.sum(axis=1)[valid]
        frequencies = counts[valid, k] / n
        # IC de Wilson point par point (non simultanés) pour les fréquences du test.
        z = 1.96
        center = (frequencies + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(frequencies * (1 - frequencies) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        errors = np.maximum(np.array([frequencies - (center - half), center + half - frequencies]), 0)
        ax.errorbar(cards[valid], frequencies, yerr=errors, color="0.55", fmt=".", capsize=2, label="Test : fréquence et IC Wilson 95 %")
        ax.set(title=f"Issue {OUTCOME_NAMES[k]}", xlabel="Carte visible (1 = as)", ylabel="Probabilité", xticks=cards)
        ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Probabilités du croupier par carte visible — modèle, fréquences et référence")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)

    error = build_stand_table(learned)[4:, 1:, 0] - build_stand_table(reference)[4:, 1:, 0]
    scale = max(float(np.abs(error).max()), 1e-6)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(error, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set(xticks=range(10), xticklabels=cards, yticks=range(18), yticklabels=range(4, 22),
           xlabel="Carte visible du croupier (1 = as)", ylabel="Total du joueur",
           title="Erreur de V_STAND : régression logistique − référence exacte")
    fig.colorbar(im, ax=ax, label="Écart de gain espéré (récompenses −1 / 0 / +1)")
    fig.tight_layout()
    fig.savefig(output / "stand_error.png", dpi=160)
    plt.close(fig)
