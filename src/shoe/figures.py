"""Figures E10 à partir des CSV de la campagne, sans entraînement.

uv run --locked python -m src.shoe.figures --input results/5_sabot_fini
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"carte": "#8c8c8c", "table": "#e08a2b", "logistique": "#2b6cb0", "boosting": "#6b8e23",
          "standard": "#222222", "hybride_carte": "#8c8c8c", "hybride_table": "#e08a2b",
          "hybride_logistique": "#2b6cb0", "standard_lineaire": "#222222", "hybride_logistique_lineaire": "#2b6cb0"}


def main(folder):
    folder = Path(folder)
    supervised = pd.read_csv(folder / "supervised.csv")
    runs = pd.read_csv(folder / "runs.csv")
    basic = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["basic_strategy_j"]
    fig, (left, right, zoom) = plt.subplots(1, 3, figsize=(15, 4.2))

    for name, rows in supervised[supervised.model != "exacte"].groupby("model"):
        m = rows.groupby("n").excess_kl.agg(["mean", "sem"])
        left.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", capsize=3, label=name, color=COLORS[name])
    left.set(xscale="log", yscale="log", xlabel="mains de croupier simulées", ylabel="excès de log-loss (KL vers la loi vraie)",
             title="Modèle 1 : qualité des probabilités")
    left.legend(frameon=False)

    for ax, rows_all in ((right, runs), (zoom, runs[runs.episodes >= 200000])):
        for agent, rows in rows_all.groupby("agent"):
            per_group = rows.groupby(["episodes", "group"]).j.mean().groupby("episodes")
            m, s = per_group.mean(), per_group.sem()
            # agents linéaires (même information que la logistique) en pointillés
            ax.errorbar(m.index, m - basic, yerr=1.96 * s, marker="o", capsize=3, label=agent.replace("_", " ").replace("lineaire", "linéaire"),
                        color=COLORS[agent], ls="--" if agent.endswith("lineaire") else "-",
                        mfc="white" if agent.endswith("lineaire") else None)
        ax.axhline(0, color="#999999", lw=1, ls="--")
        ax.set(xscale="log", xlabel="parties d'apprentissage", ylabel="gain par partie − stratégie de base")
    right.text(runs.episodes.min(), 0, " stratégie de base du paquet infini", va="bottom", fontsize=8, color="#666666")
    right.set_title("Agents en sabot fini (IC 95 % entre jeux)")
    zoom.set_title("Agrandissement, 200 000 parties et plus")
    zoom.xaxis.set_minor_formatter(NullFormatter())
    zoom.set_xticks([2e5, 1e6, 3e6], ["200 000", "1 M", "3 M"])
    right.legend(frameon=False)
    for ax in (left, right, zoom):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "shoe.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    main(parser.parse_args().input)
