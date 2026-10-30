"""Réglage de eps_min et omega pour les deux agents, sur des graines distinctes de l'évaluation (0..4).

uv run python -m src.tune  ->  results/1_premiere_campagne/q_learning/tuning.csv
"""
import itertools

import pandas as pd
from joblib import Parallel, delayed

from src.evaluate import J_STAR
from src.q_learning import CONFIG, train

GRID = dict(eps_min=[0.05, 0.1, 0.2, 0.4, 0.6, 1.0], omega=[1.0, 0.8, 0.6])
TUNING_SEEDS = [100, 101, 102]


def run(agent, params, seed):
    gap = J_STAR - train(agent, seed, cfg={**CONFIG, **params})["curve"]
    return dict(agent=agent, **params, seed=seed, gap_final=gap[-1], gap_mean=gap.mean())


if __name__ == "__main__":
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(a, p, s) for a in ("standard", "hybrid") for p in combos for s in TUNING_SEEDS]
    df = pd.DataFrame(Parallel(n_jobs=-1, verbose=0)(delayed(run)(*j) for j in jobs))
    df.to_csv("results/1_premiere_campagne/q_learning/tuning.csv", index=False)
    table = df.groupby(["agent", *GRID]).agg(gap_final=("gap_final", "mean"), std=("gap_final", "std"),
                                              gap_mean=("gap_mean", "mean"))
    print(table.sort_values(["agent", "gap_final"]).round(5).to_string())
