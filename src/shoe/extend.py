"""Prolonger les deux agents linéaires du sabot fini jusqu'à 10 millions de parties (exploratoire).

Question : à 3 millions de parties, le standard linéaire n'avait pas convergé ; l'hybride linéaire
reste-t-il au même niveau quand l'apprentissage continue ? Mêmes jeux, graines (première graine RL
de chaque jeu), réglages et sabots d'évaluation que la campagne : à 3 millions de parties, les
gains doivent retrouver exactement ceux de results/5_sabot_fini.

uv run --locked python -m src.shoe.extend --output results/5_sabot_fini/prolongation
"""
import argparse
from pathlib import Path
import time

import numpy as np
from joblib import Parallel, delayed

from src.audit.core import dump, new_output
from src.supervised.evaluate import write_csv
from .campaign import (DATA_SEED, EVAL_SHOES, GROUPS, RL_SEED, RL_SIZE, VALIDATION_SEED, VALIDATION_SIZE, gain,
                       paired_se, train, with_evaluation)
from .game import dealer_dataset
from .models import fit

CHECKPOINTS = [3_000_000, 6_000_000, 10_000_000]
AGENTS = {"standard_lineaire": 1.0, "hybride_logistique_lineaire": 0.1}  # ε_min retenus par la campagne


def logistic(i):
    X, y = dealer_dataset(RL_SIZE, DATA_SEED + 10 * i)
    Xv, yv = dealer_dataset(VALIDATION_SIZE, VALIDATION_SEED + 10 * i)
    return fit("logistique", X, y, Xv, yv)[0]


def run(output):
    output = new_output(output, {"checkpoints": CHECKPOINTS, "agents": AGENTS, "groups": GROUPS, "rl_rep": 0,
                                 "eval_shoes": [EVAL_SHOES.start, EVAL_SHOES.stop]})
    started = time.perf_counter()
    models = Parallel(n_jobs=GROUPS)(delayed(logistic)(i) for i in range(GROUPS))
    jobs = [(a, i) for a in AGENTS for i in range(GROUPS)]
    curves = Parallel(n_jobs=-1)(delayed(train)(models[i] if a.startswith("hybride") else None, RL_SEED + 10 * i,
                                                CHECKPOINTS, EVAL_SHOES, eps_min=AGENTS[a], linear=True)
                                 for a, i in jobs)
    point = {(a, i, p["episodes"]): (p["gains"], p["rounds"]) for (a, i), c in zip(jobs, curves, strict=True) for p in c}
    write_csv(output / "runs.csv", [{"agent": a, "group": i, "episodes": e, "j": gain(*v)} for (a, i, e), v in point.items()])
    hybrid, standard = "hybride_logistique_lineaire", "standard_lineaire"
    summary = {str(e): {
        "difference": with_evaluation([gain(*point[hybrid, i, e]) - gain(*point[standard, i, e]) for i in range(GROUPS)],
                                      paired_se([(point[hybrid, i, e], point[standard, i, e]) for i in range(GROUPS)])),
        "j": {a: float(np.mean([gain(*point[a, i, e]) for i in range(GROUPS)])) for a in AGENTS}}
        for e in CHECKPOINTS}
    dump(output / "summary.json", {"hybride_lineaire_minus_standard_lineaire": summary})
    dump(output / "completion.json", {"status": "complete", "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    from src.shoe.extend import run as run_extension  # fonctions importables : joblib les transmet aux processus
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run_extension(parser.parse_args().output)
