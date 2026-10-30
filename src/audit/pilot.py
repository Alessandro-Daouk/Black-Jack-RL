"""Pilote de dimensionnement, graines réservées à la validation."""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

from .core import dataset, dump, fit_estimators, interval, new_output, train
from .oracle import Oracle


def run(output):
    output = new_output(output, {"phase": "pilot", "groups": 4, "rl_per_group": 2,
        "train_size": 20000, "validation_size": 6000, "episodes": 100000,
        "precision_half_width": .0005, "pilot_sd_inflation": 1.5})
    oracle = Oracle()
    start = time.perf_counter()
    # Warm-up non inclus dans les mesures pilotes.
    X, y, _, _ = dataset(1000, 30000)
    fit_estimators(X, y, X, y)
    train(30001, oracle.table(oracle.stand), [1000], oracle.evaluate)
    records, differences = [], []
    for i in range(4):
        X, y, _, _ = dataset(20000, 31000+10*i)
        Xv, yv, _, _ = dataset(6000, 31001+10*i)
        _, tables, fit = fit_estimators(X, y, Xv, yv)
        dump(output / f"fit_{i}.json", fit)
        d = []
        for j in range(2):
            pair = {}
            for method in ("standard", "logistic"):
                result = train(32000+10*i+j, tables['logistic'], [10000, 100000], oracle.evaluate,
                               eps_min=.4 if method == 'standard' else 1., standard=method == 'standard')
                pair[method] = result['curve'][-1]['j']
                records.append({"group": i, "rl_rep": j, "method": method, "curve": result['curve']})
            d.append(pair['logistic']-pair['standard'])
        differences.append(float(np.mean(d)))
        dump(output / "runs.json", records)
        print(f"Pilot group {i+1}/4 complete", flush=True)
    sd = float(np.std(differences, ddof=1))
    groups = 8
    while t.ppf(.975, groups-1)*1.5*sd/np.sqrt(groups) > .0005:
        groups += 1
    result = {"cluster_differences": differences, "statistics": interval(differences),
              "planned_groups": groups, "planned_half_width": float(t.ppf(.975, groups-1)*1.5*sd/np.sqrt(groups)),
              "wall_seconds": time.perf_counter()-start}
    dump(output / "design.json", result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
