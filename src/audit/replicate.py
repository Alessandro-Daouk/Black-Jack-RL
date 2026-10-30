"""Répliquer les 240 entraînements historiques sans toucher à leurs résultats."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.compare_hybrid import compare
from src.compare_frequencies import run as ablation
from src.supervised.model import load_model
from src.supervised.train import run_experiment
from .core import dump, new_output


def run(output, jobs=2):
    output = new_output(output, {"phase": "replication", "jobs": jobs})
    started = time.perf_counter()
    compare(output_dir=output/'config_vs_tuned', jobs=jobs)
    ablation(output=output/'frequencies_ablation', jobs=jobs)
    run_experiment(model_dir=output/'supervised/model', results_dir=output/'supervised/results')
    checks = {}
    for name, keys in [('config_vs_tuned', ['phase', 'method', 'preset', 'seed']),
                       ('frequencies_ablation', ['method', 'seed'])]:
        for file, extra in [('runs.csv', []), ('learning_curves.csv', ['episodes'])]:
            old = pd.read_csv(Path('results/hybrid')/name/file).set_index(keys+extra)
            new = pd.read_csv(output/name/file).set_index(keys+extra)
            diff = (old.exact_expected_reward-new.exact_expected_reward).abs()
            matched = bool(old.index.sort_values().equals(new.index.sort_values()))
            # Un écart est consigné, pas masqué par une assertion qui interromprait la campagne.
            checks[name+'/'+file] = {"rows": len(diff), "max_absolute_difference": float(diff.max()),
                                     "matched_indices": matched, "tolerance_1e-9_passed": bool(matched and diff.max() < 1e-9)}
    p0 = load_model().probability_matrix()
    p1 = load_model(output/'supervised/model/dealer_model.joblib').probability_matrix()
    checks['supervised_max_probability_difference'] = float(np.max(np.abs(p0-p1)))
    checks['supervised_tolerance_1e-7_passed'] = bool(np.allclose(p0, p1, atol=1e-7, rtol=0))
    passed = all(c['tolerance_1e-9_passed'] for c in checks.values() if isinstance(c, dict)) and checks['supervised_tolerance_1e-7_passed']
    dump(output/'comparison.json', {"checks": checks, "all_checks_passed": passed, "wall_seconds": time.perf_counter()-started,
        "scope": "160 CONFIG/TUNED + 80 ablation + modele supervise ; tuning 108 runs et notebook non reexecutes"})
    dump(output/'completion.json', {"status": "complete", "all_checks_passed": passed})
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.jobs)
