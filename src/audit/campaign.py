"""Campagne sérialisée E2–E5 : plan imbriqué, coûts et artefacts par entraînement."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

from src.supervised.evaluate import metrics, write_csv
from src.supervised.train import fingerprint
from src.supervised.values import build_stand_table
from .core import categorical_risk, dataset, digest, dump, fit_estimators, new_output, train
from .oracle import Oracle

CHECKPOINTS = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000]
SIZES = [1000, 5000, 20000, 60000]


def freeze(pilot, output):
    design = json.loads((Path(pilot)/'design.json').read_text(encoding='utf-8'))
    secondary_groups = 8
    sd = design['statistics']['sd']
    while t.ppf(.975, secondary_groups-1)*1.5*sd/np.sqrt(secondary_groups) > .001:
        secondary_groups += 1
    config = {"schema": 2, "phase": "final", "groups": design['planned_groups'],
              "secondary_groups": secondary_groups, "rl_per_group": 2,
              "sizes": SIZES, "checkpoints": CHECKPOINTS, "validation_size": 6000, "test_size": 12000,
              "data_seed_base": 41000, "rl_seed_base": 51000, "order_seed": 49000,
              "pilot_design_sha256": digest(Path(pilot)/'design.json'),
              "methods": ['logistic', 'frequencies', 'jeffreys', 'exact', 'standard', 'standard_common'],
              "ablations": True, "thresholds": [.01, .005, .002, .001],
              "primary": {"size": 20000, "episodes": 100000, "contrast": "logistic-standard"},
              "practical_effect": .001, "target_half_width": .0005}
    if Path(output).exists():
        raise ValueError('Configuration déjà figée ; choisir un nouveau chemin.')
    dump(output, config)
    return config


def run(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    required = ('groups', 'rl_per_group', 'sizes', 'checkpoints', 'validation_size', 'test_size')
    if any(k not in config for k in required) or config['groups'] < 2:
        raise ValueError('Configuration incomplète.')
    output = new_output(output, config)
    oracle = Oracle()
    order = np.random.default_rng(config['order_seed'])
    started = time.perf_counter()
    Xw, yw, _, _ = dataset(1000, 40000)
    fit_estimators(Xw, yw, Xw, yw)
    train(40001, oracle.table(oracle.stand), [1000], oracle.evaluate)
    rows, curves, estimates = [], [], []
    sizes = sorted(config['sizes'])
    for i in range(config['groups']):
        secondary = i < config.get('secondary_groups', config['groups'])
        active_sizes = sizes if secondary else [config['primary']['size']]
        checkpoints = config['checkpoints'] if secondary else [k for k in config['checkpoints'] if k <= config['primary']['episodes']]
        base = config['data_seed_base']+10*i
        X, y, draws, elapsed = dataset(max(active_sizes), base)
        Xv, yv, dv, tv = dataset(config['validation_size'], base+1)
        fitted = {}
        for n in active_sizes:
            matrices, tables, fitting = fit_estimators(X[:n], y[:n], Xv, yv)
            fitting.update({"group": i, "n": n, "train_seed": base, "validation_seed": base+1,
                "train_sha256": fingerprint(X[:n], y[:n]), "validation_sha256": fingerprint(Xv, yv),
                "train_generation_seconds": float(elapsed[n-1]), "validation_generation_seconds": float(tv[-1]),
                "train_cards": int(draws[n-1]), "validation_cards": int(dv[-1]),
                "probabilities": {m: p.tolist() for m, p in matrices.items()}})
            fitted[n] = matrices, tables, fitting
        # Tous les C sont choisis avant la génération du test et l'accès aux métriques analytiques.
        Xt, yt, dt, tt = dataset(config['test_size'], base+2)
        for n, (matrices, tables, fitting) in fitted.items():
            fitting.update({"test_seed": base+2, "test_sha256": fingerprint(Xt, yt),
                            "test_generation_seconds": float(tt[-1]), "test_cards": int(dt[-1])})
            start = time.perf_counter()
            for method, probabilities in matrices.items():
                risk = categorical_risk(probabilities, oracle.probabilities, oracle.cards)
                scores = metrics(probabilities, Xt, yt, oracle.probabilities)
                estimates.append({"group": i, "n": n, "method": method,
                                  "raw_kl": risk if np.isfinite(risk) else None, "raw_kl_infinite": not np.isfinite(risk),
                                  **scores, **oracle.diagnostics(probabilities)})
            fitting['metrics_seconds'] = time.perf_counter()-start
            dump(output/f'fits/group_{i}_n_{n}.json', fitting)
        tasks = [(n, method) for n in active_sizes for method in ('logistic', 'frequencies', 'jeffreys')]
        tasks += [(0, method) for method in ('exact', 'standard', 'standard_common')]
        if config.get('ablations') and secondary:
            tasks += [(20000, method) for method in ('epsilon_low', 'omega_08', 'noise_005', 'noise_020')]
        if not secondary:
            tasks = [(20000, 'logistic'), (0, 'standard')]
        for j in range(config['rl_per_group']):
            for position in order.permutation(len(tasks)):
                n, method = tasks[position]
                standard = method in ('standard', 'standard_common')
                epsilon = .4 if method == 'standard' else .05 if method == 'epsilon_low' else 1.
                omega = .8 if method == 'omega_08' else 1.
                if n == 0:
                    probabilities = oracle.probabilities
                    V = oracle.table(oracle.stand)
                    fitting = None
                elif method.startswith('noise_'):
                    rho = .05 if method == 'noise_005' else .2
                    probabilities = (1-rho)*oracle.probabilities + rho*np.eye(6)[5]
                    V, fitting = build_stand_table(probabilities), None
                else:
                    matrices, tables, fitting = fitted[n]
                    estimator = method if method in matrices else 'logistic'
                    probabilities, V = matrices[estimator], tables[estimator]
                seed = config['rl_seed_base']+10*i+j
                result = train(seed, V, checkpoints, oracle.evaluate,
                               eps_min=epsilon, omega=omega, standard=standard)
                key = f'g{i}_r{j}_n{n}_{method}'
                path = output/f'agents/{key}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, Q=result['Q'], visits=result['visits'], updates=result['N'],
                                    states=np.array(oracle.states), probabilities=probabilities)
                start = time.perf_counter()
                with np.load(path, allow_pickle=False) as saved:
                    saved['Q'].copy()
                load_seconds = time.perf_counter()-start
                costs = {"acquisition_seconds": 0., "minimal_seconds": 0., "supervised_cards": 0,
                         "minimal_cards": 0, "scientific_extra_seconds": 0., "scientific_extra_cards": 0}
                if fitting is not None:
                    estimator = method if method in ('logistic', 'frequencies', 'jeffreys') else 'logistic'
                    is_lr = estimator == 'logistic'
                    estimation_time = fitting['selection_seconds'] if is_lr else fitting['frequency_seconds']
                    if estimator == 'jeffreys':
                        estimation_time += fitting['smoothing_seconds']
                    costs['acquisition_seconds'] = fitting['train_generation_seconds']+estimation_time+fitting['stand_build_seconds'][estimator]
                    costs['supervised_cards'] = fitting['train_cards']
                    if is_lr:
                        costs['acquisition_seconds'] += fitting['validation_generation_seconds']
                        costs['supervised_cards'] += fitting['validation_cards']
                    costs['minimal_seconds'] = fitting['train_generation_seconds']+(
                        fitting['selected_fit_seconds'] if is_lr else estimation_time)+fitting['stand_build_seconds'][estimator]
                    costs['minimal_cards'] = fitting['train_cards']
                    costs['scientific_extra_seconds'] = fitting['test_generation_seconds']+fitting['metrics_seconds']
                    costs['scientific_extra_cards'] = fitting['test_cards']
                    if not is_lr:
                        costs['scientific_extra_seconds'] += fitting['validation_generation_seconds']
                        costs['scientific_extra_cards'] += fitting['validation_cards']
                info = {"run": key, "group": i, "rl_rep": j, "n": n, "method": method, "seed": seed,
                        "eps_min": epsilon, "omega": omega, "agent_sha256": digest(path),
                        "load_seconds": load_seconds, "peak_process_rss_bytes": result['peak_process_rss_bytes'],
                        "table_bytes": result['table_bytes'], **costs}
                diagnostic = oracle.diagnostics(probabilities, result['Q'][1] > result['Q'][0])
                reach_visits = np.array([[result['visits'][a][s] for s in oracle.states] for a in (0, 1)])
                rows.append({**info, **diagnostic, "min_hit_visits": int(reach_visits[1].min()),
                             "median_hit_visits": float(np.median(reach_visits[1])),
                             "visited_hit_states": int(np.count_nonzero(reach_visits[1])),
                             **result['curve'][-1]})
                for point in result['curve']:
                    curves.append({**info, **point, "regret": oracle.j_star-point['j'],
                        "total_seconds": costs['acquisition_seconds']+point['rl_seconds'],
                        "total_cards": costs['supervised_cards']+point['cards'],
                        "minimal_total_seconds": costs['minimal_seconds']+point['rl_seconds'],
                        "minimal_total_cards": costs['minimal_cards']+point['cards'],
                        "reused_total_seconds": load_seconds+point['rl_seconds']})
                # Écriture régulière : interruption récupérable, aucune confusion avec campagne finie.
                write_csv(output/'runs.csv', rows)
                write_csv(output/'curves.csv', curves)
                print(f"Final {key} J={result['curve'][-1]['j']:.8f}", flush=True)
        write_csv(output/'estimates.csv', estimates)
    dump(output/'completion.json', {"status": "complete", "runs": len(rows),
        "wall_seconds": time.perf_counter()-started, "optimal_j": oracle.j_star,
        "run_table_sha256": digest(output/'runs.csv'), "curve_table_sha256": digest(output/'curves.csv')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.pilot:
        freeze(args.pilot, args.output)
    elif args.config:
        run(args.config, args.output)
    else:
        parser.error('--config ou --pilot requis')
