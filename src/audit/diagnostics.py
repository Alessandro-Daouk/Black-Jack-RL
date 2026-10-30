"""Contrôles indépendants E1, asymptotique KL et perturbations E5."""
import argparse
import inspect
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import _logistic, _linear_loss

from src.environment import make_env
from src.hybrid_agent import load_agent
from src.supervised.evaluate import write_csv
from src.supervised.model import load_model
from .core import categorical_risk, digest, dump, interval, new_output, smoothed
from .oracle import Oracle


def run(output):
    output = new_output(output, {"phase": "diagnostics", "mc_games": 100000,
        "mc_seed_base": 35000, "kl_seed": 61000, "kl_repeats": 2000,
        "kl_sizes": [50, 100, 500, 1000, 5000, 20000, 60000, 200000]})
    oracle = Oracle()
    start = time.perf_counter()
    policies = {'always_stand': np.zeros(280), 'always_hit': np.ones(280),
                'random': np.full(280, .5), 'hit_below_17': np.array([s[0]<17 for s in oracle.states]),
                'optimal': oracle.policy}
    baseline_rows = []
    for k, (name, policy) in enumerate(policies.items()):
        rewards = np.zeros(100000)
        rng = np.random.default_rng(36000+k)
        with make_env() as env:
            for episode in range(len(rewards)):
                state, _ = env.reset(seed=35000+k if episode == 0 else None)
                while True:
                    action = int(rng.random() < policy[oracle.index[state]])
                    state, reward, terminal, truncated, _ = env.step(action)
                    assert not truncated
                    if terminal:
                        rewards[episode] = reward
                        break
        result = interval(rewards)
        exact = oracle.evaluate(policy)
        baseline_rows.append({'policy': name, 'games': len(rewards), 'env_seed': 35000+k,
                              'action_seed': 36000+k, 'exact_j': exact, 'mc_mean': result['mean'],
                              'ci95_low': result['ci95'][0], 'ci95_high': result['ci95'][1],
                              'z': (result['mean']-exact)/(result['sd']/np.sqrt(len(rewards))),
                              'wins': int((rewards == 1).sum()), 'draws': int((rewards == 0).sum()),
                              'losses': int((rewards == -1).sum())})
        print(f"Baseline {name} exact={exact:.8f}, MC={result['mean']:.8f}", flush=True)
    write_csv(output/'baselines.csv', baseline_rows)
    rng = np.random.default_rng(61000)
    joint = (oracle.cards[:, None]*oracle.probabilities).ravel()
    rows, summaries = [], []
    for n in (50, 100, 500, 1000, 5000, 20000, 60000, 200000):
        risks = {method: [] for method in ('raw', 'jeffreys', 'laplace')}
        for repeat in range(2000):
            counts = rng.multinomial(n, joint).reshape(10, 6)
            totals = counts.sum(axis=1, keepdims=True)
            raw = np.divide(counts, totals, out=np.full((10, 6), 1/6), where=totals>0)
            for method, p in [('raw', raw), ('jeffreys', smoothed(counts, .5)), ('laplace', smoothed(counts, 1.))]:
                risk = categorical_risk(p, oracle.probabilities, oracle.cards)
                risks[method].append(risk)
                rows.append({'n': n, 'replicate': repeat, 'method': method, 'kl': risk if np.isfinite(risk) else 'inf',
                             'empty_cells': int((counts == 0).sum())})
        for method, values in risks.items():
            finite = np.array(values)[np.isfinite(values)]
            summaries.append({'n': n, 'method': method, 'infinite_observations': int(2000-len(finite)),
                              'finite_realizations_only': interval(finite) if len(finite)>1 else None,
                              'unconditional_expectation': 'infinite' if method == 'raw' else 'finite',
                              'asymptotic_smoothed_reference': 25/n})
    write_csv(output/'kl_runs.csv', rows)
    dump(output/'kl_summary.json', summaries)
    # Échelle du terme quadratique : n_d fixé et vraie loi de la carte 6.
    p = oracle.probabilities[5]
    multinomial = rng.multinomial(1000, p, size=2000)
    quadratic = 1000*np.sum((multinomial/1000-p)**2/p, axis=1)
    write_csv(output/'pearson_runs.csv', [{'replicate': i, 'statistic': float(v)} for i, v in enumerate(quadratic)])
    dump(output/'pearson_summary.json', {'mean': float(quadratic.mean()), 'variance': float(quadratic.var(ddof=1)),
                                      'limit_chi_square_df': 5, 'limit_mean': 5, 'limit_variance': 10})
    perturbations = []
    for target in (21, 22):
        for rho in (0., .01, .05, .2):
            probabilities = (1-rho)*oracle.probabilities+rho*np.eye(6)[target-17]
            perturbations.append({'target': target, 'rho': rho, **oracle.diagnostics(probabilities)})
    write_csv(output/'perturbations.csv', perturbations)
    predictor = load_model()
    probabilities = predictor.probability_matrix()
    model = predictor.pipeline.named_steps['classifier']
    counts = np.asarray(predictor.metadata['training_counts'])
    freq = counts/counts.sum(axis=1, keepdims=True)
    residual = probabilities-freq+model.coef_.T/(model.C*counts.sum(axis=1, keepdims=True))
    sklearn_check = {'C': model.C, 'max_probability_stationarity_residual': float(np.max(np.abs(residual))),
        'max_bias_gradient_normalized': float(np.max(np.abs(np.sum(counts.sum(axis=1, keepdims=True)*(probabilities-freq),axis=0)/counts.sum()))),
        'objective': 'mean negative log likelihood + ||W||^2 / (2*C*n); intercept unpenalized',
        'source_files': {Path(m.__file__).name: digest(m.__file__) for m in (_logistic, _linear_loss)}}
    dump(output/'sklearn_objective.json', sklearn_check)
    source = inspect.getsource(_logistic).splitlines()
    selected = [f'{i+1}: {line}' for i, line in enumerate(source) if 'l2_reg_strength = 1.0 /' in line or 'sw_sum =' in line]
    (output/'sklearn_source_excerpt.txt').write_text('\n'.join(selected)+'\n\n'+inspect.getsource(_linear_loss.LinearModelLoss.loss_gradient),encoding='utf-8')
    dump(output/'delivered_model_bounds.json', oracle.diagnostics(probabilities))
    agent = load_agent()
    dump(output/'delivered_agent.json', oracle.diagnostics(probabilities, agent.policy()))
    q, _ = oracle.optimal(oracle.stand_from(probabilities))
    pi = q > oracle.stand_from(probabilities)
    occupation = oracle.occupation(pi)
    write_csv(output/'state_sensitivity.csv', [{'player': s[0], 'dealer': s[1], 'ace': s[2],
        'optimal_hit': int(oracle.policy[i]), 'limit_hit': int(pi[i]),
        'margin': float(abs(oracle.q_hit[i]-oracle.stand[i])), 'occupation_limit': float(occupation[i]),
        'value_stand_error': float(oracle.stand_from(probabilities)[i]-oracle.stand[i]),
        'value_hit_error': float(q[i]-oracle.q_hit[i])} for i, s in enumerate(oracle.states)])
    dump(output/'completion.json', {'status': 'complete', 'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
