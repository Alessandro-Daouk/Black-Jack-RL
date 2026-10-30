"""Références RL supplémentaires et décomposition de variance (E9), paquet infini.

Expected SARSA, contrôle Monte-Carlo, Dyna-Q et RL à modèle appris par l'expérience
(équivalence certaine), réglés comme le standard puis entraînés avec les graines RL de la
campagne finale ; puis 10 graines RL par jeu supervisé pour séparer la variance des données
supervisées de celle de l'apprentissage. Lit la campagne finale sans la modifier.

uv run --locked python -m src.audit.baselines --output results/4_autres_methodes/autres_algorithmes
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from joblib import Parallel, delayed

from src.environment import HIT, SHAPE, STAND, make_env
from src.supervised.evaluate import write_csv
from src.supervised.values import build_stand_table
from .complements import FINAL, GRID, TUNING_SEEDS, final_points
from .core import dump, interval, new_output, train
from .oracle import Oracle

KINDS = ("expected_sarsa", "monte_carlo", "dyna_q")
PLANNING_STEPS = 10  # Dyna-Q : mises à jour simulées par pas réel, valeur usuelle non réglée
CHECKPOINTS = [1000, 5000, 10000, 20000, 40000, 100000, 200000]
VARIANCE_REPS = 10
VARIANCE_SEED_BASE = 91000


def train_td(kind, seed, checkpoints, evaluate, eps_min=1., omega=1.):
    """Même environnement, même ε décroissant et même pas 1/N^ω que le Q-learning standard."""
    if kind not in KINDS or not 0 < eps_min <= 1 or not .5 < omega <= 1:
        raise ValueError("Algorithme, epsilon ou omega invalide.")
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, *SHAPE))
    N = np.zeros((2, *SHAPE), dtype=np.int64)
    model, seen = {}, []  # Dyna-Q : transitions observées par couple (s, a)
    targets, curve = set(checkpoints), []

    def greedy(s):
        return HIT if Q[HIT][s] > Q[STAND][s] else STAND

    def step_update(s, a, y):
        N[a][s] += 1
        Q[a][s] += (y - Q[a][s]) / N[a][s] ** omega

    def target(r, s2, term, eps):
        if term:
            return r
        if kind == "expected_sarsa":
            g = greedy(s2)
            return r + eps / 2 * (Q[HIT][s2] + Q[STAND][s2]) + (1 - eps) * Q[g][s2]
        return r + max(Q[HIT][s2], Q[STAND][s2])

    with make_env() as env:
        for episode in range(max(checkpoints)):
            s, _ = env.reset(seed=seed if episode == 0 else None)
            eps = max(eps_min, .99995 ** episode)
            path = []
            while True:
                a = int(rng.integers(2)) if rng.random() < eps else greedy(s)
                s2, r, term, trunc, _ = env.step(a)
                if trunc:
                    raise RuntimeError("Troncature inattendue.")
                if kind == "monte_carlo":
                    path.append((s, a))
                else:
                    step_update(s, a, target(r, s2, term, eps))
                if kind == "dyna_q":
                    if (s, a) not in model:
                        model[s, a] = []
                        seen.append((s, a))
                    model[s, a].append((r, s2, term))
                    for _ in range(PLANNING_STEPS):
                        ps, pa = seen[rng.integers(len(seen))]
                        outcomes = model[ps, pa]
                        pr, ps2, pterm = outcomes[rng.integers(len(outcomes))]
                        step_update(ps, pa, target(pr, ps2, pterm, eps))
                if term:
                    break
                s = s2
            for ps, pa in path:  # γ = 1 et une seule récompense : le retour vaut r pour tout le chemin
                step_update(ps, pa, r)
            if episode + 1 in targets:
                curve.append({"episodes": episode + 1, "j": evaluate(Q[HIT] > Q[STAND])})
    return curve


def train_model_based(seed, checkpoints, oracle):
    """Équivalence certaine : MDP estimé par comptage des transitions jouées, résolu exactement.

    Comportement uniforme (ε = 1, comme l'hybride réglé) ; à chaque point, la politique est
    optimale pour le modèle estimé. Couples jamais essayés : valeur 0, comme l'initialisation Q.
    """
    rng = np.random.default_rng(seed)
    n = len(oracle.states)
    nxt, bust, n_hit = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    stand_sum, n_stand = np.zeros(n), np.zeros(n)
    targets, curve = set(checkpoints), []
    with make_env() as env:
        for episode in range(max(checkpoints)):
            s, _ = env.reset(seed=seed if episode == 0 else None)
            while True:
                a = int(rng.integers(2))
                s2, r, term, trunc, _ = env.step(a)
                if trunc:
                    raise RuntimeError("Troncature inattendue.")
                i = oracle.index[tuple(int(v) for v in s)]
                if a == STAND:
                    stand_sum[i] += r
                    n_stand[i] += 1
                else:
                    n_hit[i] += 1
                    if term:
                        bust[i] += 1
                    else:
                        nxt[i, oracle.index[tuple(int(v) for v in s2)]] += 1
                if term:
                    break
                s = s2
            if episode + 1 in targets:
                stand = np.divide(stand_sum, n_stand, out=np.zeros(n), where=n_stand > 0)
                v, q = np.zeros(n), np.zeros(n)
                for i in reversed(range(n)):  # successeurs de HIT d'abord, comme Oracle.optimal
                    q[i] = (nxt[i] @ v - bust[i]) / n_hit[i] if n_hit[i] else 0.
                    v[i] = max(q[i], stand[i])
                curve.append({"episodes": episode + 1, "j": oracle.evaluate(q > stand)})
    return curve


def tune(kind, oracle):
    jobs = [(p, s) for p in (dict(zip(GRID, v)) for v in itertools.product(*GRID.values())) for s in TUNING_SEEDS]
    curves = Parallel(n_jobs=-1)(delayed(train_td)(kind, s, [100000], oracle.evaluate, **p) for p, s in jobs)
    rows = [{"agent": kind, **p, "seed": s, "regret": oracle.j_star - c[-1]["j"]}
            for (p, s), c in zip(jobs, curves, strict=True)]
    means = {}
    for row in rows:
        means.setdefault((row["eps_min"], row["omega"]), []).append(row["regret"])
    best = min(means, key=lambda key: np.mean(means[key]))
    return rows, {"eps_min": best[0], "omega": best[1], "mean_regret": float(np.mean(means[best]))}


def variance_components(values):
    """ANOVA à un facteur aléatoire : values[i, r] = jeu supervisé i, graine RL r."""
    a = np.asarray(values, float)
    g, r = a.shape
    within = float(a.var(axis=1, ddof=1).mean())
    between = max(float(a.mean(axis=1).var(ddof=1)) - within / r, 0.)
    return {"groups": g, "rl_reps": r, "sd_between_datasets": between ** .5, "sd_within_rl": within ** .5,
            "share_datasets": between / (between + within) if between + within else 0.}


def run(output):
    config = json.loads((FINAL / "config.json").read_text(encoding="utf-8"))
    output = new_output(output, {"grid": GRID, "tuning_seeds": TUNING_SEEDS, "kinds": KINDS,
                                 "planning_steps": PLANNING_STEPS, "checkpoints": CHECKPOINTS,
                                 "model_based": "comptage, comportement uniforme, aucun réglage",
                                 "variance_reps": VARIANCE_REPS, "variance_seed_base": VARIANCE_SEED_BASE})
    started = time.perf_counter()
    oracle = Oracle()
    groups, secondary = range(config["groups"]), range(config["secondary_groups"])
    jobs = [(i, j) for i in groups for j in (0, 1)]
    seed = {(i, j): config["rl_seed_base"] + 10 * i + j for i, j in jobs}
    summary, tuning_rows, rows = {"tuning": {}}, [], []

    for kind in KINDS:
        kind_rows, summary["tuning"][kind] = tune(kind, oracle)
        tuning_rows += kind_rows
        choice = summary["tuning"][kind]
        curves = Parallel(n_jobs=-1)(delayed(train_td)(kind, seed[k], CHECKPOINTS, oracle.evaluate,
                                                       choice["eps_min"], choice["omega"]) for k in jobs)
        rows += [{"method": kind, "group": i, "rl_rep": j, **p} for (i, j), c in zip(jobs, curves, strict=True) for p in c]
    curves = Parallel(n_jobs=-1)(delayed(train_model_based)(seed[k], CHECKPOINTS, oracle) for k in jobs)
    rows += [{"method": "model_based", "group": i, "rl_rep": j, **p} for (i, j), c in zip(jobs, curves, strict=True)
             for p in c]
    write_csv(output / "tuning.csv", tuning_rows)
    write_csv(output / "curves.csv", rows)

    # Contrastes appariés par graine RL, moyennés par jeu supervisé puis IC entre jeux.
    j = {(r["method"], r["group"], r["rl_rep"], r["episodes"]): r["j"] for r in rows}
    points = final_points()  # budgets 10 000, 40 000, 100 000 pour les 28 jeux
    summary["contrasts"] = {}
    for method in (*KINDS, "model_based"):
        summary["contrasts"][method] = {str(b): {
            "regret": interval([np.mean([oracle.j_star - j[method, i, r, b] for r in (0, 1)]) for i in groups]),
            "minus_standard": interval([np.mean([j[method, i, r, b] - points["standard", i, r, b] for r in (0, 1)])
                                        for i in groups]),
            "hybrid_minus": interval([np.mean([points["hybrid", i, r, b] - j[method, i, r, b] for r in (0, 1)])
                                      for i in groups])} for b in (10000, 40000, 100000)}
        summary["contrasts"][method]["regret_by_budget_secondary"] = {
            str(b): float(np.mean([oracle.j_star - j[method, i, r, b] for i in secondary for r in (0, 1)]))
            for b in CHECKPOINTS}

    # Décomposition de variance : 10 graines RL par jeu, hybride logistique 20 000 mains et standard réglé.
    tables = {i: build_stand_table(np.array(json.loads(
        (FINAL / f"fits/group_{i}_n_20000.json").read_text(encoding="utf-8"))["probabilities"]["logistic"]))
        for i in secondary}
    reps = [(i, r) for i in secondary for r in range(VARIANCE_REPS)]
    rl = {k: VARIANCE_SEED_BASE + 100 * k[0] + k[1] for k in reps}
    hybrid = Parallel(n_jobs=-1)(delayed(train)(rl[k], tables[k[0]], [100000], oracle.evaluate) for k in reps)
    standard = Parallel(n_jobs=-1)(delayed(train)(rl[k], None, [100000], oracle.evaluate, eps_min=.4, standard=True)
                                   for k in reps)
    shape = (len(secondary), VARIANCE_REPS)
    hj = np.array([h["curve"][-1]["j"] for h in hybrid]).reshape(shape)
    sj = np.array([s["curve"][-1]["j"] for s in standard]).reshape(shape)
    write_csv(output / "variance_runs.csv", [{"group": i, "rl_rep": r, "seed": rl[i, r], "hybrid_j": float(hj[i, r]),
                                              "standard_j": float(sj[i, r])} for i, r in reps])
    summary["variance_100000"] = {"hybrid_regret": variance_components(oracle.j_star - hj),
                                  "difference": variance_components(hj - sj),
                                  "difference_interval": interval((hj - sj).mean(axis=1)),
                                  "pairs_hybrid_better": int(np.count_nonzero(hj > sj)), "pairs": hj.size}
    dump(output / "summary.json", summary)
    dump(output / "completion.json", {"status": "complete", "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
