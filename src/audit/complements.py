"""Compléments après relecture (E8) : planification, Double Q-learning, réglage sans privilège.

Lit la campagne finale sans la modifier.
uv run --locked python -m src.audit.complements --output results/v2/complements
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
from joblib import Parallel, delayed

from src.environment import HIT, SHAPE, STAND, make_env
from src.supervised.evaluate import write_csv
from src.evaluate import PI_STAR, simulate
from src.supervised.train import fingerprint
from src.supervised.values import build_stand_table
from .core import dataset, dump, fit_estimators, interval, new_output, train
from .oracle import Oracle

FINAL = Path("results/v2/final")
GRID = dict(eps_min=[0.05, 0.1, 0.2, 0.4, 0.6, 1.0], omega=[1.0, 0.8, 0.6])  # = src/tune.py
TUNING_SEEDS = [100, 101, 102]  # = src/tune.py
TUNING_DATA_SEEDS = [61000, 61010, 61020]
SIZES = [1000, 5000, 20000, 60000]
BUDGETS = [10000, 40000, 100000]


def card_frequencies(X):
    """Loi des cartes estimée par les cartes visibles : n tirages indépendants, sans biais d'arrêt."""
    counts = np.bincount(X[:, 0], minlength=11)[1:]
    if np.any(counts == 0):
        raise ValueError("Chaque carte doit apparaître au moins une fois.")
    return counts / counts.sum()


def planned_policy(cards, stand_probabilities=None):
    """Politique optimale du modèle estimé ; STAND déduite de la règle du croupier si non fournie."""
    model = Oracle(cards)
    probabilities = model.probabilities if stand_probabilities is None else stand_probabilities
    stand = model.stand_from(probabilities)
    q, _ = model.optimal(stand)
    return model.table(q > stand)


def train_double(seed, checkpoints, evaluate, eps_min=1., omega=1.):
    """Double Q-learning tabulaire, même environnement, même ε et même pas que le standard."""
    if not 0 < eps_min <= 1 or not .5 < omega <= 1:
        raise ValueError("Epsilon ou omega invalide.")
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, 2, *SHAPE))
    N = np.zeros((2, 2, *SHAPE), dtype=np.int64)
    targets, curve = set(checkpoints), []
    with make_env() as env:
        for episode in range(max(checkpoints)):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            epsilon = max(eps_min, .99995 ** episode)
            while True:
                if rng.random() < epsilon:
                    action = int(rng.integers(2))
                else:
                    hit, stand = Q[0][HIT][state] + Q[1][HIT][state], Q[0][STAND][state] + Q[1][STAND][state]
                    action = HIT if hit > stand else STAND
                nxt, reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise RuntimeError("Troncature inattendue.")
                k = int(rng.random() < .5)
                A, B, n = Q[k], Q[1 - k], N[k]
                y = reward
                if not terminated:
                    best = HIT if A[HIT][nxt] > A[STAND][nxt] else STAND
                    y += B[best][nxt]
                n[action][state] += 1
                A[action][state] += (y - A[action][state]) / n[action][state] ** omega
                if terminated:
                    break
                state = nxt
            if episode + 1 in targets:
                total = Q[0] + Q[1]
                curve.append({"episodes": episode + 1, "j": evaluate(total[HIT] > total[STAND])})
    return curve


def final_points():
    """J à chaque budget de BUDGETS pour le standard et l'hybride logistique à 20 000 mains."""
    points = {}
    with (FINAL / "curves.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method = {("standard", "0"): "standard", ("logistic", "20000"): "hybrid"}.get((row["method"], row["n"]))
            if method and int(row["episodes"]) in BUDGETS:
                points[method, int(row["group"]), int(row["rl_rep"]), int(row["episodes"])] = float(row["j"])
    return points


def grouped(values, groups):
    """Moyenne des graines RL par jeu supervisé, puis intervalle entre jeux."""
    return interval([np.mean([v for (g, _), v in values.items() if g == i]) for i in groups])


def tune(kind, oracle, stands):
    jobs = [(p, s) for p in (dict(zip(GRID, v)) for v in itertools.product(*GRID.values())) for s in TUNING_SEEDS]
    if kind == "double":
        results = Parallel(n_jobs=-1)(delayed(train_double)(s, [100000], oracle.evaluate, **p) for p, s in jobs)
        regrets = [oracle.j_star - r[-1]["j"] for r in results]
    else:
        results = Parallel(n_jobs=-1)(delayed(train)(s, stands[s], [100000], oracle.evaluate, **p) for p, s in jobs)
        regrets = [oracle.j_star - r["curve"][-1]["j"] for r in results]
    rows = [{"agent": kind, **p, "seed": s, "regret": r} for (p, s), r in zip(jobs, regrets, strict=True)]
    means = {}
    for row in rows:
        means.setdefault((row["eps_min"], row["omega"]), []).append(row["regret"])
    best = min(means, key=lambda key: np.mean(means[key]))
    return rows, {"eps_min": best[0], "omega": best[1], "mean_regret": float(np.mean(means[best]))}


def run(output):
    config = json.loads((FINAL / "config.json").read_text(encoding="utf-8"))
    output = new_output(output, {"grid": GRID, "tuning_seeds": TUNING_SEEDS,
                                 "tuning_data_seeds": TUNING_DATA_SEEDS})
    started = time.perf_counter()
    oracle = Oracle()
    groups = range(config["groups"])
    secondary = range(config["secondary_groups"])
    summary = {}

    # A1 : planification, sur les mêmes jeux et les mêmes probabilités logistiques que la campagne finale.
    planning = []
    for i in groups:
        base = config["data_seed_base"] + 10 * i
        X, y, _, _ = dataset(max(SIZES) if i in secondary else 20000, base)
        for n in SIZES if i in secondary else [20000]:
            fit = json.loads((FINAL / f"fits/group_{i}_n_{n}.json").read_text(encoding="utf-8"))
            if fingerprint(X[:n], y[:n]) != fit["train_sha256"]:
                raise ValueError(f"Jeu régénéré différent de la campagne finale : groupe {i}, n={n}")
            cards = card_frequencies(X[:n])
            for method, stand in (("planning_stand_estimated", np.array(fit["probabilities"]["logistic"])),
                                  ("planning_full", None)):
                j = oracle.evaluate(planned_policy(cards, stand))
                planning.append({"group": i, "n": n, "method": method, "j": j, "regret": oracle.j_star - j,
                                 "max_card_error": float(np.max(np.abs(cards - oracle.cards)))})
    write_csv(output / "planning.csv", planning)
    points = final_points()
    hybrid = {g: np.mean([points["hybrid", g, r, 100000] for r in (0, 1)]) for g in groups}
    for method in ("planning_stand_estimated", "planning_full"):
        rows = [r for r in planning if r["method"] == method]
        summary[method] = {
            "regret_by_size_secondary": {n: interval([r["regret"] for r in rows if r["n"] == n and r["group"] in secondary])
                                         for n in SIZES},
            "regret_20000": interval([r["regret"] for r in rows if r["n"] == 20000]),
            "minus_hybrid_20000_100000": interval([r["j"] - hybrid[r["group"]] for r in rows if r["n"] == 20000])}

    # A2 : Double Q-learning, réglé comme le standard puis entraîné sur les graines finales.
    tuning_rows, summary["double_q_tuning"] = tune("double", oracle, None)
    choice = summary["double_q_tuning"]
    jobs = [(i, j) for i in groups for j in (0, 1)]
    curves = Parallel(n_jobs=-1)(delayed(train_double)(config["rl_seed_base"] + 10 * i + j, BUDGETS, oracle.evaluate,
                                                       choice["eps_min"], choice["omega"]) for i, j in jobs)
    double = [{"group": i, "rl_rep": j, "eps_min": choice["eps_min"], "omega": choice["omega"], **p}
              for (i, j), curve in zip(jobs, curves, strict=True) for p in curve]
    write_csv(output / "double_q.csv", double)
    dq = {(r["group"], r["rl_rep"], r["episodes"]): r["j"] for r in double}
    summary["double_q"] = {str(b): {
        "double_minus_standard": grouped({(i, j): dq[i, j, b] - points["standard", i, j, b] for i, j in jobs}, groups),
        "hybrid_minus_double": grouped({(i, j): points["hybrid", i, j, b] - dq[i, j, b] for i, j in jobs}, groups),
        "double_regret": grouped({(i, j): oracle.j_star - dq[i, j, b] for i, j in jobs}, groups)} for b in BUDGETS}

    # A3 : réglage de l'hybride avec un modèle logistique estimé au lieu de la loi exacte.
    stands = {}
    for seed, data_seed in zip(TUNING_SEEDS, TUNING_DATA_SEEDS, strict=True):
        X, y, _, _ = dataset(20000, data_seed)
        Xv, yv, _, _ = dataset(config["validation_size"], data_seed + 1)
        stands[seed] = fit_estimators(X, y, Xv, yv)[1]["logistic"]
    hybrid_rows, choice = tune("hybrid", oracle, stands)
    summary["hybrid_tuning_estimated"] = {**choice, "historical": {"eps_min": 1.0, "omega": 1.0},
                                          "changed": (choice["eps_min"], choice["omega"]) != (1.0, 1.0)}
    write_csv(output / "tuning.csv", tuning_rows + hybrid_rows)
    if summary["hybrid_tuning_estimated"]["changed"]:
        tables = {i: build_stand_table(np.array(json.loads(
            (FINAL / f"fits/group_{i}_n_20000.json").read_text(encoding="utf-8"))["probabilities"]["logistic"]))
            for i in groups}
        results = Parallel(n_jobs=-1)(delayed(train)(config["rl_seed_base"] + 10 * i + j, tables[i], [100000],
                                                     oracle.evaluate, choice["eps_min"], choice["omega"])
                                      for i, j in jobs)
        retuned = {(i, j): r["curve"][-1]["j"] for (i, j), r in zip(jobs, results, strict=True)}
        summary["hybrid_retuned_minus_standard_100000"] = grouped(
            {k: v - points["standard", *k, 100000] for k, v in retuned.items()}, groups)

    # A4 : repères pour le seuil de 0,001, regret d'une seule décision différente de pi*.
    occupancy = oracle.occupation(oracle.policy)
    single = []
    for k, state in enumerate(oracle.states):
        flipped = oracle.policy.copy()
        flipped[k] = not flipped[k]
        single.append({"state": state, "regret": oracle.j_star - oracle.evaluate(flipped),
                       "occupancy": float(occupancy[k])})
    regrets = np.array([s["regret"] for s in single])
    worst = sorted(single, key=lambda s: -s["regret"])[:5]
    n_games, (_, half) = 100000, simulate(PI_STAR, 100000, 2028)
    sd = half * np.sqrt(n_games) / 1.96  # écart-type du gain d'une partie sous pi*
    summary["threshold"] = {"practical_effect": config["practical_effect"],
                            "share_of_house_edge": config["practical_effect"] / -oracle.j_star,
                            "single_state_errors_at_least_threshold": int(np.count_nonzero(regrets >= config["practical_effect"])),
                            "single_state_median_regret": float(np.median(regrets)),
                            "worst_single_state_errors": [{**s, "state": [int(v) for v in s["state"]]} for s in worst],
                            "game_sd": float(sd),
            "games_for_half_width_threshold": float((1.96 * sd / config["practical_effect"]) ** 2)}

    dump(output / "summary.json", summary)
    dump(output / "completion.json", {"status": "complete", "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
