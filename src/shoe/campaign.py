"""Campagne E10 : le modèle 1 apporte-t-il quelque chose quand il dispose d'une information
que l'état RL n'a pas ?

1. Modèle 1 : quatre estimateurs, quatre tailles, 9 jeux indépendants ; KL exact vers la loi vraie.
2. Réglage (ε_min x ω) de chaque agent sur des graines, données et sabots d'évaluation distincts.
3. Agents : Q-learning standard sur (total, carte, as, classe de compte), et trois hybrides dont
   STAND vient du modèle « carte », « table » ou « logistique ». La stratégie de base du paquet
   infini, sans apprentissage, sert de référence.
Évaluation : Monte-Carlo apparié, mêmes sabots pour toutes les politiques.

uv run --locked python -m src.shoe.campaign --output results/5_sabot_fini
"""
import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import t

from src.audit.core import dump, interval, new_output
from src.audit.oracle import Oracle
from src.supervised.evaluate import write_csv
from .game import REWARD, SHAPE, Shoe, dealer_dataset, evaluate, exact_dealer, play_round
from .models import MODELS, fit, scores

GROUPS = 9
SIZES = [1000, 5000, 20000, 100000]
RL_SIZE = 100000
VALIDATION_SIZE, TEST_SIZE = 10000, 3000
DATA_SEED, VALIDATION_SEED, TEST_SEED, RL_SEED = 120000, 125000, 128000, 130000
TUNING_SEEDS, TUNING_DATA_SEED = [100, 101, 102], 140000
CHECKPOINTS = [10000, 50000, 200000, 1000000, 3000000]
EVAL_SHOES = range(1_000_000, 1_300_000)
TUNING_SHOES = range(2_000_000, 2_020_000)
GRID = dict(eps_min=[.1, .4, 1.], omega=[1., .8])
AGENTS = {"standard": None, "hybride_carte": "carte", "hybride_table": "table",
          "hybride_logistique": "logistique"}


def train(model, seed, checkpoints, shoes, eps_min=1., omega=1.):
    """model=None : Q-learning standard. Sinon hybride : Q_HIT appris, STAND = modèle 1 sur les cartes non vues."""
    rng = np.random.default_rng(seed)
    shoe = Shoe(np.random.default_rng(seed + 1))  # cartes et exploration sur deux flux distincts
    Q = np.zeros((2, *SHAPE))
    N = np.zeros((2, *SHAPE), dtype=np.int64)
    eps, curve, rl_seconds = 1., [], 0.

    def stand(obs, unseen):
        return Q[0][obs] if model is None else model.one(obs[1], unseen, obs[3]) @ REWARD[obs[0]]

    def greedy(obs, unseen):
        return int(Q[1][obs] > stand(obs, unseen))

    def behave(obs, unseen):
        return int(rng.integers(2)) if rng.random() < eps else greedy(obs, unseen)

    start = time.perf_counter()
    for k in range(max(checkpoints)):
        eps = max(eps_min, .99995 ** k)
        r, path = play_round(shoe, behave)
        for t, (obs, unseen, a) in enumerate(path):
            if model is not None and a == 0:
                continue
            y = r if t == len(path) - 1 else max(Q[1][path[t + 1][0]], stand(*path[t + 1][:2]))
            N[a][obs] += 1
            Q[a][obs] += (y - Q[a][obs]) / N[a][obs] ** omega
        if k + 1 in checkpoints:
            rl_seconds += time.perf_counter() - start
            gains, rounds = evaluate(greedy, shoes)
            curve.append({"episodes": k + 1, "rl_seconds": rl_seconds, "gains": gains, "rounds": rounds})
            start = time.perf_counter()
    return curve


def gain(g, n):
    return float(g.sum() / n.sum())


def paired_se(pairs):
    """Erreur-type Monte-Carlo de la moyenne des J(a) − J(b) sur les mêmes sabots.

    Linéarisation des rapports gain/manches, moyennée sur les paires sabot par sabot : les
    paires partagent le jeu d'évaluation, leurs erreurs ne sont donc pas indépendantes.
    """
    def linear(x):
        return (x[0] - gain(*x) * x[1]) / x[1].mean()
    d = np.mean([linear(a) - linear(b) for a, b in pairs], axis=0)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def with_evaluation(values, mc_se):
    """IC entre jeux, puis IC qui ajoute l'erreur du jeu d'évaluation commun (indépendante des jeux)."""
    out = interval(values)
    half = float(t.ppf(.975, out["n"] - 1) * np.sqrt(out["sd"] ** 2 / out["n"] + mc_se ** 2))
    return {**out, "evaluation_mc_se": mc_se, "ci95_with_evaluation": [out["mean"] - half, out["mean"] + half]}


def supervised_group(i):
    start = time.perf_counter()
    X, y = dealer_dataset(max(SIZES), DATA_SEED + 10 * i)  # max(SIZES) = RL_SIZE
    simulation = time.perf_counter() - start
    Xv, yv = dealer_dataset(VALIDATION_SIZE, VALIDATION_SEED + 10 * i)
    Xt, yt = dealer_dataset(TEST_SIZE, TEST_SEED + 10 * i)
    exact = np.array([exact_dealer(x[0], x[1:]) for x in Xt])
    rows, kept = [], {}
    for n, name in itertools.product(SIZES, MODELS):
        model, seconds = fit(name, X[:n], y[:n], Xv, yv)
        rows.append({"group": i, "n": n, "model": name, "fit_seconds": seconds,
                     "selected": getattr(model, "C", getattr(model, "iterations", "")), **scores(model, Xt, yt, exact)})
        if n == RL_SIZE:
            kept[name] = model
    ref = scores(SimpleNamespace(proba=lambda _: exact), Xt, yt, exact)
    rows.append({"group": i, "n": 0, "model": "exacte", "fit_seconds": 0, "selected": "", **ref})
    return rows, kept, simulation


def tuning_models(seed):
    data = dealer_dataset(RL_SIZE, TUNING_DATA_SEED + seed) + dealer_dataset(VALIDATION_SIZE, TUNING_DATA_SEED + seed + 1)
    return {name: fit(name, *data)[0] for name in ("carte", "table", "logistique")}


def tuning_run(name, model, params, seed):
    curve = train(model, seed, [CHECKPOINTS[-1]], TUNING_SHOES, **params)
    return {"agent": name, **params, "seed": seed, "j": gain(curve[-1]["gains"], curve[-1]["rounds"])}


def run(output):
    output = new_output(output, {"groups": GROUPS, "sizes": SIZES, "rl_size": RL_SIZE,
                                 "validation_size": VALIDATION_SIZE, "test_size": TEST_SIZE,
                                 "seeds": {"data": DATA_SEED, "validation": VALIDATION_SEED, "test": TEST_SEED,
                                           "rl": RL_SEED, "tuning": TUNING_SEEDS, "tuning_data": TUNING_DATA_SEED},
                                 "checkpoints": CHECKPOINTS, "eval_shoes": [EVAL_SHOES.start, EVAL_SHOES.stop],
                                 "tuning_shoes": [TUNING_SHOES.start, TUNING_SHOES.stop], "grid": GRID,
                                 "agents": AGENTS})
    started = time.perf_counter()
    summary = {}

    # 1. Modèle 1.
    results = Parallel(n_jobs=GROUPS)(delayed(supervised_group)(i) for i in range(GROUPS))
    rows = [r for res in results for r in res[0]]
    write_csv(output / "supervised.csv", rows)
    models = {i: res[1] for i, res in enumerate(results)}
    summary["simulation_seconds_rl_size"] = interval([res[2] for res in results])
    summary["supervised"] = {f"{name}_{n}": {k: interval([r[k] for r in rows if r["model"] == name and r["n"] == n])
                                             for k in ("log_loss", "excess_kl", "stand_mae", "fit_seconds")}
                             for name in MODELS for n in SIZES}
    summary["supervised"]["exacte"] = {"log_loss": interval([r["log_loss"] for r in rows if r["model"] == "exacte"])}
    for n in SIZES:
        diff = {name: [r["excess_kl"] for r in rows if r["model"] == name and r["n"] == n] for name in MODELS}
        summary["supervised"][f"kl_table_minus_logistique_{n}"] = interval(np.subtract(diff["table"], diff["logistique"]))
        summary["supervised"][f"kl_boosting_minus_logistique_{n}"] = interval(
            np.subtract(diff["boosting"], diff["logistique"]))

    # 2. Réglage sur des données, graines et sabots distincts.
    tuned = dict(zip(TUNING_SEEDS, Parallel(n_jobs=len(TUNING_SEEDS))(delayed(tuning_models)(s) for s in TUNING_SEEDS),
                     strict=True))
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    tuning = Parallel(n_jobs=-1)(delayed(tuning_run)(agent, None if m is None else tuned[s][m], p, s)
                                 for agent, m in AGENTS.items() for p in combos for s in TUNING_SEEDS)
    write_csv(output / "tuning.csv", tuning)
    choice = {}
    for agent in AGENTS:
        means = {(p["eps_min"], p["omega"]): np.mean([t["j"] for t in tuning if t["agent"] == agent
                                                     and (t["eps_min"], t["omega"]) == (p["eps_min"], p["omega"])])
                 for p in combos}
        best = max(means, key=means.get)
        choice[agent] = {"eps_min": best[0], "omega": best[1], "tuning_j": float(means[best])}
    summary["tuning"] = choice

    # 3. Agents finaux et stratégie de base.
    oracle = Oracle()
    basic_table = oracle.table(oracle.policy)
    basic = evaluate(lambda obs, _: int(basic_table[obs[:3]]), EVAL_SHOES)
    jobs = [(agent, i, r) for agent in AGENTS for i in range(GROUPS) for r in (0, 1)]
    curves = Parallel(n_jobs=-1)(delayed(train)(None if m is None else models[i][m], RL_SEED + 10 * i + r,
                                                CHECKPOINTS, EVAL_SHOES, **{k: choice[agent][k] for k in GRID})
                                 for agent, i, r in jobs for m in [AGENTS[agent]])
    runs = {job: curve for job, curve in zip(jobs, curves, strict=True)}
    point = {(a, i, r, b): (p["gains"], p["rounds"]) for (a, i, r), c in runs.items()
             for b, p in zip(CHECKPOINTS, c, strict=True)}
    write_csv(output / "runs.csv", [{"agent": a, "group": i, "rl_rep": r, "episodes": b, "rl_seconds": p["rl_seconds"],
                                     "j": gain(*point[a, i, r, b]), "rounds": int(p["rounds"].sum()),
                                     "minus_basic": gain(*point[a, i, r, b]) - gain(*basic),
                                     "minus_basic_mc_se": paired_se([(point[a, i, r, b], basic)])}
                                    for (a, i, r), c in runs.items() for b, p in zip(CHECKPOINTS, c, strict=True)])
    summary["basic_strategy_j"] = gain(*basic)
    summary["eval_rounds_basic"] = int(basic[1].sum())
    keys = [(i, r) for i in range(GROUPS) for r in (0, 1)]

    def contrast(a, b, e):
        """a, b : agents ou None pour la stratégie de base ; différences moyennées par jeu."""
        get = (lambda x, i, r: basic if x is None else point[x, i, r, e])
        diff = [np.mean([gain(*get(a, i, r)) - gain(*get(b, i, r)) for r in (0, 1)]) for i in range(GROUPS)]
        return {"difference": with_evaluation(diff, paired_se([(get(a, i, r), get(b, i, r)) for i, r in keys])),
                "pairs_better": int(sum(gain(*get(a, i, r)) > gain(*get(b, i, r)) for i, r in keys))}

    summary["agents"] = {agent: {str(e): {
        "j": interval([np.mean([gain(*point[agent, i, r, e]) for r in (0, 1)]) for i in range(GROUPS)]),
        "minus_basic": contrast(agent, None, e)} for e in CHECKPOINTS} for agent in AGENTS}
    summary["contrasts"] = {f"{a}-{b}": {str(e): contrast(a, b, e) for e in CHECKPOINTS}
                            for a, b in (("hybride_logistique", "standard"), ("hybride_logistique", "hybride_table"),
                                         ("hybride_logistique", "hybride_carte"), ("hybride_table", "standard"),
                                         ("hybride_carte", "standard"))}
    summary["rl_seconds_final"] = {a: interval([runs[a, i, r][-1]["rl_seconds"] for i in range(GROUPS)
                                                 for r in (0, 1)]) for a in AGENTS}
    dump(output / "summary.json", summary)
    dump(output / "completion.json", {"status": "complete", "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
