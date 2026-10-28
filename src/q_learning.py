"""Modèle 2 : Q-learning standard et hybride (Q_HIT appris, V_STAND fourni).

Une seule table Q[action, p, d, a] pour les deux agents : pour l'hybride, Q[STAND] est fixé
à V_STAND et jamais mis à jour, ce qui donne exactement la cible et la politique hybrides.
"""
import time

import numpy as np

from src.environment import HIT, SHAPE, STAND, make_env
from src.evaluate import V_TRUE, J

CONFIG = dict(
    episodes=100_000,
    eps0=1.0,
    eps_min=0.05,
    decay=0.99995,
    gamma=1.0,
    omega=1.0,  # pas alpha = 1 / N^omega ; omega < 1 oublie plus vite les premières cibles
    eval_every=2_000,
    seeds=[0, 1, 2, 3, 4],
)

# réglés par src/tune.py sur les graines 100-102 ; omega = 1 (alpha = 1/N) reste le meilleur pour les deux
TUNED = dict(standard={**CONFIG, "eps_min": 0.4}, hybrid={**CONFIG, "eps_min": 1.0})


def greedy_policy(Q):
    return Q[HIT] > Q[STAND]


def choose_action(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(2))
    return HIT if Q[HIT][s] > Q[STAND][s] else STAND


def update(Q, N, s, act, r, s2, term, gamma=1.0, omega=1.0):
    # alpha = 1/N(s, a)^omega ; aucune valeur future si la manche est terminée
    y = r if term else r + gamma * max(Q[HIT][s2], Q[STAND][s2])
    N[act][s] += 1
    Q[act][s] += (y - Q[act][s]) / N[act][s] ** omega


def train(agent, seed, V=V_TRUE, cfg=CONFIG):
    """agent : "standard" ou "hybrid". V : V_STAND donnée à l'hybride (vraie ou estimée)."""
    rng = np.random.default_rng(seed)
    env = make_env()
    Q = np.zeros((2, *SHAPE))
    N = np.zeros((2, *SHAPE), dtype=np.int64)
    if agent == "hybrid":
        Q[STAND] = V
    curve, t0 = [], time.perf_counter()

    for k in range(cfg["episodes"]):
        eps = max(cfg["eps_min"], cfg["eps0"] * cfg["decay"] ** k)
        s, _ = env.reset(seed=seed if k == 0 else None)
        while True:
            act = choose_action(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(act)
            if not (agent == "hybrid" and act == STAND):
                update(Q, N, s, act, r, s2, term, cfg["gamma"], cfg["omega"])
            if term or trunc:
                break
            s = s2
        if (k + 1) % cfg["eval_every"] == 0:
            curve.append(J(greedy_policy(Q)))

    return dict(Q=Q, N=N, curve=np.array(curve), seconds=time.perf_counter() - t0)
