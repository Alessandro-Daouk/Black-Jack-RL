"""Référence exacte : Q*_HIT, politique optimale, gain moyen exact J(pi) et simulation."""
import numpy as np

from src.environment import HIT, P_CARD, SHAPE, STAND, hit, make_env
from src.stand_value import exact_dealer_probs, stand_table

V_TRUE = stand_table(exact_dealer_probs())

# états parcourus par total dur décroissant : les successeurs de HIT sont calculés avant
STATES = [(p, d, a) for a in (0, 1) for p in range(12 if a else 4, 22) for d in range(1, 11)]
STATES.sort(key=lambda s: s[0] - 10 * s[2], reverse=True)
REACH = np.zeros(SHAPE, dtype=bool)
for _s in STATES:
    REACH[_s] = True

MU0 = np.zeros(SHAPE)  # loi de l'état initial : deux cartes joueur + carte visible
for c1 in range(1, 11):
    for c2 in range(1, 11):
        h, a = c1 + c2, int((c1 == 1 or c2 == 1) and c1 + c2 + 10 <= 21)
        MU0[h + 10 * a, 1:, a] += P_CARD[c1 - 1] * P_CARD[c2 - 1] * P_CARD


def solve(V, policy=None):
    # policy=None : (Q*_HIT, V*). Sinon (Q^pi(., HIT), V^pi) pour un tableau booléen « tirer ».
    Q, W = np.zeros(SHAPE), np.zeros(SHAPE)
    for p, d, a in STATES:
        q = 0.0
        for c in range(1, 11):
            bust, p2, a2 = hit(p, a, c)
            q += P_CARD[c - 1] * (-1 if bust else W[p2, d, a2])
        Q[p, d, a] = q
        W[p, d, a] = max(q, V[p, d, a]) if policy is None else (q if policy[p, d, a] else V[p, d, a])
    return Q, W


def J(policy):
    # gain moyen exact par manche, toujours avec la vraie V_STAND
    return float((MU0 * solve(V_TRUE, policy)[1]).sum())


Q_STAR = solve(V_TRUE)[0]
PI_STAR = Q_STAR > V_TRUE
J_STAR = J(PI_STAR)


def simulate(policy, n, seed):
    # gain moyen sur n manches Blackjack-v1 et demi-largeur de l'IC à 95 %
    env = make_env()
    gains = np.empty(n)
    for i in range(n):
        s, _ = env.reset(seed=seed if i == 0 else None)
        term = trunc = False
        while not (term or trunc):
            s, r, term, trunc, _ = env.step(HIT if policy[s] else STAND)
        gains[i] = r
    return gains.mean(), 1.96 * gains.std() / np.sqrt(n)
