"""Distribution du croupier (exacte ou estimée) et valeur de STAND."""
import numpy as np

from src.environment import P_CARD, SHAPE, draw_card

OUTCOMES = [17, 18, 19, 20, 21, "BUST"]  # ordre des classes du modèle 1


def dealer_dist(h, u):
    # loi du résultat final depuis le total dur h (u=1 si as) ; reste sur 17 soft
    t = h + 10 if u and h + 10 <= 21 else h
    out = np.zeros(6)
    if 17 <= t <= 21:
        out[t - 17] = 1
    elif t > 21:
        out[5] = 1
    else:
        for c in range(1, 11):
            out += P_CARD[c - 1] * dealer_dist(h + c, max(u, int(c == 1)))
    return out


def exact_dealer_probs():
    return {d: dealer_dist(d, int(d == 1)) for d in range(1, 11)}


def play_dealer(d, rng):
    # indice dans OUTCOMES du résultat d'une main simulée montrant d
    h, u = d, d == 1
    while True:
        t = h + 10 if u and h + 10 <= 21 else h
        if t >= 17:
            return 5 if t > 21 else t - 17
        c = draw_card(rng)
        h, u = h + c, u or c == 1


def empirical_dealer_probs(n, rng):
    # fréquences sur n mains simulées, carte visible tirée au hasard (comme le dataset du modèle 1)
    counts = np.zeros((11, 6))
    for _ in range(n):
        d = draw_card(rng)
        counts[d, play_dealer(d, rng)] += 1
    # shortcut: carte visible jamais tirée -> loi uniforme ; ne survient en pratique que pour n < 100
    return {d: counts[d] / counts[d].sum() if counts[d].sum() else np.full(6, 1 / 6) for d in range(1, 11)}


def tv_distance(probs, ref):
    # epsilon_TV = max_d ||probs(.|d) - ref(.|d)||_TV
    return max(0.5 * np.abs(probs[d] - ref[d]).sum() for d in range(1, 11))


def reward_vector(p):
    # R(p, D) pour chaque classe de OUTCOMES
    return np.array([1 if D < p else 0 if D == p else -1 for D in OUTCOMES[:5]] + [1])


def stand_table(dealer_probs):
    # V_STAND[p, d, a] = sum_D P(D | d) R(p, D)
    V = np.zeros(SHAPE)
    for p in range(4, 22):
        for d in range(1, 11):
            V[p, d, :] = dealer_probs[d] @ reward_vector(p)
    return V
