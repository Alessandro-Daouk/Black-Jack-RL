"""Oracle indépendant : mains explicites, propagation de masse et matrice absorbante.

N'importe ni les transitions, ni STAND, ni l'évaluateur historiques. Destiné à
l'évaluation et aux contrôles ; jamais appelé par une mise à jour d'apprentissage.
"""
from collections import deque

import numpy as np
from scipy.linalg import solve_triangular

CARDS = np.arange(1, 11)
DEFAULT_CARDS = np.array([1 / 13] * 9 + [4 / 13])


def observation(hand, dealer):
    hard = sum(hand)
    usable = int(1 in hand and hard <= 11)
    return hard + 10 * usable, dealer, usable


def representatives():
    """Une main réelle pour chaque état, atteignabilité obtenue par exploration."""
    hands = {}
    pending = deque()
    for a in CARDS:
        for b in CARDS:
            hand = (int(a), int(b))
            key = observation(hand, 1)
            if key not in hands:
                hands[key] = hand
                pending.append(key)
    while pending:
        key = pending.popleft()
        for c in CARDS:
            hand = hands[key] + (int(c),)
            nxt = observation(hand, 1)
            if nxt[0] <= 21 and nxt not in hands:
                hands[nxt] = hand
                pending.append(nxt)
    return {(p, d, a): hand for (p, _, a), hand in hands.items() for d in CARDS}


def dealer_distribution(card_probabilities=DEFAULT_CARDS):
    """Propager la masse depuis deux cartes jusqu'à absorption, sans récursion."""
    probabilities = np.asarray(card_probabilities, dtype=float)
    if probabilities.shape != (10,) or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Dix probabilités strictement positives, de somme 1, requises.")
    rows = np.zeros((10, 6))
    for d in CARDS:
        mass = np.zeros((32, 2))
        for c, weight in zip(CARDS, probabilities, strict=True):
            mass[d + c, int(d == 1 or c == 1)] += weight
        for h in range(2, 32):
            for has_ace in (0, 1):
                weight = mass[h, has_ace]
                total = h + 10 if has_ace and h <= 11 else h
                if total >= 17:
                    rows[d - 1, min(total, 22) - 17] += weight
                else:
                    for c, prob in zip(CARDS, probabilities, strict=True):
                        mass[h + c, int(has_ace or c == 1)] += weight * prob
    return rows


class Oracle:
    def __init__(self, card_probabilities=DEFAULT_CARDS):
        self.cards = np.asarray(card_probabilities, dtype=float)
        self.probabilities = dealer_distribution(self.cards)
        self.hands = representatives()
        self.states = sorted(self.hands, key=lambda s: (s[0] - 10 * s[2], s[1], s[2]))
        self.index = {s: i for i, s in enumerate(self.states)}
        n = len(self.states)
        self.transition = np.zeros((n, n))
        self.bust = np.zeros(n)
        self.initial = np.zeros(n)
        for i, s in enumerate(self.states):
            for c, weight in zip(CARDS, self.cards, strict=True):
                nxt = observation(self.hands[s] + (int(c),), s[1])
                if nxt[0] > 21:
                    self.bust[i] += weight
                else:
                    self.transition[i, self.index[nxt]] += weight
        for a, pa in zip(CARDS, self.cards, strict=True):
            for b, pb in zip(CARDS, self.cards, strict=True):
                for d, pd in zip(CARDS, self.cards, strict=True):
                    self.initial[self.index[observation((a, b), d)]] += pa * pb * pd
        self.stand = self.stand_from(self.probabilities)
        self.q_hit, self.value = self.optimal(self.stand)
        self.policy = self.q_hit > self.stand
        self.j_star = float(self.initial @ self.value)

    def vector(self, table):
        a = np.asarray(table)
        if a.shape == (len(self.states),):
            return a
        if a.shape != (22, 11, 2):
            raise ValueError("Table (22,11,2) ou vecteur des états requis.")
        return np.array([a[s] for s in self.states])

    def table(self, vector):
        table = np.zeros((22, 11, 2), dtype=np.asarray(vector).dtype)
        for s, v in zip(self.states, vector, strict=True):
            table[s] = v
        return table

    def stand_from(self, probabilities):
        return np.array([np.asarray(probabilities)[d - 1] @
                         np.array([np.sign(p - k) if k != 22 else 1 for k in range(17, 23)])
                         for p, d, _ in self.states])

    def optimal(self, stand):
        stand = self.vector(stand)
        v, q = np.zeros(len(self.states)), np.zeros(len(self.states))
        for i in reversed(range(len(v))):
            q[i] = -self.bust[i] + self.transition[i] @ v
            v[i] = max(q[i], stand[i])
        return q, v

    def evaluate(self, policy, stand=None):
        """Politique = probabilité de HIT ; accepte aussi une politique déterministe."""
        p = self.vector(policy).astype(float)
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Politique invalide.")
        vstand = self.stand if stand is None else self.vector(stand)
        values = solve_triangular(np.eye(len(p)) - p[:, None] * self.transition,
                                  (1 - p) * vstand - p * self.bust, lower=False)
        return float(self.initial @ values)

    def occupation(self, policy):
        p = self.vector(policy).astype(float)
        return solve_triangular((np.eye(len(p)) - p[:, None] * self.transition).T,
                                self.initial, lower=True)

    def diagnostics(self, probabilities, policy=None):
        v = self.stand_from(probabilities)
        q, _ = self.optimal(v)
        limit = q > v
        pi = limit if policy is None else self.vector(policy).astype(bool)
        eta = float(np.max(np.abs(v - self.stand)))
        margin = np.abs(self.q_hit - self.stand)
        occupancy = self.occupation(pi)
        disadvantage = self.value - np.where(pi, self.q_hit, self.stand)
        local = np.abs(v - self.stand)
        # Majorant local de l'erreur optimale : chaque politique s'arrête au plus une fois.
        b_hit = np.zeros(len(v))
        b = np.zeros(len(v))
        for i in reversed(range(len(v))):
            b_hit[i] = self.transition[i] @ b
            b[i] = max(local[i], b_hit[i])
        ambiguous = margin <= local + b_hit + 1e-14
        return {"eta": eta, "max_hit_error_limit": float(np.max(np.abs(q-self.q_hit))),
                "max_tv": float(np.max(np.abs(probabilities-self.probabilities).sum(axis=1)/2)),
                "limit_j": self.evaluate(limit), "j": self.evaluate(pi),
                "regret": self.j_star-self.evaluate(pi), "global_limit_regret_bound": 2*eta,
                "local_limit_regret_bound": float(self.occupation(limit) @ (margin*ambiguous)),
                "occupancy_regret_identity": float(occupancy @ disadvantage),
                "occupation_action_difference": float(occupancy @ (pi != self.policy)),
                "uniform_action_differences": int(np.count_nonzero(pi != self.policy)),
                "expected_decisions": float(occupancy.sum()),
                "ambiguous_states_global": int(np.count_nonzero(margin <= 2*eta)),
                "ambiguous_states_local": int(np.count_nonzero(ambiguous))}
