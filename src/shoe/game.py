"""Sabot fini : un jeu de 52 cartes, rebattu avant une manche s'il en reste moins de 20.

Mêmes règles que Blackjack-v1 (natural=False, croupier reste sur 17 souple, ni double ni
partage, pas de vérification du blackjack du croupier), mais cartes tirées sans remise.
Le joueur voit toutes les cartes sauf la carte cachée du croupier ; « non vues » désigne le
sabot restant plus cette carte cachée. L'état RL ajoute au triplet de Blackjack-v1 une classe
du compte vrai Hi-Lo : c'est une compression de la composition, pas un état markovien.
"""
from functools import lru_cache

import numpy as np

DECK = np.array([4] * 9 + [16])        # effectifs des cartes 1..10 (as = 1) dans un jeu
CARDS = np.repeat(np.arange(1, 11), DECK)
RESHUFFLE = 20
HILO = np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1])
BUCKETS = np.array([-2., -.5, .5, 2.])  # compte vrai -> 5 classes
SHAPE = (22, 11, 2, len(BUCKETS) + 1)
REWARD = np.array([[1 if k == 22 else np.sign(p - k) for k in range(17, 23)] for p in range(22)])


def value(hand):
    total = sum(hand)
    return total + 10 if 1 in hand and total + 10 <= 21 else total


def observation(hand, up, bucket):
    total = sum(hand)
    usable = int(1 in hand and total + 10 <= 21)
    return total + 10 * usable, up, usable, bucket


def true_count_bucket(unseen, full):
    tc = HILO @ (full - unseen) / (unseen.sum() / 52)
    return int(np.searchsorted(BUCKETS, tc, side="right"))


class Shoe:
    def __init__(self, rng):
        self.rng = rng
        self.shuffle()

    def shuffle(self):
        self.cards = list(self.rng.permutation(CARDS))
        self.unseen, self.full = DECK.copy(), DECK.copy()

    def draw(self, seen=True):
        if not self.cards:  # exceptionnel : une manche épuise les 20 cartes, on ajoute un jeu
            self.cards = list(self.rng.permutation(CARDS))
            self.unseen, self.full = self.unseen + DECK, self.full + DECK
        c = int(self.cards.pop())
        if seen:
            self.unseen[c - 1] -= 1
        return c

    def bucket(self):
        return true_count_bucket(self.unseen, self.full)


def play_round(shoe, decide):
    """Une manche ; decide(obs, unseen) -> 0 (STAND) ou 1 (HIT).

    Renvoie la récompense et les décisions [(obs, unseen, action)] ; unseen est une copie.
    """
    if len(shoe.cards) < RESHUFFLE:
        shoe.shuffle()
    p1, up, p2, hole = shoe.draw(), shoe.draw(), shoe.draw(), shoe.draw(seen=False)
    hand, path = [p1, p2], []
    while True:
        obs = observation(hand, up, shoe.bucket())
        unseen = shoe.unseen.copy()
        action = decide(obs, unseen)
        path.append((obs, unseen, action))
        if action == 0:
            break
        hand.append(shoe.draw())
        if value(hand) > 21:
            shoe.unseen[hole - 1] -= 1
            return -1, path
    shoe.unseen[hole - 1] -= 1
    dealer = [up, hole]
    while value(dealer) < 17:
        dealer.append(shoe.draw())
    d, p = value(dealer), value(hand)
    return (1 if d > 21 or p > d else 0 if p == d else -1), path


@lru_cache(maxsize=200_000)
def _dealer(hard, ace, unseen):
    total = hard + 10 if ace and hard + 10 <= 21 else hard
    out = np.zeros(6)
    if total >= 17:
        out[min(total, 22) - 17] = 1
        return out
    n = sum(unseen)
    for c in range(1, 11):
        if unseen[c - 1]:
            rest = list(unseen)
            rest[c - 1] -= 1
            out += unseen[c - 1] / n * _dealer(hard + c, ace or c == 1, tuple(rest))
    return out


def exact_dealer(up, unseen):
    """Loi exacte de l'issue du croupier [17..21, BUST] : carte cachée puis tirages parmi les non vues."""
    return _dealer(int(up), up == 1, tuple(int(k) for k in unseen)).copy()


def dealer_dataset(n, seed):
    """n mains de croupier simulées dans des sabots entamés au hasard, sans jouer de partie.

    k cartes déjà vues (2 à 37, uniforme), puis carte visible, carte cachée et tirages.
    X[:, 0] = carte visible, X[:, 1:] = effectifs des cartes non vues ; y = 17..21 ou 22.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, 11), dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        deck = rng.permutation(CARDS)
        k = int(rng.integers(2, 38))
        X[i, 0] = deck[k]
        X[i, 1:] = DECK - np.bincount(deck[:k + 1], minlength=11)[1:]
        dealer, j = [int(deck[k]), int(deck[k + 1])], k + 2
        while value(dealer) < 17:
            dealer.append(int(deck[j]))
            j += 1
        y[i] = min(value(dealer), 22)
    return X, y


def evaluate(decide, seeds):
    """Gains et nombres de manches par sabot ; mêmes sabots pour toutes les politiques (appariement)."""
    gains, rounds = np.zeros(len(seeds), dtype=np.int16), np.zeros(len(seeds), dtype=np.int16)
    for i, seed in enumerate(seeds):
        shoe = Shoe(np.random.default_rng(seed))
        while len(shoe.cards) >= RESHUFFLE:
            r, _ = play_round(shoe, decide)
            gains[i] += r
            rounds[i] += 1
    return gains, rounds
