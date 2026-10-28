"""Règles de l'expérience : Blackjack-v1 (Gymnasium), paquet infini."""
import gymnasium as gym
import numpy as np

STAND, HIT = 0, 1                            # actions de Blackjack-v1
P_CARD = np.array([1 / 13] * 9 + [4 / 13])  # cartes 1..10 (as = 1)
SHAPE = (22, 11, 2)                          # tables indexées [p, d, a]


def make_env():
    return gym.make("Blackjack-v1", natural=False, sab=False)


def draw_card(rng):
    return min(int(rng.integers(1, 14)), 10)


def hit(p, a, c):
    # transition de HIT depuis (p, ., a) en tirant c : (bust, p', a')
    h = p - 10 * a + c
    if h > 21:
        return True, None, None
    a2 = int((a == 1 or c == 1) and h + 10 <= 21)
    return False, h + 10 * a2, a2
