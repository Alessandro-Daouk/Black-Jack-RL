"""Simulations indépendantes du croupier : paquet infini, arrêt à 17 souple."""

from numbers import Integral

import numpy as np

from .constants import BUST, CLASSES, DECK


def require_integer(value: int, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} doit être un entier.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} est hors du domaine autorisé.")
    return int(value)


def draw_card(rng: np.random.Generator) -> int:
    return int(rng.choice(DECK))


def hand_value(cards: list[int]) -> int:
    total = sum(cards)
    return total + 10 if 1 in cards and total + 10 <= 21 else total


def finish_dealer(cards: list[int], rng: np.random.Generator) -> int:
    """Termine une main déjà distribuée ; la liste d'entrée est préservée."""
    hand = list(cards)
    while hand_value(hand) < 17:
        hand.append(draw_card(rng))
    total = hand_value(hand)
    return BUST if total > 21 else total


def simulate_dealer(rng: np.random.Generator) -> tuple[int, int]:
    visible, hidden = draw_card(rng), draw_card(rng)
    return visible, finish_dealer([visible, hidden], rng)


def generate_dataset(n_samples: int = 60_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n_samples = require_integer(n_samples, "n_samples", 1)
    seed = require_integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    samples = np.array([simulate_dealer(rng) for _ in range(n_samples)], dtype=np.int64)
    return samples[:, :1], samples[:, 1]


def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[1] != 1 or y.shape != (len(X),) or len(X) == 0:
        raise ValueError("Formes requises : X=(n, 1), y=(n,), avec n > 0.")
    if not np.issubdtype(X.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Les cartes et classes doivent être des entiers.")
    if not np.all((1 <= X) & (X <= 10)) or not np.all(np.isin(y, CLASSES)):
        raise ValueError("Cartes ou classes invalides.")


def conditional_counts(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    validate_dataset(X, y)
    counts = np.zeros((10, 6), dtype=np.int64)
    np.add.at(counts, (X[:, 0] - 1, y - 17), 1)
    return counts


def empirical_probabilities(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fréquences non lissées, calculées uniquement sur les données fournies."""
    counts = conditional_counts(X, y)
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Chaque carte visible doit apparaître dans l'entraînement.")
    return counts / totals
