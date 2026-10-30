"""Valeur de STAND et conversion vers les tableaux du Q-learning (src/q_learning.py)."""

from collections.abc import Mapping

import numpy as np

from .constants import BUST, CLASSES, STAND_SHAPE
from .data import require_integer


def validate_probabilities(probabilities: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != shape:
        raise ValueError(f"Forme attendue : {shape}, reçue : {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Les probabilités doivent être finies et comprises entre 0 et 1.")
    if not np.allclose(values.sum(axis=-1), 1, atol=1e-8, rtol=0):
        raise ValueError("Les probabilités doivent sommer à 1.")
    return values


def reward_vector(player_sum: int) -> np.ndarray:
    player_sum = require_integer(player_sum, "player_sum", 4)
    if player_sum > 21:
        return -np.ones(6)
    return np.array([1 if k == BUST else np.sign(player_sum - k) for k in CLASSES], dtype=float)


def stand_value(player_sum: int, probabilities: Mapping[int, float]) -> float:
    if set(probabilities) != set(CLASSES):
        raise ValueError("Les six classes 17, 18, 19, 20, 21, 22 sont requises.")
    values = validate_probabilities([probabilities[k] for k in CLASSES], (6,))
    return float(values @ reward_vector(player_sum))


def build_stand_table(probabilities: np.ndarray) -> np.ndarray:
    """Table V[p, d, a], compatible avec src.q_learning.train(..., V=V).

    Les indices p<4 et d=0, inaccessibles dans le jeu, restent à zéro comme
    dans src/q_learning.py. Pour p=4..21 et d=1..10, les deux valeurs de a sont identiques.
    """
    probabilities = validate_probabilities(probabilities, (10, 6))
    table = np.zeros(STAND_SHAPE)
    for player_sum in range(4, 22):
        table[player_sum, 1:, :] = (probabilities @ reward_vector(player_sum))[:, None]
    return table
