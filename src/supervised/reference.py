"""Référence récursive pour l'évaluation ; jamais utilisée comme cible d'entraînement."""

from functools import lru_cache

import numpy as np

from .constants import CARD_PROBABILITIES


@lru_cache(maxsize=None)
def _terminal_distribution(hard_total: int, has_ace: bool) -> tuple[float, ...]:
    total = hard_total + 10 if has_ace and hard_total + 10 <= 21 else hard_total
    probabilities = np.zeros(6)
    if total >= 17:
        probabilities[5 if total > 21 else total - 17] = 1.0
    else:
        for card, probability in enumerate(CARD_PROBABILITIES, start=1):
            probabilities += probability * np.asarray(
                _terminal_distribution(hard_total + card, has_ace or card == 1)
            )
    return tuple(float(value) for value in probabilities)


def exact_probabilities() -> np.ndarray:
    """Lignes : cartes 1..10 ; colonnes : 17,18,19,20,21,BUST.

    La carte cachée est toujours distribuée. Tous les tirages ont leur poids
    réel 1/13 ou 4/13 ; il n'y a pas de conditionnement sur l'absence de blackjack.
    """
    rows = []
    for visible in range(1, 11):
        probabilities = np.zeros(6)
        for hidden, weight in enumerate(CARD_PROBABILITIES, start=1):
            probabilities += weight * np.asarray(
                _terminal_distribution(visible + hidden, visible == 1 or hidden == 1)
            )
        rows.append(probabilities)
    return np.array(rows)
