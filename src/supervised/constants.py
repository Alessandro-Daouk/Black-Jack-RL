"""Conventions du modèle supervisé et de l'adaptateur pour Wassim."""

import numpy as np

BUST = 22
CLASSES = (17, 18, 19, 20, 21, BUST)
OUTCOME_NAMES = ("17", "18", "19", "20", "21", "BUST")
CARDS = np.arange(1, 11)
DECK = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
CARD_PROBABILITIES = np.array([1 / 13] * 9 + [4 / 13])
STAND_SHAPE = (22, 11, 2)
