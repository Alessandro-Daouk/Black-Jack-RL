"""Résumés statistiques communs aux expériences complémentaires."""

import numpy as np
from scipy.stats import t


def describe(values):
    """IC de Student approximatif de la moyenne, unités indépendantes = graines."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Au moins deux valeurs finies sont requises.")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "min": float(values.min()),
            "max": float(values.max()), "mean_approx_95pct_interval": [mean - half, mean + half]}

