"""Résumés et provenance communs aux expériences complémentaires."""

from importlib.metadata import version
import platform

import numpy as np
from scipy.stats import t

from src.hybrid_agent import ROOT, file_sha256


def describe(values):
    """IC de Student approximatif de la moyenne, unités indépendantes = graines."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Au moins deux valeurs finies sont requises.")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "min": float(values.min()),
            "max": float(values.max()), "mean_approx_95pct_interval": [mean - half, mean + half]}


def provenance():
    paths = sorted((ROOT / "src").rglob("*.py"))
    return {"versions": {"python": platform.python_version(), **{name: version(name) for name in
            ("numpy", "scipy", "scikit-learn", "gymnasium", "joblib")}},
            "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
            "protocol_sha256": file_sha256(ROOT / "docs/protocole_complementaire.md")}
