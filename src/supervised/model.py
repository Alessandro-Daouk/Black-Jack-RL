"""Apprentissage, export et chargement du prédicteur supervisé."""

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .constants import CARDS, CLASSES
from .data import require_integer, validate_dataset
from .values import build_stand_table, validate_probabilities

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models/supervised/dealer_model.joblib"


def fit_classifier(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Pipeline:
    validate_dataset(X, y)
    if not np.array_equal(np.unique(y), CLASSES) or not np.array_equal(np.unique(X), CARDS):
        raise ValueError("L'entraînement doit couvrir les dix cartes et les six classes.")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C doit être strictement positif et fini.")
    pipeline = Pipeline([
        ("one_hot", OneHotEncoder(categories=[CARDS], sparse_output=False, handle_unknown="error")),
        ("classifier", LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-9)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, y)
    return pipeline


@dataclass
class DealerPredictor:
    pipeline: Pipeline
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.array_equal(self.pipeline.classes_, CLASSES):
            raise ValueError("Ordre des classes incompatible : 17,18,19,20,21,22 requis.")
        self.probability_matrix()

    def probability_matrix(self) -> np.ndarray:
        return validate_probabilities(self.pipeline.predict_proba(CARDS[:, None]), (10, 6))

    def predict_dealer_probabilities(self, dealer_showing: int) -> dict[int, float]:
        dealer_showing = require_integer(dealer_showing, "dealer_showing", 1, 10)
        probabilities = self.pipeline.predict_proba([[dealer_showing]])[0]
        validate_probabilities(probabilities, (6,))
        return dict(zip(CLASSES, map(float, probabilities), strict=True))

    def probability_vectors(self) -> dict[int, np.ndarray]:
        """Vecteurs ordonnés [17,18,19,20,21,BUST] attendus par stand_table()."""
        return {d: row.copy() for d, row in enumerate(self.probability_matrix(), start=1)}

    def stand_values(self) -> np.ndarray:
        return build_stand_table(self.probability_matrix())

    def save(self, path: str | Path = DEFAULT_MODEL) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": 1, "pipeline": self.pipeline, "metadata": self.metadata}, path)


def load_model(path: str | Path = DEFAULT_MODEL) -> DealerPredictor:
    """Charger une fois un fichier joblib produit par ce projet et de confiance."""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Format de modèle inconnu.")
    return DealerPredictor(payload["pipeline"], payload["metadata"])
