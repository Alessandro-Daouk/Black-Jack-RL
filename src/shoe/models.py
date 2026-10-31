"""Modèle 1 en sabot fini : loi de l'issue du croupier sachant carte visible et cartes non vues.

Quatre estimateurs, du plus pauvre au plus riche en information :
- carte : fréquences par carte visible (le modèle 1 du paquet infini, ignore le sabot) ;
- table : fréquences par carte visible et classe de compte (l'information de l'état RL) ;
- logistique : régression multinomiale sur la composition exacte, interactions avec la carte ;
- boosting : arbres de gradient sur la carte et la composition, sans forme imposée.
Les deux tables sont lissées (Jeffreys, α = 1/2) pour garder une log-loss finie.
"""
from functools import lru_cache
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .game import DECK, dealer_dataset, true_count_bucket

C_GRID = (.01, .1, 1., 10., 100.)
FEATURE_SEED = 150000


def composition(X):
    """[1, proportions des dix cartes non vues, cartes non vues / 52]."""
    u = X[:, 1:]
    return np.column_stack([np.ones(len(X)), u / u.sum(axis=1, keepdims=True), u.sum(axis=1) / 52])


@lru_cache(maxsize=1)
def standardizer():
    """Moyenne et écart-type de la composition sur des sabots entamés au hasard (même loi que le modèle 1)."""
    z = composition(dealer_dataset(20000, FEATURE_SEED)[0])[:, 1:]
    return z.mean(axis=0), z.std(axis=0)


def state_features(unseen):
    """Variables des agents RL linéaires : [1, composition standardisée des cartes non vues]."""
    mu, sd = standardizer()
    return np.concatenate(([1.], (np.append(unseen / unseen.sum(), unseen.sum() / 52) - mu) / sd))


def features(X):
    # interactions carte visible x composition : une régression par carte visible, ajustées ensemble
    onehot = np.eye(10)[X[:, 0] - 1]
    return (onehot[:, :, None] * composition(X)[:, None, :]).reshape(len(X), -1)


def buckets(X):
    return np.array([true_count_bucket(x[1:], DECK) for x in X])


def jeffreys(index, y, shape):
    counts = np.zeros((*shape, 6))
    np.add.at(counts, (*index, y - 17), 1)
    return (counts + .5) / (counts.sum(axis=-1, keepdims=True) + 3)


class Upcard:
    name = "carte"

    def __init__(self, X, y, Xv=None, yv=None):
        self.p = jeffreys((X[:, 0] - 1,), y, (10,))

    def proba(self, X):
        return self.p[X[:, 0] - 1]

    def one(self, up, unseen, bucket):
        return self.p[up - 1]


class Table(Upcard):
    name = "table"

    def __init__(self, X, y, Xv=None, yv=None):
        self.p = jeffreys((X[:, 0] - 1, buckets(X)), y, (10, 5))

    def proba(self, X):
        return self.p[X[:, 0] - 1, buckets(X)]

    def one(self, up, unseen, bucket):
        return self.p[up - 1, bucket]


class Logistic:
    name = "logistique"

    def __init__(self, X, y, Xv, yv):
        F, Fv = features(X), features(Xv)
        best = None
        for c in C_GRID:
            model = LogisticRegression(C=c, max_iter=5000, tol=1e-8)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(F, y)
            loss = -np.log(model.predict_proba(Fv)[np.arange(len(yv)), yv - 17]).mean()
            if best is None or loss < best[0]:
                best = loss, model, c
        self.C, model = best[2], best[1]
        if not np.array_equal(model.classes_, np.arange(17, 23)):
            raise ValueError("Les six issues doivent apparaître à l'entraînement.")
        self.model = model
        self.W = model.coef_.reshape(6, 10, -1)  # (issue, carte visible, composante)
        self.b = model.intercept_

    def proba(self, X):
        return self.model.predict_proba(features(X))

    def one(self, up, unseen, bucket):
        # même calcul que predict_proba, sans le coût d'appel de scikit-learn à chaque décision
        z = np.concatenate(([1.], unseen / unseen.sum(), [unseen.sum() / 52]))
        logits = self.W[:, up - 1] @ z + self.b
        e = np.exp(logits - logits.max())
        return e / e.sum()


class Boosting:
    name = "boosting"
    max_iter = 500  # plafond ; le nombre retenu est choisi sur validation

    def __init__(self, X, y, Xv, yv):
        # nombre d'arbres choisi sur la même validation que C (scikit-learn n'accepte pas de validation externe)
        best = None
        model = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=.05, categorical_features=[0],
                                               early_stopping=False, random_state=0)
        model.fit(self._x(X), y)
        for k, p in enumerate(model.staged_predict_proba(self._x(Xv)), start=1):
            loss = -np.log(p[np.arange(len(yv)), yv - 17]).mean()
            if best is None or loss < best[0]:
                best = loss, k
        self.iterations = best[1]
        self.model = HistGradientBoostingClassifier(max_iter=best[1], learning_rate=.05, categorical_features=[0],
                                                    early_stopping=False, random_state=0).fit(self._x(X), y)

    @staticmethod
    def _x(X):
        return np.column_stack([X[:, 0], composition(X)[:, 1:]])

    def proba(self, X):
        return self.model.predict_proba(self._x(X))

    def one(self, up, unseen, bucket):
        return self.proba(np.concatenate(([up], unseen))[None])[0]


MODELS = {"carte": Upcard, "table": Table, "logistique": Logistic, "boosting": Boosting}


def fit(name, X, y, Xv, yv):
    start = time.perf_counter()
    model = MODELS[name](X, y, Xv, yv)
    return model, time.perf_counter() - start


def scores(model, X, y, exact):
    """Log-loss test, excès exact (KL moyen vers la loi vraie) et erreur moyenne de V_STAND (totaux 12 à 21)."""
    p = np.clip(model.proba(X), 1e-15, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(exact > 0, exact * (np.log(exact) - np.log(p)), 0).sum(axis=1)
    stand_error = np.abs((p - exact) @ np.array([[1 if k == 22 else np.sign(t - k) for k in range(17, 23)]
                                                for t in range(12, 22)]).T)
    return {"log_loss": float(-np.log(p[np.arange(len(y)), y - 17]).mean()), "excess_kl": float(kl.mean()),
            "stand_mae": float(stand_error.mean()), "stand_max": float(stand_error.max())}
