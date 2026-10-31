import numpy as np
import pytest

from src.audit.oracle import Oracle
from src.shoe.campaign import gain, paired_se, train
from src.shoe.game import DECK, Shoe, dealer_dataset, evaluate, exact_dealer, play_round, true_count_bucket
from src.shoe.models import MODELS, Boosting, Logistic, fit, scores, state_features


@pytest.fixture(scope="module")
def data():
    return dealer_dataset(3000, 1), dealer_dataset(1000, 2)


def test_exact_dealer_full_deck_close_to_infinite_deck():
    oracle = Oracle()
    for up in range(1, 11):
        unseen = DECK.copy()
        unseen[up - 1] -= 1
        p = exact_dealer(up, unseen)
        assert p.sum() == pytest.approx(1)
        assert np.max(np.abs(p - oracle.probabilities[up - 1])) < .02


def test_exact_dealer_degenerate_shoe():
    only_tens = np.zeros(10, dtype=int)
    only_tens[9] = 5
    np.testing.assert_allclose(exact_dealer(7, only_tens), [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(exact_dealer(6, only_tens), [0, 0, 0, 0, 0, 1])


def test_exact_dealer_matches_simulated_labels():
    X, y = dealer_dataset(4000, 7)
    P = np.array([exact_dealer(x[0], x[1:]) for x in X])
    d = np.eye(6)[y - 17] - P
    assert np.all(np.abs(d.mean(axis=0)) < 4 * d.std(axis=0) / np.sqrt(len(d)))


def test_dataset_shapes_and_domain(data):
    (X, y), _ = data
    assert X.shape == (3000, 11) and set(np.unique(y)) <= set(range(17, 23))
    assert np.all((X[:, 0] >= 1) & (X[:, 0] <= 10)) and np.all(X[:, 1:] >= 0)
    assert X[:, 1:].sum(axis=1).min() >= 52 - 38 and X[:, 1:].sum(axis=1).max() <= 52 - 3


def test_play_round_keeps_unseen_cards_consistent():
    rng = np.random.default_rng(0)
    shoe = Shoe(np.random.default_rng(1))
    for _ in range(2000):
        r, path = play_round(shoe, lambda obs, unseen: int(rng.integers(2)))
        assert r in (-1, 0, 1) and path
        # en fin de manche la carte cachée est révélée : non vues = sabot restant
        np.testing.assert_array_equal(shoe.unseen, np.bincount(shoe.cards, minlength=11)[1:])


def test_models_valid_and_logistic_fast_path_matches_sklearn(data, monkeypatch):
    monkeypatch.setattr(Boosting, "max_iter", 30)
    (X, y), (Xv, yv) = data
    exact = np.array([exact_dealer(x[0], x[1:]) for x in Xv[:200]])
    for name in MODELS:
        model, _ = fit(name, X, y, Xv, yv)
        p = model.proba(Xv[:50])
        np.testing.assert_allclose(p.sum(axis=1), 1)
        for x, row in zip(Xv[:20], p, strict=False):
            np.testing.assert_allclose(model.one(int(x[0]), x[1:], true_count_bucket(x[1:], DECK)), row, atol=1e-10)
        assert np.isfinite(list(scores(model, Xv[:200], yv[:200], exact).values())).all()
    assert isinstance(fit("logistique", X, y, Xv, yv)[0], Logistic)


def test_train_smoke_and_paired_se(data):
    (X, y), (Xv, yv) = data
    model, _ = fit("table", X, y, Xv, yv)
    shoes = range(200)
    curve = train(model, 3, [500, 1000], shoes, eps_min=.4)
    assert [p["episodes"] for p in curve] == [500, 1000]
    basic = evaluate(lambda obs, unseen: int(obs[0] < 17), shoes)
    assert -1 <= gain(*basic) <= 1
    assert paired_se([(basic, basic)]) == 0


def test_state_features_are_standardized():
    X, _ = dealer_dataset(2000, 9)
    F = np.array([state_features(x[1:]) for x in X])
    assert F.shape == (2000, 12) and np.all(F[:, 0] == 1)
    assert np.all(np.abs(F[:, 1:].mean(axis=0)) < .1) and np.all(np.abs(F[:, 1:].std(axis=0) - 1) < .1)


@pytest.mark.parametrize("model_name", [None, "logistique"])
def test_linear_agents_train(data, model_name):
    (X, y), (Xv, yv) = data
    model = None if model_name is None else fit(model_name, X, y, Xv, yv)[0]
    curve = train(model, 4, [500, 1500], range(200), eps_min=.4, linear=True)
    assert [p["episodes"] for p in curve] == [500, 1500]
    assert -1 <= gain(curve[-1]["gains"], curve[-1]["rounds"]) <= 1
