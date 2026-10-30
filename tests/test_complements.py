import numpy as np
import pytest

from src.audit.complements import card_frequencies, planned_policy, train_double
from src.audit.oracle import Oracle
from src.evaluate import PI_STAR


def test_planning_with_exact_cards_recovers_optimal_policy():
    oracle = Oracle()
    np.testing.assert_array_equal(oracle.vector(planned_policy(oracle.cards)), oracle.vector(PI_STAR))
    assert oracle.evaluate(planned_policy(oracle.cards)) == pytest.approx(oracle.j_star, abs=1e-14)
    assert oracle.evaluate(planned_policy(oracle.cards, oracle.probabilities)) == pytest.approx(oracle.j_star, abs=1e-14)


def test_card_frequencies_requires_every_card():
    assert card_frequencies(np.arange(1, 11)[:, None]).tolist() == [.1] * 10
    with pytest.raises(ValueError):
        card_frequencies(np.ones((5, 1), dtype=int))


def test_double_q_learns_and_is_deterministic():
    oracle = Oracle()
    first = train_double(7, [200, 2000], oracle.evaluate)
    assert [p["episodes"] for p in first] == [200, 2000]
    assert first == train_double(7, [200, 2000], oracle.evaluate)
    assert first[-1]["j"] > -0.3  # nettement mieux que la politique aléatoire (-0,396)
