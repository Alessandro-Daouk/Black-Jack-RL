import numpy as np

from src.evaluate import V_TRUE
from src.stand_value import empirical_dealer_probs, exact_dealer_probs, tv_distance

EXACT = exact_dealer_probs()


def test_dealer_probs_sum_to_one():
    for d in range(1, 11):
        assert abs(EXACT[d].sum() - 1) < 1e-12


def test_dealer_probs_known_values():
    # valeurs publiées : croupier reste sur 17 soft, paquet infini, sans « peek »
    assert abs(EXACT[1][5] - 0.1153) < 1e-4    # as visible : bust
    assert abs(EXACT[1][4] - 0.3616) < 1e-4    # as visible : 21
    assert abs(EXACT[10][5] - 0.2121) < 1e-4   # 10 visible : bust


def test_stand_value_in_range():
    assert np.all((V_TRUE >= -1) & (V_TRUE <= 1))


def test_stand_value_constant_below_17():
    for d in range(1, 11):
        assert np.allclose(V_TRUE[4:17, d, :], 2 * EXACT[d][5] - 1)


def test_empirical_probs_converge():
    probs = empirical_dealer_probs(50_000, np.random.default_rng(0))
    assert tv_distance(probs, EXACT) < 0.03
