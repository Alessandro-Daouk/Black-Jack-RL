import numpy as np
import pytest

from src.supervised.constants import CLASSES
from src.supervised.reference import exact_probabilities
from src.supervised.values import build_stand_table, stand_value


def certain(outcome):
    return {k: float(k == outcome) for k in CLASSES}


@pytest.mark.parametrize("total,outcome,expected", [(18, 22, 1), (18, 20, -1), (20, 20, 0),
                                                   (21, 20, 1), (22, 22, -1), (26, 17, -1)])
def test_stand_rewards(total, outcome, expected):
    assert stand_value(total, certain(outcome)) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_rejected(bad):
    probabilities = certain(22)
    probabilities[17] = bad
    with pytest.raises(ValueError):
        stand_value(18, probabilities)


def test_missing_classes_and_non_normalized_probabilities_rejected():
    with pytest.raises(ValueError):
        stand_value(18, {22: 1})
    with pytest.raises(ValueError):
        stand_value(18, {k: 0.1 for k in CLASSES})


def test_wassim_table_shape_ace_invariance_and_low_totals():
    probabilities = exact_probabilities()
    V = build_stand_table(probabilities)
    assert V.shape == (22, 11, 2)
    np.testing.assert_array_equal(V[:, :, 0], V[:, :, 1])
    assert np.all(V[:4] == 0) and np.all(V[:, 0] == 0)
    assert np.max(np.abs(V)) <= 1 + 1e-12
    for total in range(4, 17):
        np.testing.assert_allclose(V[total, 1:, 0], 2 * probabilities[:, 5] - 1, atol=1e-12)


@pytest.mark.parametrize("total", [3, -1, 15.5, True])
def test_invalid_player_totals(total):
    with pytest.raises(ValueError):
        stand_value(total, certain(22))
