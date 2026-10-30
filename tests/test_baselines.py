import numpy as np
import pytest

from src.audit.baselines import KINDS, train_model_based, train_td, variance_components
from src.audit.oracle import Oracle


@pytest.fixture(scope="module")
def oracle():
    return Oracle()


@pytest.mark.parametrize("kind", KINDS)
def test_td_baselines_learn_something(kind, oracle):
    curve = train_td(kind, 0, [200, 3000], oracle.evaluate, eps_min=.4)
    assert [p["episodes"] for p in curve] == [200, 3000]
    assert curve[-1]["j"] > -.2  # mieux que « toujours rester » après 3 000 parties


def test_td_rejects_unknown_kind(oracle):
    with pytest.raises(ValueError):
        train_td("sarsa_lambda", 0, [10], oracle.evaluate)


def test_model_based_converges_towards_optimum(oracle):
    curve = train_model_based(0, [1000, 20000], oracle)
    regrets = [oracle.j_star - p["j"] for p in curve]
    assert regrets[1] < regrets[0] and regrets[1] < .03


def test_variance_components():
    rng = np.random.default_rng(0)
    groups = rng.normal(0, 1, (200, 1))
    values = groups + rng.normal(0, .5, (200, 10))
    out = variance_components(values)
    assert out["sd_between_datasets"] == pytest.approx(1, rel=.15)
    assert out["sd_within_rl"] == pytest.approx(.5, rel=.05)
    assert variance_components(np.ones((3, 4)))["share_datasets"] == 0
