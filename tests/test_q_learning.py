import numpy as np

from src.environment import HIT, SHAPE, STAND
from src.evaluate import J_STAR, PI_STAR, Q_STAR, V_TRUE, simulate
from src.q_learning import CONFIG, choose_action, train, update

SMALL = {**CONFIG, "episodes": 2_000, "eval_every": 10**9}
S, S2 = (15, 10, 0), (18, 10, 0)


def test_terminal_transition_does_not_bootstrap():
    Q, N = np.zeros((2, *SHAPE)), np.zeros((2, *SHAPE), dtype=np.int64)
    Q[:, 18, 10, 0] = 5.0
    update(Q, N, S, HIT, -1, S2, term=True)
    assert Q[HIT][S] == -1


def test_non_terminal_transition_bootstraps():
    Q, N = np.zeros((2, *SHAPE)), np.zeros((2, *SHAPE), dtype=np.int64)
    Q[STAND][S2] = 0.5
    update(Q, N, S, HIT, 0, S2, term=False)
    assert Q[HIT][S] == 0.5


def test_choose_action_is_valid():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, *SHAPE))
    assert {choose_action(Q, S, eps, rng) for eps in (0, 0.5, 1) for _ in range(100)} <= {STAND, HIT}


def test_unseen_states_default_to_zero_and_values_finite():
    run = train("standard", seed=0, cfg=SMALL)
    assert np.all(np.isfinite(run["Q"]))
    assert np.all(run["Q"][run["N"] == 0] == 0)


def test_hybrid_never_updates_stand():
    run = train("hybrid", seed=0, cfg=SMALL)
    assert np.array_equal(run["Q"][STAND], V_TRUE)
    assert run["N"][STAND].sum() == 0


def test_same_seed_is_reproducible():
    a, b = train("hybrid", seed=3, cfg=SMALL), train("hybrid", seed=3, cfg=SMALL)
    assert np.array_equal(a["Q"], b["Q"])


def test_optimal_easy_cases():
    assert np.allclose(Q_STAR[21, 1:, 0], -1)                          # 21 dur : tirer = bust
    assert np.all(Q_STAR[4:12, 1:, 0] >= V_TRUE[4:12, 1:, 0] - 1e-12)  # p <= 11 : tirer >= rester


def test_exact_value_matches_simulation():
    m, ci = simulate(PI_STAR, 50_000, seed=1)
    assert abs(m - J_STAR) < 2 * ci
