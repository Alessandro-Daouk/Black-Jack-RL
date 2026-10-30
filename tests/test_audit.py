import copy

import gymnasium as gym
import numpy as np
import pytest

from src.audit.core import categorical_risk, dataset, fit_estimators, smoothed, train
from src.audit.oracle import Oracle, observation
from src.environment import hit
from src.evaluate import J, J_STAR, PI_STAR, REACH, V_TRUE
from src.q_learning import CONFIG, train as original_train
from src.supervised.data import generate_dataset
from src.supervised.evaluate import metrics
from src.audit.variant import UniformBlackjack


@pytest.fixture(scope='module')
def oracle():
    return Oracle()


def test_independent_oracle_mass_reachability_and_values(oracle):
    assert len(oracle.states) == 280
    assert min(s[0]-10*s[2] for s in oracle.states) == 2
    np.testing.assert_array_equal(oracle.table(np.ones(280, bool)), REACH)
    np.testing.assert_allclose(oracle.transition.sum(axis=1)+oracle.bust, 1, atol=1e-14)
    np.testing.assert_allclose(oracle.probabilities.sum(axis=1), 1, atol=1e-14)
    assert oracle.initial.sum() == pytest.approx(1)
    assert not np.tril(oracle.transition).any()
    np.testing.assert_allclose(oracle.stand, oracle.vector(V_TRUE), atol=1e-14)
    assert oracle.j_star == pytest.approx(J_STAR, abs=1e-14)
    np.testing.assert_array_equal(oracle.policy, oracle.vector(PI_STAR))
    for seed in range(10):
        p = np.random.default_rng(seed).integers(0, 2, 280)
        assert oracle.evaluate(p) == pytest.approx(J(oracle.table(p)), abs=1e-14)


def test_all_reachable_hit_transitions_against_gym(oracle):
    class FixedRng:
        def __init__(self, card):
            self.card = card
        def choice(self, *args, **kwargs):
            return self.card
    with gym.make('Blackjack-v1', natural=False, sab=False) as env:
        for state in oracle.states:
            for card in range(1, 11):
                env.reset(seed=5)
                env.unwrapped.player = list(oracle.hands[state])
                env.unwrapped.dealer = [state[1], 10]
                env.unwrapped.np_random = FixedRng(card)
                nxt, reward, terminal, truncated, _ = env.step(1)
                expected = observation(oracle.hands[state]+(card,), state[1])
                assert nxt == expected
                bust, p, a = hit(state[0], state[2], card)
                assert terminal == bust == (expected[0] > 21)
                assert reward == (-1 if terminal else 0)
                assert not truncated
                if not bust:
                    assert (p, state[1], a) == expected
                    assert p-10*a > state[0]-10*state[2]


def test_gym_natural_and_tie_no_bonus():
    with gym.make('Blackjack-v1', natural=False, sab=False) as env:
        env.reset(seed=8)
        env.unwrapped.player = [1, 10]
        env.unwrapped.dealer = [10, 10]
        assert env.step(0)[1:4] == (1., True, False)
        env.unwrapped.player = [10, 7]
        env.unwrapped.dealer = [1, 6]
        rng_state = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)
        assert env.step(0)[1:4] == (0., True, False)
        assert env.unwrapped.np_random.bit_generator.state == rng_state


@pytest.mark.parametrize('standard,epsilon,omega', [(True, .4, 1.), (False, 1., 1.), (False, .05, .8)])
def test_instrumentation_preserves_historical_trajectories(oracle, standard, epsilon, omega):
    old = original_train('standard' if standard else 'hybrid', 119,
                         cfg={**CONFIG, 'episodes': 1000, 'eval_every': 1000, 'eps_min': epsilon, 'omega': omega})
    new = train(119, V_TRUE, [0, 100, 1000], oracle.evaluate, epsilon, omega, standard)
    np.testing.assert_array_equal(old['Q'], new['Q'])
    np.testing.assert_array_equal(old['N'], new['N'])
    assert new['curve'][-1]['j'] == pytest.approx(old['curve'][-1], abs=1e-14)
    assert new['curve'][-1]['cards'] >= 4000
    assert new['visits'].sum() == new['curve'][-1]['decisions']


def test_data_instrumentation_and_direct_stand():
    X, y, cards, times = dataset(1000, 17)
    X0, y0 = generate_dataset(1000, 17)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    assert cards[-1] >= 2000 and np.all(np.diff(cards) >= 2)
    assert np.all(np.diff(times) >= 0)
    _, _, fit = fit_estimators(X, y, X, y)
    assert fit['direct_stand_max_error'] < 1e-14


def test_raw_kl_is_infinite_and_smoothing_is_finite(oracle):
    counts = np.zeros((10, 6), dtype=int)
    counts[:, 0] = 1
    assert np.isinf(categorical_risk(counts, oracle.probabilities, oracle.cards))
    q = smoothed(counts)
    assert np.isfinite(categorical_risk(q, oracle.probabilities, oracle.cards))
    np.testing.assert_allclose(q.sum(axis=1), 1)
    scores = metrics(counts, np.array([[1]]), np.array([17]), oracle.probabilities)
    assert np.isinf(scores['exact_excess_log_loss'])
    assert np.isfinite(scores['clipped_excess_log_loss_eps_machine'])


def test_bounds_and_occupation_identity_for_valid_perturbations(oracle):
    for seed in range(12):
        p = np.random.default_rng(seed).dirichlet(np.ones(6), size=10)
        d = oracle.diagnostics(p)
        assert d['eta'] <= 2*d['max_tv']+1e-14
        assert d['max_hit_error_limit'] <= d['eta']+1e-14
        assert d['regret'] <= d['global_limit_regret_bound']+1e-14
        assert d['regret'] <= d['local_limit_regret_bound']+1e-14
        assert d['regret'] == pytest.approx(d['occupancy_regret_identity'], abs=1e-14)
        random_policy = np.random.default_rng(seed).integers(0, 2, 280)
        r = oracle.diagnostics(p, random_policy)
        assert r['regret'] == pytest.approx(r['occupancy_regret_identity'], abs=1e-14)


def test_physical_card_count_excludes_gym_cosmetic_random_draws(monkeypatch, oracle):
    import gymnasium.envs.toy_text.blackjack as blackjack
    original = blackjack.draw_card
    count = 0
    def draw(rng):
        nonlocal count
        count += 1
        return original(rng)
    monkeypatch.setattr(blackjack, 'draw_card', draw)
    result = train(1234, V_TRUE, [200], oracle.evaluate)
    assert result['curve'][-1]['cards'] == count


def test_uniform_variant_transitions_and_stand_rewards(oracle):
    with UniformBlackjack() as env:
        env.reset(seed=99)
        for state in oracle.states:
            for card in range(1,11):
                env.player = list(oracle.hands[state])
                env.dealer = [state[1],10]
                env.draw = lambda: card
                nxt, reward, terminal, truncated, _ = env.step(1)
                assert nxt == observation(oracle.hands[state]+(card,), state[1])
                assert terminal == (nxt[0]>21) and not truncated
                assert reward == (-1 if terminal else 0)
        for player in range(4,22):
            for dealer in range(17,23):
                env.player = [player]  # somme dure, aucun as
                env.dealer = [dealer]
                assert env.step(0)[1] == (1 if dealer>21 else np.sign(player-dealer))


def test_cluster_statistics_censoring_and_cost_selection():
    import pandas as pd
    from src.audit.analyze import paired, thresholds, matched_costs
    frame = pd.DataFrame([{'group':g,'rl_rep':r,'method':m,'j':g if m=='candidate' else 0.}
                          for g in range(3) for r in range(2) for m in ('candidate','reference')])
    result = paired(frame,'candidate','reference')
    assert result['n']==3 and result['rl_pairs']==6 and result['sd']==1.
    records=[]
    for i, (regret,cost) in enumerate([(.1,0),(.005,1),(.02,2)]):
        records.append({'run':'test','group':0,'rl_rep':0,'n':1000,'method':'logistic','episodes':i*100,
                        'regret':regret,'j':-regret,'total_seconds':cost,'total_cards':cost*10000,
                        'reused_total_seconds':cost})
    curves=pd.DataFrame(records)
    first=thresholds(curves,[.01,.001])
    assert first.iloc[0].episodes==100 and first.iloc[0].final_below_threshold==0
    assert first.iloc[1].success==0 and pd.isna(first.iloc[1].episodes)
    matched=matched_costs(curves)
    chosen=matched[(matched.resource=='total_seconds') & (matched.budget==2)].iloc[0]
    assert chosen.episodes==200 and chosen.regret==.02  # dernier, pas meilleur checkpoint


def test_tost_matches_ninety_percent_interval_rule():
    from scipy.stats import t
    from src.audit.analyze import cluster_summary
    values = np.array([.0001, -.0002, .0003, .0000, -.0001, .0002])
    r = cluster_summary(values, margin=.001)
    half = t.ppf(.95, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
    assert r['ci90'] == pytest.approx([values.mean()-half, values.mean()+half])
    assert r['equivalent_within_margin_5pct'] and r['tost_p'] < .05
    shifted = cluster_summary(values+.0009, margin=.001)
    assert not shifted['equivalent_within_margin_5pct']
    assert (shifted['ci90'][1] >= .001) == (shifted['tost_p'] >= .05)

