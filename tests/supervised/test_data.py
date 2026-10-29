import copy

import gymnasium as gym
from gymnasium.envs.toy_text.blackjack import sum_hand
import numpy as np
import pytest

from src.supervised.constants import CARD_PROBABILITIES, CLASSES
from src.supervised.data import finish_dealer, generate_dataset, hand_value
from src.supervised.reference import exact_probabilities


@pytest.mark.parametrize("cards,total", [([1, 6], 17), ([1, 1, 9], 21), ([1, 10, 10], 21),
                                        ([10, 10, 5], 25), ([1, 1, 1, 8], 21), ([1, 1, 10, 6], 18)])
def test_aces(cards, total):
    assert hand_value(cards) == total == sum_hand(cards)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_dealer_matches_gymnasium_for_every_initial_pair(seed):
    # Même état du générateur et même main initiale ; compare des transitions complètes.
    with gym.make("Blackjack-v1", natural=False, sab=False) as env:
        for visible in range(1, 11):
            for hidden in range(1, 11):
                env.reset(seed=seed)
                env.unwrapped.dealer = [visible, hidden]
                env.unwrapped.player = [10, 7]
                rng = np.random.default_rng()
                rng.bit_generator.state = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)
                outcome = finish_dealer([visible, hidden], rng)
                _, reward, terminated, truncated, _ = env.step(0)
                gym_total = sum_hand(env.unwrapped.dealer)
                assert outcome == (22 if gym_total > 21 else gym_total)
                assert terminated and not truncated
                assert reward == (1 if outcome == 22 else 0 if outcome == 17 else -1)


def test_soft_17_stops_without_drawing():
    rng = np.random.default_rng(1)
    before = copy.deepcopy(rng.bit_generator.state)
    hand = [1, 6]
    assert finish_dealer(hand, rng) == 17
    assert rng.bit_generator.state == before
    assert hand == [1, 6]


def test_dataset_is_reproducible_and_has_real_card_weights():
    X, y = generate_dataset(20_000, 42)
    X2, y2 = generate_dataset(20_000, 42)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert not np.array_equal(y, generate_dataset(20_000, 123)[1])
    assert X.shape == (20_000, 1) and set(y) == set(CLASSES)
    frequencies = np.bincount(X[:, 0], minlength=11)[1:] / len(X)
    np.testing.assert_allclose(frequencies, CARD_PROBABILITIES, atol=0.012)


def test_exact_reference_known_values():
    probabilities = exact_probabilities()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert abs(probabilities[0, 5] - 0.11528627) < 1e-7
    assert abs(probabilities[0, 4] - 0.36155813) < 1e-7
    assert abs(probabilities[9, 5] - 0.21210908) < 1e-7


@pytest.mark.parametrize("n,seed", [(0, 42), (-1, 42), (True, 42), (5, -1), (5, 1.5)])
def test_invalid_generation_arguments(n, seed):
    with pytest.raises(ValueError):
        generate_dataset(n, seed)
