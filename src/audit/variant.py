"""Robustesse locale E6 : cartes uniformes 1..10, mêmes décisions et gains."""
import argparse
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium.envs.toy_text.blackjack import BlackjackEnv, sum_hand
import numpy as np

from src.supervised.data import conditional_counts
from src.supervised.evaluate import write_csv
from src.supervised.train import fingerprint
from src.supervised.values import build_stand_table
from .core import dataset, dump, interval, new_output, smoothed, train
from .oracle import Oracle
from .campaign import CHECKPOINTS


class UniformBlackjack(BlackjackEnv):
    """Gym conserve observation/as ; reset et tirages remplacent seulement la loi des cartes.

Pas de tirages cosmétiques : les secondes ne sont pas comparées à Blackjack-v1.
"""
    def __init__(self):
        super().__init__(natural=False, sab=False)

    def draw(self):
        return int(self.np_random.integers(1, 11))

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.dealer = [self.draw(), self.draw()]
        self.player = [self.draw(), self.draw()]
        return self._get_obs(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Action invalide.')
        if action == 1:
            self.player.append(self.draw())
            terminal = sum_hand(self.player) > 21
            reward = -float(terminal)
        else:
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw())
            p, d = sum_hand(self.player), sum_hand(self.dealer)
            reward = 1. if d > 21 else float(np.sign(p-d))
            terminal = True
        return self._get_obs(), reward, terminal, False, {}


def run(output, groups):
    output = new_output(output, {'phase': 'E6', 'groups': groups, 'rl_repeats': 2,
        'card_probabilities': [.1]*10, 'train_size': 20000, 'checkpoints': CHECKPOINTS,
        'data_seed_base': 71000, 'rl_seed_base': 81000, 'role': 'secondary_descriptive',
        'no_cross_environment_wall_time_comparison': True})
    start = time.perf_counter()
    oracle = Oracle(np.full(10, .1))
    # Validation MC du nouvel oracle, sans sélection d'hyperparamètres.
    rewards = []
    with UniformBlackjack() as env:
        for episode in range(100000):
            state, _ = env.reset(seed=70001 if episode == 0 else None)
            while True:
                state, reward, done, _, _ = env.step(int(oracle.policy[oracle.index[state]]))
                if done:
                    rewards.append(reward)
                    break
    dump(output/'oracle_mc.json', {'j_star': oracle.j_star, 'mc': interval(rewards), 'seed': 70001})
    rows, curves = [], []
    for i in range(groups):
        X, y, cards, times = dataset(20000, 71000+10*i, np.full(10, .1))
        counts = conditional_counts(X, y)
        matrices = {'frequencies': counts/counts.sum(axis=1,keepdims=True), 'jeffreys': smoothed(counts)}
        dump(output/f'fit_{i}.json', {'seed': 71000+10*i, 'sha256': fingerprint(X,y),
            'counts': counts.tolist(), 'cards': int(cards[-1]), 'seconds': float(times[-1])})
        for j in range(2):
            for method in ('standard', 'frequencies', 'jeffreys'):
                p = matrices.get(method, oracle.probabilities)
                result = train(81000+10*i+j, build_stand_table(p), CHECKPOINTS, oracle.evaluate,
                               eps_min=.4 if method=='standard' else 1., standard=method=='standard',
                               env_factory=UniformBlackjack)
                info = {'group': i, 'rl_rep': j, 'method': method, 'seed': 81000+10*i+j}
                rows.append({**info, **oracle.diagnostics(p, result['Q'][1]>result['Q'][0]), **result['curve'][-1]})
                curves.extend({**info, **r, 'regret': oracle.j_star-r['j']} for r in result['curve'])
                write_csv(output/'runs.csv', rows)
                write_csv(output/'curves.csv', curves)
                print(f"E6 group={i} rl={j} method={method}", flush=True)
    dump(output/'completion.json', {'status':'complete', 'runs':len(rows), 'j_star':oracle.j_star,
                                   'wall_seconds':time.perf_counter()-start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.design.read_text(encoding='utf-8'))
    run(args.output, config['secondary_groups'])
