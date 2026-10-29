"""Évaluer l'agent hybride figé sur des parties indépendantes sans exploration."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.environment import HIT, make_env
from src.evaluate import J, J_STAR
from src.hybrid_agent import DEFAULT_AGENT, file_sha256, load_agent, results_dir_for
from src.supervised.data import require_integer
from src.supervised.model import DEFAULT_MODEL


def evaluate_agent(agent, games=10_000, seed=2026) -> dict:
    games = require_integer(games, "games", 2)
    seed = require_integer(seed, "seed", 0)
    if seed == agent.metadata["seed"]:
        raise ValueError("Utiliser une graine d'évaluation différente de l'entraînement RL.")
    rewards = np.empty(games)
    unvisited_hit_decisions = decisions = 0
    with make_env() as env:
        for game in range(games):
            state, _ = env.reset(seed=seed if game == 0 else None)
            terminated = truncated = False
            while not (terminated or truncated):
                decision = agent.decide(state)
                decisions += 1
                unvisited_hit_decisions += int(decision.action == HIT and decision.hit_visits == 0)
                state, reward, terminated, truncated, _ = env.step(decision.action)
            if truncated:
                raise RuntimeError("Épisode tronqué : résultat de blackjack incomplet.")
            rewards[game] = reward
    mean = float(rewards.mean())
    half_width = float(1.96 * rewards.std(ddof=1) / np.sqrt(games))
    exact_reward = J(agent.policy())
    return {
        "games": games, "evaluation_seed": seed, "rl_training_seed": agent.metadata["seed"],
        "rl_training_episodes": agent.metadata["training_config"]["episodes"],
        "exploration_epsilon": 0,
        "wins": int((rewards == 1).sum()), "draws": int((rewards == 0).sum()),
        "losses": int((rewards == -1).sum()), "mean_reward": mean,
        "mean_reward_approx_95pct_interval": [mean - half_width, mean + half_width],
        "exact_expected_reward": float(exact_reward), "optimal_exact_expected_reward": float(J_STAR),
        "gap_to_optimal_exact": float(J_STAR - exact_reward),
        "decisions": decisions, "unvisited_hit_decisions": unvisited_hit_decisions,
        "dealer_model_sha256": agent.metadata["dealer_model_sha256"],
        "scope": "Un seul entraînement RL. IC du test conditionnel à cet agent fixé ; "
                 "pas de conclusion de supériorité sur le Q-learning standard.",
    }


def write_report(result: dict, metadata: dict, output: Path) -> None:
    interval = result["mean_reward_approx_95pct_interval"]
    cfg = metadata["training_config"]
    # Les agents antérieurs à 58f5ce3 n'enregistraient ni le réglage ni omega (alpha = 1/N, soit omega = 1).
    text = f"""# Évaluation de l'agent hybride

Le prédicteur supervisé reste figé pendant le Q-learning. Les sources RL de
Wassim proviennent du commit `{metadata['wassim_source_commit']}`.

## Protocole

- RL : {result['rl_training_episodes']:,} épisodes, graine {result['rl_training_seed']}.
- Réglage RL : `{metadata.get('training_preset', 'non enregistré')}` ; eps0={cfg['eps0']:g}, eps_min={cfg['eps_min']:g},
  decay={cfg['decay']:g}, omega={cfg.get('omega', 1.0):g}, gamma={cfg['gamma']:g}.
- Évaluation : {result['games']:,} parties, graine {result['evaluation_seed']}, sans exploration.
- Règles : Blackjack-v1, natural=False, sab=False ; récompenses -1/0/+1.
- Données du prédicteur : {metadata['dealer_training_splits']['train']['samples']:,} mains
  d'entraînement ; validation et test indépendants, détaillés dans `training.json`.

## Résultats de simulation

- Victoires : {result['wins']:,} ({result['wins']/result['games']:.2%}).
- Égalités : {result['draws']:,} ({result['draws']/result['games']:.2%}).
- Défaites : {result['losses']:,} ({result['losses']/result['games']:.2%}).
- Gain moyen observé : {result['mean_reward']:+.6f} par manche.
- Intervalle normal approximatif à 95 % : [{interval[0]:+.6f}, {interval[1]:+.6f}].

Cet intervalle ne mesure que le bruit de simulation pour l'agent fixé ; il ne
mesure pas les différences entre plusieurs entraînements.

## Évaluation exacte de la politique

- Gain moyen exact : {result['exact_expected_reward']:+.9f} par manche.
- Gain optimal exact : {result['optimal_exact_expected_reward']:+.9f}.
- Écart à l'optimum : {result['gap_to_optimal_exact']:.9f} unité de récompense par manche.

Le calcul exact utilise la récurrence de Wassim et la distribution initiale de
Gymnasium. Le gain observé peut dépasser le gain optimal exact par fluctuation
d'échantillonnage : cela ne signifie pas que la politique bat l'optimum.
Même la politique optimale a une espérance négative dans ces règles.

## Vérifications et limites

- STAND est resté figé pendant l'apprentissage.
- HIT a été observé dans {metadata['visited_hit_states']} des {metadata['reachable_states']} états atteignables.
- {result['unvisited_hit_decisions']} décision HIT sans observation préalable de cette action pendant le test.
- Empreinte du prédicteur : `{result['dealer_model_sha256']}`.
- Empreinte de l'agent : `{result['agent_sha256']}`.

`training.json` contient les paramètres, versions et empreintes des sources RL.
`learning_curve.csv` contient les évaluations exactes pendant l'entraînement.
Ce rapport décrit un seul entraînement RL : il ne suffit pas à comparer des méthodes.
La comparaison sur plusieurs graines des réglages CONFIG et TUNED, avec le Q-learning
standard et l'hybride exact comme références, est dans `results/hybrid/config_vs_tuned/`.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None,
                        help="Par défaut : results/hybrid/<nom du fichier de l'agent>/evaluation.json.")
    args = parser.parse_args()
    try:
        agent = load_agent(args.agent, args.model)
        result = evaluate_agent(agent, args.games, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    result["agent_sha256"] = file_sha256(args.agent)
    output = args.output or results_dir_for(args.agent) / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(result, agent.metadata, output.parent / "report.md")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
