"""Jouer avec l'agent hybride ou expliquer sa décision pour un état donné."""

import argparse
from pathlib import Path

from src.environment import make_env
from src.hybrid_agent import DEFAULT_AGENT, load_agent
from src.supervised.data import require_integer
from src.supervised.model import DEFAULT_MODEL


def explain(agent, state) -> None:
    decision = agent.decide(state)
    print(f"État {decision.state} : STAND={decision.stand_value:+.6f}, "
          f"HIT={decision.hit_value:+.6f} → {decision.action_name} "
          f"(HIT observé {decision.hit_visits} fois à l'entraînement)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--games", type=int, default=5)
    mode.add_argument("--state", nargs=3, type=int, metavar=("TOTAL", "CARTE", "AS_UTILISABLE"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trace", action="store_true", help="Afficher les valeurs et décisions à chaque tour.")
    args = parser.parse_args()
    try:
        require_integer(args.games, "games", 1)
        require_integer(args.seed, "seed", 0)
        agent = load_agent(args.agent, args.model)
        if args.state is not None:
            explain(agent, args.state)
            return
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    with make_env() as env:
        for game in range(args.games):
            state, _ = env.reset(seed=args.seed if game == 0 else None)
            print(f"Partie {game + 1} — état initial : {state}")
            terminated = truncated = False
            while not (terminated or truncated):
                if args.trace:
                    explain(agent, state)
                state, reward, terminated, truncated, _ = env.step(agent.act(state))
            outcome = "victoire" if reward > 0 else "défaite" if reward < 0 else "égalité"
            print(f"Résultat : {outcome} ({reward:+g})\n")


if __name__ == "__main__":
    main()
