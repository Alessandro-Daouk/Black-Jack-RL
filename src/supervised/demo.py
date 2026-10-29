"""Démonstration locale : python -m src.supervised.demo --dealer 10 --player 15."""

import argparse

from .model import DEFAULT_MODEL, load_model
from .values import stand_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dealer", type=int, default=10)
    parser.add_argument("--player", type=int, default=15)
    args = parser.parse_args()
    try:
        predictor = load_model(args.model)
        probabilities = predictor.predict_dealer_probabilities(args.dealer)
        value = stand_value(args.player, probabilities)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Carte visible : {args.dealer} | total joueur : {args.player}")
    for outcome, probability in probabilities.items():
        print(f"{'BUST' if outcome == 22 else str(outcome):>4} : {probability:.4%}")
    print(f"Gain espéré de STAND : {value:+.6f}")
    print("Le choix HIT/STAND nécessite aussi la valeur de HIT fournie par l'agent RL.")


if __name__ == "__main__":
    main()
