"""Métriques probabilistes et évaluation indépendante d'un modèle sauvegardé."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .constants import CARD_PROBABILITIES, CLASSES, OUTCOME_NAMES
from .data import conditional_counts, generate_dataset, validate_dataset
from .model import DEFAULT_MODEL, load_model
from .reference import exact_probabilities
from .values import build_stand_table, validate_probabilities


def sample_losses(probabilities: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    validate_dataset(X, y)
    probabilities = validate_probabilities(probabilities, (10, 6))
    # Même convention de protection que la log-loss numérique de scikit-learn.
    return -np.log(np.clip(probabilities[X[:, 0] - 1, y - 17], np.finfo(float).eps, 1))


def metrics(probabilities: np.ndarray, X: np.ndarray, y: np.ndarray, reference: np.ndarray) -> dict:
    losses = sample_losses(probabilities, X, y)
    predictions = probabilities[X[:, 0] - 1]
    one_hot = np.eye(6)[y - 17]
    # Le KL mathématique n'est pas une log-loss numériquement écrêtée : si une
    # classe vraie positive reçoit zéro, son risque est réellement infini.
    clipped_cross_entropy = -(reference * np.log(np.clip(probabilities, np.finfo(float).eps, 1))).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact_cross_entropy = -np.where(reference > 0, reference * np.log(probabilities), 0).sum(axis=1)
    exact_entropy = -(reference * np.log(reference)).sum(axis=1)
    stand_error = build_stand_table(probabilities)[4:, 1:, 0] - build_stand_table(reference)[4:, 1:, 0]
    return {
        "test_log_loss": float(losses.mean()),
        "test_brier": float(np.square(predictions - one_hot).sum(axis=1).mean()),
        "test_accuracy": float(np.mean(np.asarray(CLASSES)[predictions.argmax(axis=1)] == y)),
        "test_macro_f1": float(f1_score(y, np.asarray(CLASSES)[predictions.argmax(axis=1)],
                                       labels=list(CLASSES), average="macro", zero_division=0)),
        "exact_expected_log_loss": float(CARD_PROBABILITIES @ exact_cross_entropy),
        "exact_excess_log_loss": float(CARD_PROBABILITIES @ (exact_cross_entropy - exact_entropy)),
        "clipped_excess_log_loss_eps_machine": float(CARD_PROBABILITIES @ (clipped_cross_entropy - exact_entropy)),
        "max_total_variation": float(0.5 * np.abs(probabilities - reference).sum(axis=1).max()),
        "max_probability_error": float(np.abs(probabilities - reference).max()),
        "max_stand_error": float(np.abs(stand_error).max()),
        "mean_stand_error_uniform_states": float(np.abs(stand_error).mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_predictor(predictor, X: np.ndarray, y: np.ndarray, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    reference = exact_probabilities()
    learned = predictor.probability_matrix()
    baseline = validate_probabilities(predictor.metadata["empirical_probabilities"], (10, 6))
    rows = [
        {"method": name, **metrics(probabilities, X, y, reference)}
        for name, probabilities in (("logistic", learned), ("frequencies", baseline), ("exact_reference", reference))
    ]
    write_csv(output / "metrics.csv", rows)
    counts = conditional_counts(X, y)
    calibration = []
    for d in range(10):
        n = int(counts[d].sum())
        for k, label in enumerate(OUTCOME_NAMES):
            calibration.append({
                "dealer_showing": d + 1, "outcome": label, "test_count": n,
                "test_frequency": float(counts[d, k] / n) if n else "",
                "logistic": float(learned[d, k]), "frequencies": float(baseline[d, k]),
                "exact_reference": float(reference[d, k]),
            })
    write_csv(output / "calibration.csv", calibration)
    deltas = sample_losses(learned, X, y) - sample_losses(baseline, X, y)
    delta = float(deltas.mean())
    half_width = float(1.96 * deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else None
    summary = {
        "test_samples": len(y), "metrics": rows,
        "paired_log_loss_delta_logistic_minus_frequencies": delta,
        "paired_delta_approx_95pct_interval": [delta - half_width, delta + half_width] if half_width is not None else None,
        "interval_scope": "Incertitude du jeu de test uniquement, prédicteurs entraînés fixés ; approximation normale.",
    }
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from .plots import make_plots
    make_plots(learned, baseline, reference, counts, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("results/supervised/reevaluation"))
    args = parser.parse_args()
    predictor = load_model(args.model)
    splits = predictor.metadata["splits"]
    if args.seed in {splits["train"]["seed"], splits["validation"]["seed"]}:
        parser.error("Choisir une graine différente de l'entraînement et de la validation.")
    X, y = generate_dataset(args.samples, args.seed)
    summary = evaluate_predictor(predictor, X, y, args.output)
    summary["test_seed"] = args.seed
    summary["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
