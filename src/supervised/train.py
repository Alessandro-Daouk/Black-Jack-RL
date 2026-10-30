"""Entraîner, sélectionner sur validation, exporter et évaluer le modèle 1."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import log_loss

from .constants import CLASSES, OUTCOME_NAMES
from .data import conditional_counts, empirical_probabilities, generate_dataset
from .model import DealerPredictor, fit_classifier, load_model


def fingerprint(X, y) -> str:
    # Format explicite pour une empreinte indépendante de l'endianness de la machine.
    return hashlib.sha256(X.astype("<i8").tobytes() + y.astype("<i8").tobytes()).hexdigest()


def run_experiment(train_size=60_000, validation_size=12_000, test_size=12_000,
                   train_seed=42, validation_seed=43, test_seed=123,
                   model_dir=Path("models/supervised"), results_dir=Path("results/supervised")) -> dict:
    if len({train_seed, validation_seed, test_seed}) != 3:
        raise ValueError("Les trois graines doivent être distinctes.")
    for output in (Path(model_dir), Path(results_dir)):
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choisir des dossiers de sortie vides pour préserver les modèles et résultats.")
    start = time.perf_counter()
    print("Simulation des jeux d'entraînement et de validation…", flush=True)
    X_train, y_train = generate_dataset(train_size, train_seed)
    X_val, y_val = generate_dataset(validation_size, validation_seed)
    baseline = empirical_probabilities(X_train, y_train)
    candidates, pipelines = [], []
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        pipeline = fit_classifier(X_train, y_train, C=C)
        score = float(log_loss(y_val, pipeline.predict_proba(X_val), labels=list(CLASSES)))
        candidates.append({"C": C, "validation_log_loss": score})
        pipelines.append(pipeline)
        print(f"C={C:g} : log-loss validation = {score:.6f}", flush=True)
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation_log_loss"])
    # Choix terminé avant toute génération ou consultation du jeu de test.
    best = candidates[selected]
    print(f"C retenu : {best['C']:g}. Évaluation sur le test indépendant…", flush=True)
    X_test, y_test = generate_dataset(test_size, test_seed)
    splits = {
        name: {"samples": len(y), "seed": seed, "sha256": fingerprint(X, y)}
        for name, X, y, seed in (
            ("train", X_train, y_train, train_seed),
            ("validation", X_val, y_val, validation_seed),
            ("test", X_test, y_test, test_seed),
        )
    }
    metadata = {
        "schema_version": 1, "model": "OneHotEncoder + LogisticRegression (lbfgs)",
        "classes": list(CLASSES), "outcome_names": list(OUTCOME_NAMES),
        "rules": {"environment": "Blackjack-v1", "natural": False, "sab": False,
                  "infinite_deck": True, "stand_on_soft_17": True},
        "splits": splits, "selected_C": best["C"], "candidates": candidates,
        "selection": "Minimum de log-loss sur validation ; apprentissage sur train uniquement, sans refit.",
        "empirical_probabilities": baseline.tolist(),
        "training_counts": conditional_counts(X_train, y_train).tolist(),
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scikit-learn", "scipy", "gymnasium", "joblib", "matplotlib")}},
    }
    predictor = DealerPredictor(pipelines[selected], metadata)
    model_dir, results_dir = Path(model_dir), Path(results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "dealer_model.joblib"
    predictor.save(model_path)
    predictor = load_model(model_path)
    np.save(model_dir / "stand_values.npy", predictor.stand_values(), allow_pickle=False)
    exports = {str(d): predictor.predict_dealer_probabilities(d) for d in range(1, 11)}
    (model_dir / "probabilities.json").write_text(json.dumps(exports, indent=2) + "\n", encoding="utf-8")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from .evaluate import evaluate_predictor, write_csv
    write_csv(results_dir / "validation.csv", candidates)
    summary = evaluate_predictor(predictor, X_test, y_test, results_dir)
    summary.update({"test_seed": test_seed, "elapsed_seconds": time.perf_counter() - start,
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
    (results_dir / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(metadata, summary, results_dir)
    print(f"Modèle : {model_path}\nRapport : {results_dir / 'report.md'}", flush=True)
    return summary


def write_report(metadata: dict, summary: dict, output: Path) -> None:
    model, baseline, oracle = summary["metrics"]
    interval = summary["paired_delta_approx_95pct_interval"]
    lines = [
        "# Première expérience supervisée", "",
        "Prédicteur du résultat du croupier dans Blackjack-v1, natural=False, sab=False.",
        "Une carte visible en entrée ; six probabilités en sortie. Aucun état caché en entrée.", "",
        "## Protocole", "",
        *[f"- {name} : {split['samples']:,} mains, graine {split['seed']}." for name, split in metadata["splits"].items()],
        f"- C choisi sur validation : {metadata['selected_C']:g}, parmi 0.01, 0.1, 1, 10 et 100.",
        "- La validation n'est pas réutilisée pour l'apprentissage. Le test ne sert à aucun choix.",
        "- La référence exacte est réservée à l'évaluation, sans génération de labels d'entraînement.", "",
        "## Résultats", "",
    ]
    for row, name in ((model, "Régression logistique"), (baseline, "Fréquences d'entraînement"), (oracle, "Référence exacte")):
        lines.extend([f"**{name}** : log-loss test {row['test_log_loss']:.6f} ; "
                      f"Brier {row['test_brier']:.6f} ; accuracy {row['test_accuracy']:.2%}.", ""])
    lines.extend([
        f"Erreur maximale d'une probabilité du modèle face à la référence : {model['max_probability_error']:.6f}.",
        f"Erreur maximale sur V_STAND : {model['max_stand_error']:.6f} unité de récompense.",
        f"Excès de log-loss espérée exact : {model['exact_excess_log_loss']:.6f} pour le modèle, "
        f"{baseline['exact_excess_log_loss']:.6f} pour les fréquences.", "",
        "L'excès exact est la divergence KL moyenne, pondérée par la vraie distribution des cartes visibles.",
        "Il mesure l'écart au prédicteur idéal sans bruit d'un jeu de test fini.", "",
        f"Différence appariée de log-loss test (modèle − fréquences) : "
        f"{summary['paired_log_loss_delta_logistic_minus_frequencies']:+.6f}.",
    ])
    if interval is not None:
        lines.append(f"Intervalle normal approximatif à 95 % : [{interval[0]:+.6f}, {interval[1]:+.6f}].")
        if interval[0] <= 0 <= interval[1]:
            lines.append("L'intervalle contient zéro : ce test ne met pas en évidence d'avantage net de log-loss.")
    lines.extend([
        "Cet intervalle ne mesure que la variabilité du test pour ces deux modèles fixés.", "",
        "## Lecture et limites", "",
        "Le problème n'a que dix valeurs d'entrée. La baseline de fréquences estime directement",
        "les mêmes dix distributions ; une supériorité nette du classificateur n'est pas acquise.",
        "Une faible accuracy est compatible avec des probabilités fiables : l'issue de chaque main est aléatoire.",
        "La référence exacte minimise la log-loss en espérance, mais pas nécessairement sur chaque test fini.", "",
        "Il s'agit d'un premier entraînement avec une seule graine d'apprentissage. Répéter les",
        "entraînements sur d'autres graines sera nécessaire pour une conclusion générale.",
        "Ces métriques ne démontrent pas un gain de performance de l'agent RL : celui-ci doit",
        "être réentraîné avec cette V_STAND, puis comparé à budgets équivalents sur plusieurs graines.", "",
        "Le Brier est la somme des six erreurs quadratiques, moyennée sur les mains, sans division par six.",
        "L'erreur moyenne sur STAND utilise un poids uniforme sur les 180 couples total/carte",
        "(totaux 4..21, cartes 1..10), sans prétendre reproduire leur fréquence dans les parties.", "",
        "## Fichiers", "",
        "- `metrics.csv` : métriques test et erreurs face à la référence exacte.",
        "- `validation.csv` : choix de C.",
        "- `calibration.csv` : les 60 probabilités, fréquences de test et références.",
        "- `calibration.png` : comparaison par issue, IC de Wilson à 95 % point par point.",
        "- `stand_error.png` : carte des erreurs de gain espéré de STAND.",
        "- `evaluation.json` : résultats, graine de test et empreinte du modèle.", "",
        "Les paramètres, versions, graines, empreintes des jeux et fréquences d'apprentissage",
        "sont enregistrés dans `models/supervised/metadata.json`.", "",
        "![Probabilités du croupier](calibration.png)", "",
        "![Erreur sur STAND](stand_error.png)", "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--validation-size", type=int, default=12_000)
    parser.add_argument("--test-size", type=int, default=12_000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)
    parser.add_argument("--test-seed", type=int, default=123)
    parser.add_argument("--model-dir", type=Path, default=Path("models/supervised"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/supervised"))
    args = parser.parse_args()
    try:
        run_experiment(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
