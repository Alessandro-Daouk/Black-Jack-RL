"""Courbes d'apprentissage sur plusieurs jeux indépendants, sans remplacer le modèle livré."""

import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed, parallel_config
import numpy as np
from sklearn.metrics import log_loss

from src.experiment_utils import describe
from .constants import CARDS, CLASSES
from .data import empirical_probabilities, generate_dataset
from .evaluate import metrics, sample_losses, write_csv
from .model import fit_classifier
from .reference import exact_probabilities
from .train import fingerprint

SIZES = (5_000, 20_000, 60_000, 200_000)
CANDIDATES = (.01, .1, 1., 10., 100.)


def replicate(index, sizes, validation_size, test_size):
    seeds = {"train": 10_000 + 3 * index, "validation": 10_001 + 3 * index, "test": 10_002 + 3 * index}
    X, y = generate_dataset(max(sizes), seeds["train"])
    Xv, yv = generate_dataset(validation_size, seeds["validation"])
    fitted, metadata = [], []
    for n in sizes:
        candidates, matrices = [], []
        for c in CANDIDATES:
            model = fit_classifier(X[:n], y[:n], C=c)
            candidates.append({"C": c, "validation_log_loss": float(log_loss(
                yv, model.predict_proba(Xv), labels=list(CLASSES)))})
            matrices.append(model.predict_proba(CARDS[:, None]))
        chosen = min(range(len(candidates)), key=lambda i: candidates[i]["validation_log_loss"])
        fitted.append((n, candidates[chosen]["C"], matrices[chosen], empirical_probabilities(X[:n], y[:n])))
        metadata.append({"train_size": n, "selected_C": candidates[chosen]["C"], "candidates": candidates,
                         "train_sha256": fingerprint(X[:n], y[:n]),
                         "logistic_probabilities": matrices[chosen].tolist(),
                         "empirical_probabilities": fitted[-1][3].tolist()})
    # Les choix de C pour toutes les tailles sont terminés avant l'évaluation finale.
    Xt, yt = generate_dataset(test_size, seeds["test"])
    reference = exact_probabilities()
    rows = []
    for n, c, learned, frequencies in fitted:
        for method, probabilities in (("logistic", learned), ("frequencies", frequencies)):
            train_loss = float(sample_losses(probabilities, X[:n], y[:n]).mean())
            scores = metrics(probabilities, Xt, yt, reference)
            rows.append({"replicate": index, "method": method, "train_size": n, **seeds,
                         "selected_C": c if method == "logistic" else None,
                         "train_log_loss": train_loss,
                         "validation_log_loss": float(sample_losses(probabilities, Xv, yv).mean()),
                         **scores, "test_minus_train_log_loss": scores["test_log_loss"] - train_loss})
    return rows, {"replicate": index, "seeds": seeds, "validation_sha256": fingerprint(Xv, yv),
                  "test_sha256": fingerprint(Xt, yt), "fits": metadata}


def run(output=Path("results/1_premiere_campagne/modele1/learning_curve"), sizes=SIZES, repeats=5,
        validation_size=12_000, test_size=24_000, jobs=2):
    sizes = tuple(sorted(set(sizes)))
    if repeats < 2 or not sizes or min(sizes) < 1 or min(validation_size, test_size) < 1:
        raise ValueError("Tailles positives et au moins deux répétitions requises.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choisir un dossier de résultats vide pour préserver les expériences.")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"sizes": sizes, "repeats": repeats, "validation_size": validation_size,
                "test_size": test_size, "C_candidates": CANDIDATES,
                "selection": "C sur validation, sans refit ; aucun remplacement du modèle livré.",
                "nested_training_sizes": True, "independent_unit": "jeu simulé (répétition)",
                "uncertainty": "écart-type et IC de Student approximatif de la moyenne entre répétitions"}
    (output / "parameters.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Courbes : {repeats} jeux indépendants, tailles {sizes}…", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs)(delayed(replicate)(i, sizes, validation_size, test_size)
                                       for i in range(repeats))
    rows = [row for result, _ in results for row in result]
    keys = ("train_log_loss", "validation_log_loss", "test_log_loss", "test_minus_train_log_loss",
            "exact_excess_log_loss", "max_probability_error", "max_stand_error", "test_macro_f1")
    arms = [{"method": method, "train_size": n, **{key: describe([
        r[key] for r in rows if r["train_size"] == n and r["method"] == method]) for key in keys}}
        for method in ("logistic", "frequencies") for n in sizes]
    paired = [{"train_size": n, "logistic_minus_frequencies_exact_excess": describe([
        next(r["exact_excess_log_loss"] for r in rows if r["replicate"] == i and r["train_size"] == n
             and r["method"] == "logistic") - next(r["exact_excess_log_loss"] for r in rows
             if r["replicate"] == i and r["train_size"] == n and r["method"] == "frequencies")
        for i in range(repeats)])} for n in sizes]
    summary = {"protocol": protocol, "arms": arms, "paired": paired}
    write_csv(output / "runs.csv", rows)
    (output / "fits.json").write_text(json.dumps([m for _, m in results], indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(summary, output)
    report(summary, output)
    print(f"Courbes et mesures écrites dans {output}", flush=True)
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = summary["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    logistic = [a for a in arms if a["method"] == "logistic"]
    x = [a["train_size"] for a in logistic]
    for key, label in (("train_log_loss", "Entraînement"), ("validation_log_loss", "Validation"),
                       ("test_log_loss", "Test indépendant")):
        axes[0].plot(x, [a[key]["mean"] for a in logistic], marker="o", label=label)
    for method, label in (("logistic", "Régression logistique"), ("frequencies", "Fréquences")):
        selected = [a for a in arms if a["method"] == method]
        mean = np.array([a["exact_excess_log_loss"]["mean"] for a in selected])
        sd = np.array([a["exact_excess_log_loss"]["sd"] for a in selected])
        axes[1].plot(x, mean, marker="o", label=label)
        axes[1].fill_between(x, np.maximum(mean-sd, 1e-12), mean+sd, alpha=.15)
    axes[0].set(title="Log-loss : moyenne des jeux indépendants", ylabel="Log-loss (nats / main)")
    axes[1].set(title="Erreur probabiliste exacte : moyenne ± 1 écart-type",
                ylabel="Excès de log-loss attendu (nats / main)", yscale="log")
    for ax in axes:
        ax.set(xlabel="Mains d'entraînement", xscale="log")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output / "learning_curve.png", dpi=170)
    plt.close(fig)


def report(summary, output):
    lines = ["# Taille des données et généralisation", "",
             "C choisi sur validation ; test indépendant et référence analytique consultés après sélection.",
             "Les tailles partagent des préfixes d'un même jeu dans chaque répétition ; les répétitions sont indépendantes.",
             "Les modèles livrés ne sont pas remplacés par cette étude de sensibilité.", "",
             "| Méthode | Mains train | Log-loss train | Log-loss test | Excès exact moyen | Écart-type de l'excès |",
             "|---|---:|---:|---:|---:|---:|"]
    for a in summary["arms"]:
        lines.append(f"| {a['method']} | {a['train_size']} | {a['train_log_loss']['mean']:.6f} | "
                     f"{a['test_log_loss']['mean']:.6f} | {a['exact_excess_log_loss']['mean']:.8f} | "
                     f"{a['exact_excess_log_loss']['sd']:.8f} |")
    lines += ["", "![Courbes](learning_curve.png)", "", "## Interprétation et limites", "",
              "L'excès exact est la divergence KL pondérée par la distribution des cartes visibles.",
              "Il évite le bruit d'un test fini pour mesurer la qualité probabiliste. Les tailles de données",
              "ne sont pas des répétitions indépendantes ; les statistiques sont calculées entre jeux à taille fixée.",
              f"Seulement {summary['protocol']['repeats']} jeux indépendants : les IC de Student sont approximatifs.",
              "La validation a servi à sélectionner C : son score est optimiste. Le test ne sert à aucun réglage.",
              "Les conclusions concernent ce simulateur à paquet infini, pas d'autres règles de blackjack.",
              "Cette étude ne propage pas la variabilité des jeux supervisés à de nouveaux entraînements RL.", "",
              "`runs.csv` contient les métriques par jeu, `fits.json` les graines, empreintes, choix de C et probabilités.",
              "`summary.json` contient les agrégats. Reproduction : `python -m src.supervised.learning_curve`."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/1_premiere_campagne/modele1/learning_curve"))
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
