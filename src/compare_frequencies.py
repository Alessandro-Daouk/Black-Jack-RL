"""Ablation : régression logistique, fréquences empiriques, loi exacte et Q-learning standard."""

import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed, parallel_config
import numpy as np

from src.environment import STAND
from src.evaluate import J, J_STAR, V_TRUE
from src.experiment_utils import describe
from src.hybrid_agent import file_sha256
from src.q_learning import TUNED, greedy_policy, train
from src.supervised.evaluate import write_csv
from src.supervised.model import DEFAULT_MODEL, load_model
from src.supervised.values import build_stand_table

METHODS = ("logistic", "frequencies", "exact", "standard")


def run_one(method, seed, episodes, tables):
    agent = "standard" if method == "standard" else "hybrid"
    cfg = {**TUNED[agent], "episodes": episodes, "seeds": [seed], "eval_every": min(2000, episodes)}
    V = tables.get(method, V_TRUE)
    result = train(agent, seed, V=V, cfg=cfg)
    if agent == "hybrid":
        assert np.array_equal(result["Q"][STAND], V) and not result["N"][STAND].any()
    score = float(J(greedy_policy(result["Q"])))
    row = {"method": method, "seed": seed, "episodes": episodes, "eps_min": cfg["eps_min"],
           "exact_expected_reward": score, "gap_to_optimal": J_STAR-score,
           "training_seconds": result["seconds"]}
    curve = [{"method": method, "seed": seed, "episodes": (i+1)*cfg["eval_every"],
              "exact_expected_reward": float(value)} for i, value in enumerate(result["curve"])]
    return row, curve


def run(output=Path("results/1_premiere_campagne/hybride/frequencies_ablation"), model_path=DEFAULT_MODEL,
        episodes=100_000, seeds=tuple(range(400, 420)), jobs=2):
    if episodes < 1 or len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Épisodes positifs et au moins deux graines distinctes positives ou nulles.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choisir un dossier de résultats vide.")
    predictor = load_model(model_path)
    frequencies = np.asarray(predictor.metadata["empirical_probabilities"])
    counts = np.asarray(predictor.metadata["training_counts"])
    np.testing.assert_allclose(frequencies, counts/counts.sum(axis=1, keepdims=True), atol=1e-15)
    tables = {"logistic": predictor.stand_values(), "frequencies": build_stand_table(frequencies),
              "exact": V_TRUE}
    protocol = {"episodes": episodes, "seeds": list(seeds), "methods": METHODS,
                "tuning": "Paramètres TUNED fixés avant l'expérience ; aucun choix sur ces graines.",
                "dealer_model_sha256": file_sha256(model_path), "splits": predictor.metadata["splits"],
                "frequency_source": "Même jeu d'entraînement que la régression ; aucune donnée supplémentaire.",
                "presets": TUNED, "primary_contrast": "logistic - frequencies, gain exact final par graine",
                "uncertainty": "IC de Student approximatif de la différence moyenne appariée ; pas de test d'équivalence"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "parameters.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Ablation : {len(seeds)*len(METHODS)} entraînements de {episodes:,} épisodes…", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs)(delayed(run_one)(m, seed, episodes, tables)
                                       for seed in seeds for m in METHODS)
    rows = [r for r, _ in results]
    curves = [r for _, curve in results for r in curve]
    arms = [{"method": m, "exact_expected_reward": describe([
        r["exact_expected_reward"] for r in rows if r["method"] == m]),
        "gap_to_optimal": describe([r["gap_to_optimal"] for r in rows if r["method"] == m])} for m in METHODS]
    lookup = {(r["method"], r["seed"]): r["exact_expected_reward"] for r in rows}
    paired = {}
    for reference in ("frequencies", "exact", "standard"):
        d = np.array([lookup["logistic", s] - lookup[reference, s] for s in seeds])
        paired["logistic_minus_" + reference] = {**describe(d), "better": int((d > 0).sum()),
                                                "equal": int((d == 0).sum()), "worse": int((d < 0).sum())}
    summary = {"protocol": protocol, "arms": arms, "paired": paired,
               "optimal_exact_expected_reward": J_STAR}
    write_csv(output / "runs.csv", rows)
    write_csv(output / "learning_curves.csv", curves)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report(summary, output)
    print(f"Ablation écrite dans {output}", flush=True)
    return summary


def report(summary, output):
    protocol = summary["protocol"]
    train_size = protocol["splits"]["train"]["samples"]
    lines = ["# Ablation : remplacer la régression par les fréquences", "",
             f"Les deux estimateurs utilisent exactement les mêmes {train_size:,} mains d'entraînement du prédicteur figé.",
             "Le budget RL et les réglages sont identiques entre hybrides ; le standard utilise TUNED[standard].",
             "La référence exacte reçoit la vraie loi pour STAND. Elle n'est pas une borne à budget fini.", "",
             "| Méthode | J moyen | Écart-type entre graines | Écart moyen à l'optimum |",
             "|---|---:|---:|---:|"]
    for a in summary["arms"]:
        lines.append(f"| {a['method']} | {a['exact_expected_reward']['mean']:+.8f} | "
                     f"{a['exact_expected_reward']['sd']:.8f} | {a['gap_to_optimal']['mean']:.8f} |")
    lines += ["", "## Différences appariées", ""]
    for name, d in summary["paired"].items():
        lo, hi = d["mean_approx_95pct_interval"]
        lines.append(f"- {name} : {d['mean']:+.8f}, IC approximatif à 95 % [{lo:+.8f}, {hi:+.8f}] ; "
                     f"meilleur/égal/moins bon : {d['better']}/{d['equal']}/{d['worse']}.")
    lines += ["", "## Portée", "",
              "Les intervalles portent sur la variabilité des entraînements RL, conditionnellement au seul jeu supervisé fixé.",
              "Ils utilisent l'approximation de Student sur les différences appariées et ne corrigent pas la multiplicité.",
              "Un intervalle contenant zéro n'établit pas l'équivalence. Le contraste principal est logistic − frequencies.",
              "L'évaluation exacte J n'a pas de bruit de simulation pour une politique fixée.",
              f"Les {protocol['episodes']:,} épisodes par entraînement n'incluent ni les mains supervisées ni le coût du réglage initial.",
              f"Graines : {protocol['seeds']}. Elles ne servent pas à changer le modèle livré.", "",
              "Reproduction : `python -m src.compare_frequencies`. Paramètres effectifs dans `parameters.json`.",
              "Mesures par entraînement dans `runs.csv`, trajectoires dans `learning_curves.csv`, agrégats dans `summary.json`."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/1_premiere_campagne/hybride/frequencies_ablation"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
