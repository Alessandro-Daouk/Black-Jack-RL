"""Entraîner le Q-learning hybride avec le prédicteur supervisé figé."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.environment import HIT, STAND
from src.evaluate import REACH
from src.hybrid_agent import HybridAgent, file_sha256, load_agent, results_dir_for
from src.q_learning import CONFIG, TUNED, train
from src.supervised.data import require_integer
from src.supervised.model import DEFAULT_MODEL, load_model

# Paramètres RL de src/q_learning.py : configuration précédente, ou réglage du commit 58f5ce3.
PRESETS = {"config": CONFIG, "tuned": TUNED["hybrid"]}
RULES = {"environment": "Blackjack-v1", "natural": False, "sab": False,
         "infinite_deck": True, "stand_on_soft_17": True}


def train_hybrid(model_path=DEFAULT_MODEL, agent_path=None, episodes=100_000,
                 seed=0, results_dir=None, *, preset: str, overwrite=False) -> HybridAgent:
    episodes = require_integer(episodes, "episodes", 1)
    seed = require_integer(seed, "seed", 0)
    if preset not in PRESETS:
        raise ValueError(f"Réglage inconnu : {preset!r}. Choisir parmi {', '.join(PRESETS)}.")
    if agent_path is None:
        raise ValueError("Choisir explicitement agent_path / --agent pour préserver les modèles livrés.")
    agent_path = Path(agent_path)
    if Path(model_path).resolve() == Path(agent_path).resolve():
        raise ValueError("L'agent RL et le modèle supervisé doivent avoir des fichiers distincts.")
    results_dir = results_dir_for(agent_path) if results_dir is None else Path(results_dir)
    if not overwrite and (agent_path.exists() or (results_dir.exists() and any(results_dir.iterdir()))):
        raise ValueError("Sortie déjà existante : choisir un nouveau chemin, ou --overwrite explicitement.")
    base = PRESETS[preset]
    cfg = {**base, "episodes": episodes, "seeds": [seed], "eval_every": min(base["eval_every"], episodes)}
    if cfg["gamma"] != 1.0:
        raise ValueError("gamma=1 est requis pour utiliser la valeur de STAND non actualisée.")
    predictor = load_model(model_path)
    if predictor.metadata.get("rules") != RULES:
        raise ValueError("Les règles du prédicteur ne correspondent pas à l'environnement RL.")
    V = predictor.stand_values()
    print(f"Prédicteur chargé et figé. Entraînement hybride ({preset}) : {episodes:,} épisodes, graine {seed}…",
          flush=True)
    result = train("hybrid", seed=seed, V=V, cfg=cfg)
    if not np.array_equal(result["Q"][STAND], V) or np.any(result["N"][STAND]):
        raise RuntimeError("La valeur de STAND a été modifiée pendant l'apprentissage.")
    metadata = {
        "schema_version": 1, "kind": "hybrid_supervised", "seed": seed,
        "training_preset": preset, "training_config": cfg, "dealer_model_sha256": file_sha256(model_path),
        "dealer_model_filename": Path(model_path).name,
        "dealer_training_splits": predictor.metadata["splits"],
        "rules": predictor.metadata["rules"],
        "training_seconds": result["seconds"],
        "reachable_states": int(REACH.sum()),
        "visited_hit_states": int(np.count_nonzero(result["N"][HIT][REACH])),
        "stand_frozen": True,
    }
    agent = HybridAgent(predictor, result["Q"][HIT], result["N"][HIT], metadata)
    # Les évaluations précédentes ne décrivent plus le modèle dès qu'il est remplacé.
    for stale in ("evaluation.json", "report.md"):
        (results_dir / stale).unlink(missing_ok=True)
    agent.save(agent_path)
    # Rechargement réel avant d'annoncer un artefact utilisable.
    agent = load_agent(agent_path, model_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "training.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (results_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episodes", "exact_expected_reward"])
        writer.writerows(((i + 1) * cfg["eval_every"], float(score)) for i, score in enumerate(result["curve"]))
    print(f"Agent sauvegardé : {agent_path}", flush=True)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=PRESETS,
                        help="config : CONFIG (eps_min=0.05) ; tuned : TUNED['hybrid'] (eps_min=1.0).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--agent", type=Path, required=True, help="Chemin explicite de la nouvelle Q-table.")
    parser.add_argument("--overwrite", action="store_true", help="Remplacer la sortie et invalider son ancienne évaluation.")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Par défaut : results/1_premiere_campagne/hybride/<nom du fichier de l'agent>/.")
    args = parser.parse_args()
    try:
        train_hybrid(args.model, args.agent, args.episodes, args.seed, args.results_dir,
                     preset=args.preset, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
