"""Vérifie le modèle 1 avec un checkout local de la branche wassim, dans une copie temporaire."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DRIVER = r'''
import json
import sys
import numpy as np
from src.environment import STAND, make_env
from src.q_learning import CONFIG, train, choose_action
from src.stand_value import stand_table, exact_dealer_probs
from src.supervised.model import load_model
from src.supervised.reference import exact_probabilities

predictor = load_model(sys.argv[1])
V = predictor.stand_values()
np.testing.assert_allclose(V, stand_table(predictor.probabilities_for_wassim()), atol=1e-12)
np.testing.assert_allclose(exact_probabilities(), np.array(list(exact_dealer_probs().values())), atol=1e-12)
config = {**CONFIG, "episodes": int(sys.argv[2]), "eval_every": int(sys.argv[2])}
result = train("hybrid", seed=0, V=V, cfg=config)
np.testing.assert_array_equal(result["Q"][STAND], V)
assert result["N"][STAND].sum() == 0
assert np.all(np.isfinite(result["Q"]))
rng = np.random.default_rng(2026)
rewards = []
with make_env() as env:
    for i in range(100):
        state, _ = env.reset(seed=2026 if i == 0 else None)
        for step in range(100):
            state, reward, term, trunc, _ = env.step(choose_action(result["Q"], state, 0, rng))
            if term or trunc:
                rewards.append(float(reward))
                break
        else:
            raise AssertionError("Une manche n'a pas terminé.")
assert len(rewards) == 100 and set(rewards) <= {-1, 0, 1}
print(json.dumps({"status": "passed", "training_episodes": config["episodes"],
                  "completed_demo_games": len(rewards), "stand_table_shape": list(V.shape),
                  "stand_frozen_during_training": True,
                  "exact_reference_agrees": True,
                  "note": "Test d'intégration fonctionnel, sans conclusion comparative sur la performance RL."}))
'''


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Checkout local contenant src/q_learning.py de Wassim.")
    parser.add_argument("--model", type=Path, default=root / "models/supervised/dealer_model.joblib")
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=root / "results/supervised/integration.json")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / "src/q_learning.py").is_file() or args.episodes < 1:
        parser.error("Un checkout de Wassim et un nombre positif d'épisodes sont requis.")
    with tempfile.TemporaryDirectory(prefix="blackjack-integration-") as folder:
        checkout = Path(folder)
        shutil.copytree(repo / "src", checkout / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "src/supervised", checkout / "src/supervised", ignore=shutil.ignore_patterns("__pycache__"))
        env = {**os.environ, "PYTHONPATH": str(checkout)}
        completed = subprocess.run([sys.executable, "-c", DRIVER, str(args.model.resolve()), str(args.episodes)],
                                   cwd=checkout, env=env, capture_output=True, text=True, check=True)
        report = json.loads(completed.stdout)
    report["wassim_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report["wassim_worktree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    report["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
