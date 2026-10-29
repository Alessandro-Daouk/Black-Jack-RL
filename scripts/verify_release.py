"""Vérifier les empreintes et la cohérence du modèle livré sans le réentraîner."""

import json
import math

from src.evaluate import J, J_STAR
from src.hybrid_agent import ROOT, file_sha256, load_agent


def verify():
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        path = ROOT / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Empreinte différente : {entry['path']}")
    agent = load_agent(ROOT / manifest["default_agent"], ROOT / manifest["default_predictor"])
    evaluation = json.loads((ROOT / manifest["evaluation"]).read_text(encoding="utf-8"))
    if file_sha256(ROOT / manifest["default_agent"]) != evaluation["agent_sha256"]:
        raise ValueError("L'évaluation ne correspond pas à l'agent livré.")
    if file_sha256(ROOT / manifest["default_predictor"]) != evaluation["dealer_model_sha256"]:
        raise ValueError("L'évaluation ne correspond pas au prédicteur livré.")
    score = J(agent.policy())
    if not math.isclose(score, evaluation["exact_expected_reward"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Le gain recalculé ne correspond pas au rapport.")
    upstream = json.loads((ROOT / "docs/wassim_source.json").read_text(encoding="utf-8"))
    for entry in upstream["files"]:
        if file_sha256(ROOT / entry["local_path"]) != entry["sha256"]:
            raise ValueError(f"Source de Wassim modifiée : {entry['local_path']}")
    return {"status": "passed", "agent": manifest["default_agent"], "exact_expected_reward": score,
            "optimal_exact_expected_reward": J_STAR, "artifacts_checked": len(manifest["artifacts"])}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
