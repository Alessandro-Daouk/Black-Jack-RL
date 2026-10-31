"""Exporter les modèles entraînés pour l'application web (web/models.js).

uv run --locked python -m src.export_web
"""
import json
from pathlib import Path

import numpy as np

from src.environment import HIT, STAND
from src.evaluate import J, J_STAR, PI_STAR, Q_STAR, V_TRUE
from src.hybrid_agent import load_agent

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "results/2_campagne_principale/entrainements/agents/g0_r0_n0_standard.npz"
OUTPUT = ROOT / "web/models.js"


def entry(key, name, detail, policy, stand, hit):
    # tables [total][carte visible][as utilisable], comme dans src/q_learning.py
    return {"id": key, "name": name, "detail": detail, "exactGain": round(J(policy), 6),
            "policy": policy.astype(int).tolist(), "stand": np.round(stand, 4).tolist(),
            "hit": np.round(hit, 4).tolist()}


def models():
    agent = load_agent()
    with np.load(STANDARD) as archive:
        Q = archive["Q"]
    return [
        entry("hybride", "Hybride", "Apprend HIT ; STAND estimée par le modèle 1 sur 60\u202f000 mains de croupier. "
              "Agent livré, 100\u202f000 parties.", agent.policy(), agent.stand_values, agent.q_hit),
        entry("standard", "Q-learning", "Apprend STAND et HIT en jouant. Campagne principale, 200\u202f000 parties.",
              Q[HIT] > Q[STAND], Q[STAND], Q[HIT]),
        entry("optimal", "Optimal", "Stratégie exacte, calculée par programmation dynamique.", PI_STAR, V_TRUE, Q_STAR),
    ]


def render():
    data = {"optimalGain": round(J_STAR, 6), "models": models()}
    return ("// Généré par src/export_web.py ; ne pas modifier à la main.\nwindow.MODELS = "
            + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")


def main():
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)} : {OUTPUT.stat().st_size // 1024} Kio")


if __name__ == "__main__":
    main()
