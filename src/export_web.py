"""Exporter les modèles entraînés pour l'application web.

uv run --locked python -m src.export_web            # paquet infini : web/models.js, quelques secondes
uv run --locked python -m src.export_web --sabot    # sabot fini : web/models_shoe.js, quelques minutes
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.environment import HIT, STAND
from src.evaluate import J, J_STAR, PI_STAR, Q_STAR, V_TRUE
from src.hybrid_agent import load_agent

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "results/2_campagne_principale/entrainements/agents/g0_r0_n0_standard.npz"
OUTPUT = ROOT / "web/models.js"
SHOE_OUTPUT = ROOT / "web/models_shoe.js"
SHOE_EVAL = range(5_000_000, 5_020_000)  # sabots d'évaluation, distincts de toutes les campagnes


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


def shoe_data():
    """Agents linéaires du jeu 0 de la campagne du sabot fini (mêmes données, graine et réglages), réentraînés
    à l'identique, plus des états témoins calculés ici pour vérifier le calcul JavaScript."""
    from joblib import Parallel, delayed
    from src.shoe import campaign
    from src.shoe.game import REWARD, Shoe, dealer_dataset, evaluate, play_round
    from src.shoe.models import fit, standardizer

    X, y = dealer_dataset(campaign.RL_SIZE, campaign.DATA_SEED)
    Xv, yv = dealer_dataset(campaign.VALIDATION_SIZE, campaign.VALIDATION_SEED)
    logistic = fit("logistique", X, y, Xv, yv)[0]
    specs = {"hybride": (logistic, .1), "lineaire": (None, 1.)}  # ε_min retenus par la campagne
    runs = Parallel(n_jobs=len(specs))(delayed(campaign.train)(
        m, campaign.RL_SEED, [campaign.CHECKPOINTS[-1]], SHOE_EVAL, eps_min=e, linear=True, return_q=True)
        for m, e in specs.values())
    basic = evaluate(lambda obs, _: int(PI_STAR[obs[:3]]), SHOE_EVAL)

    # Coefficients arrondis, et témoins calculés avec ces mêmes coefficients : le JavaScript doit les retrouver.
    W, b = np.round(logistic.W, 8), np.round(logistic.b, 8)
    mu, sd = (np.round(v, 8) for v in standardizer())
    Q = {key: np.round(q, 6) for key, (_, q) in zip(specs, runs, strict=True)}

    def features(unseen):
        return np.concatenate(([1.], (np.append(unseen / unseen.sum(), unseen.sum() / 52) - mu) / sd))

    def logistic_stand(total, up, unseen):
        logits = W[:, up - 1] @ np.concatenate(([1.], unseen / unseen.sum(), [unseen.sum() / 52])) + b
        p = np.exp(logits - logits.max())
        return float(p / p.sum() @ REWARD[total])

    rng, shoe, seen = np.random.default_rng(7), Shoe(np.random.default_rng(8)), []
    while len(seen) < 40:
        play_round(shoe, lambda obs, unseen: seen.append((obs, unseen)) or int(rng.integers(2)))
    checks = []
    for obs, unseen in seen:
        f, cell = features(unseen), obs[:3]
        assert abs(logistic_stand(obs[0], obs[1], unseen) - logistic.one(obs[1], unseen, obs[3]) @ REWARD[obs[0]]) < 1e-5
        checks.append({"total": int(obs[0]), "upcard": int(obs[1]), "usable": int(obs[2]), "unseen": unseen.tolist(),
                       "values": {"hybride": [logistic_stand(obs[0], obs[1], unseen), float(Q["hybride"][1][cell] @ f)],
                                  "lineaire": [float(Q["lineaire"][0][cell] @ f), float(Q["lineaire"][1][cell] @ f)]}})

    def estimated(curve):
        return round(float(curve[-1]["gains"].sum() / curve[-1]["rounds"].sum()), 5)

    return {
        "evalRounds": int(basic[1].sum()), "logistic": {"W": W.tolist(), "b": b.tolist()},
        "mu": mu.tolist(), "sd": sd.tolist(), "checks": checks,
        "models": [
            {"id": "hybride", "name": "Hybride linéaire", "kind": "lineaire", "standFrom": "logistique",
             "detail": "STAND par une régression logistique sur la composition du sabot ; apprend HIT, "
                       "linéaire dans la composition. 3\u202fmillions de parties.",
             "q": Q["hybride"].tolist(), "estimatedGain": estimated(runs[0][0])},
            {"id": "lineaire", "name": "Q-learning linéaire", "kind": "lineaire", "standFrom": "q",
             "detail": "Apprend STAND et HIT en jouant, linéaire dans la composition du sabot. 3\u202fmillions de parties.",
             "q": Q["lineaire"].tolist(), "estimatedGain": estimated(runs[1][0])},
            {"id": "base", "name": "Stratégie de base", "kind": "table",
             "detail": "Stratégie optimale du paquet infini : ignore les cartes déjà sorties.",
             "policy": PI_STAR.astype(int).tolist(), "stand": np.round(V_TRUE, 4).tolist(),
             "hit": np.round(Q_STAR, 4).tolist(), "estimatedGain": round(float(basic[0].sum() / basic[1].sum()), 5)},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sabot", action="store_true", help="Exporter aussi les agents du sabot fini (quelques minutes).")
    args = parser.parse_args()
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)} : {OUTPUT.stat().st_size // 1024} Kio")
    if args.sabot:
        SHOE_OUTPUT.write_text("// Généré par src/export_web.py --sabot ; ne pas modifier à la main.\nwindow.SHOE = "
                               + json.dumps(shoe_data(), ensure_ascii=False, separators=(",", ":")) + ";\n",
                               encoding="utf-8")
        print(f"{SHOE_OUTPUT.relative_to(ROOT)} : {SHOE_OUTPUT.stat().st_size // 1024} Kio")


if __name__ == "__main__":
    from src.export_web import main as export  # fonctions importables : joblib les transmet aux processus
    export()
