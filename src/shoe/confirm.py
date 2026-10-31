"""Confirmation des conclusions du sabot fini selon un plan figé (results/6_confirmation_sabot/plan.json).

Données, graines et sabots d'évaluation neufs ; hyperparamètres déjà retenus, sans réglage.

uv run --locked python -m src.shoe.confirm --plan results/6_confirmation_sabot/plan.json --output results/6_confirmation_sabot/mesures
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from joblib import Parallel, delayed

from src.audit.core import digest, dump, new_output
from src.supervised.evaluate import write_csv
from .campaign import gain, paired_se, train, with_evaluation
from .game import dealer_dataset, exact_dealer
from .models import fit, scores


def supervised_group(plan, i):
    seeds = plan["graines"]
    X, y = dealer_dataset(plan["mains_modele1"], seeds["donnees"] + 10 * i)
    Xv, yv = dealer_dataset(plan["validation"], seeds["validation"] + 10 * i)
    Xt, yt = dealer_dataset(plan["test_modele1"], seeds["test"] + 10 * i)
    exact = np.array([exact_dealer(x[0], x[1:]) for x in Xt])
    models = {name: fit(name, X, y, Xv, yv)[0] for name in ("logistique", "table")}
    rows = [{"group": i, "model": name, **scores(m, Xt, yt, exact)} for name, m in models.items()]
    return rows, models["logistique"]


def run(plan_path, output):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    output = new_output(output, {"plan": str(plan_path), "plan_sha256": digest(plan_path)})
    started = time.perf_counter()
    groups, level, budgets = range(plan["groupes"]), plan["niveau"], plan["budgets"]
    shoes = range(*plan["sabots_evaluation"])

    results = Parallel(n_jobs=len(groups))(delayed(supervised_group)(plan, i) for i in groups)
    rows = [r for res, _ in results for r in res]
    write_csv(output / "supervised.csv", rows)
    logistic = {i: res[1] for i, res in zip(groups, results, strict=True)}

    jobs = [(a, i, r) for a in plan["agents"] for i in groups for r in range(plan["graines_rl_par_groupe"])]
    curves = Parallel(n_jobs=-1)(delayed(train)(
        None if spec["modele1"] is None else logistic[i], plan["graines"]["rl"] + 10 * i + r, budgets, shoes,
        eps_min=spec["eps_min"], omega=spec["omega"], linear=spec["lineaire"])
        for a, i, r in jobs for spec in [plan["agents"][a]])
    point = {(a, i, r, p["episodes"]): (p["gains"], p["rounds"]) for (a, i, r), c in zip(jobs, curves, strict=True)
             for p in c}
    write_csv(output / "runs.csv", [{"agent": a, "group": i, "rl_rep": r, "episodes": e, "j": gain(*v),
                                     "rounds": int(v[1].sum())} for (a, i, r, e), v in point.items()])

    reps = range(plan["graines_rl_par_groupe"])

    def contrast(a, b, e):
        diff = [np.mean([gain(*point[a, i, r, e]) - gain(*point[b, i, r, e]) for r in reps]) for i in groups]
        se = paired_se([(point[a, i, r, e], point[b, i, r, e]) for i in groups for r in reps])
        return with_evaluation(diff, se, level)

    verdicts = {}
    for h in plan["hypotheses"]:
        if h["id"] == "H4":
            kl = {m: [r["excess_kl"] for r in rows if r["model"] == m] for m in ("table", "logistique")}
            ci = with_evaluation(np.subtract(kl["table"], kl["logistique"]), 0., level)
        else:
            a, b = (x.strip() for x in h["contraste"].split(" - "))
            ci = contrast(a, b, h["parties"])
        low, high = ci["ci95_with_evaluation"]  # au niveau du plan, malgré le nom de la clé
        confirmed = -0.001 <= low and high <= 0.001 if h["id"] == "H2" else low > 0
        verdicts[h["id"]] = {"enonce": h["enonce"], "contraste": h["contraste"], "mean": ci["mean"],
                             "ci": [low, high], "niveau": level, "evaluation_mc_se": ci["evaluation_mc_se"],
                             "critere": h["confirmee_si"], "confirmee": bool(confirmed)}
    dump(output / "summary.json", {"verdicts": verdicts, "agents": {a: {str(e): float(np.mean(
        [gain(*point[a, i, r, e]) for i in groups for r in reps])) for e in budgets} for a in plan["agents"]}})
    dump(output / "completion.json", {"status": "complete", "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    # Passer par le module importé : joblib transmet alors aux processus des fonctions importables.
    from src.shoe.confirm import run as run_confirmation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_confirmation(args.plan, args.output)
