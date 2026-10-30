"""Vérifier l'intégrité des campagnes v2 sans relancer d'entraînement.

Contrôles : campagnes terminées, empreintes des tables et des agents, gains exacts
recalculés depuis les Q-tables archivées, cohérence de l'oracle indépendant avec
l'oracle historique, et lien entre l'analyse et les mesures qu'elle résume.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

from src.audit.core import digest
from src.audit.oracle import Oracle
from src.evaluate import J_STAR

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/v2"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def complete(folder):
    if not (folder / "completion.json").exists() or not (folder / "provenance.json").exists():
        raise ValueError(f"Campagne incomplète ou sans provenance : {folder.relative_to(ROOT)}")
    status = load(folder / "completion.json")
    if status.get("status") != "complete":
        raise ValueError(f"Statut inattendu : {folder.relative_to(ROOT)}")
    return status


def verify():
    oracle = Oracle()
    if not math.isclose(oracle.j_star, J_STAR, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Oracle indépendant et oracle historique en désaccord.")
    report = {"optimal_j": oracle.j_star, "campaigns": {}}

    complete(V2 / "diagnostics")
    for row in rows(V2 / "diagnostics/baselines.csv"):
        if abs(float(row["z"])) > 4:
            raise ValueError(f"Monte-Carlo incompatible avec l'oracle : {row['policy']}")
    report["campaigns"]["diagnostics"] = "baselines MC within 4 standard errors"

    final = V2 / "final"
    status = complete(final)
    for name, key in (("runs.csv", "run_table_sha256"), ("curves.csv", "curve_table_sha256")):
        if digest(final / name) != status[key]:
            raise ValueError(f"Table modifiée après la campagne : {name}")
    config = load(final / "config.json")
    if config["pilot_design_sha256"] != digest(V2 / "pilot/design.json"):
        raise ValueError("Le plan final ne correspond plus au pilote archivé.")
    largest = 0.
    runs = rows(final / "runs.csv")
    for row in runs:
        path = final / f"agents/{row['run']}.npz"
        if digest(path) != row["agent_sha256"]:
            raise ValueError(f"Agent archivé modifié : {row['run']}")
        with np.load(path, allow_pickle=False) as archive:
            Q = archive["Q"]
        largest = max(largest, abs(oracle.evaluate(Q[1] > Q[0]) - float(row["j"])))
    if largest > 1e-12:
        raise ValueError(f"Gain recalculé différent de runs.csv : {largest:.3g}")
    report["campaigns"]["final"] = {"runs": len(runs), "max_recomputed_j_difference": largest}

    for name in ("replication", "variant"):
        if (V2 / name).exists():
            report["campaigns"][name] = complete(V2 / name).get("status")

    analysis = V2 / "analysis"
    if analysis.exists():
        settings = load(analysis / "provenance.json")["config"]
        if settings["runs_sha256"] != status["run_table_sha256"] or settings["curves_sha256"] != status["curve_table_sha256"]:
            raise ValueError("L'analyse ne porte pas sur les mesures finales archivées.")
        report["campaigns"]["analysis"] = "linked to final measurements"
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
