"""Comparer CONFIG et TUNED["hybrid"] pour l'hybride supervisé, à budget RL égal.

Protocole fixé avant d'exécuter la comparaison :

1. Validation, graines 200 à 219 : l'hybride qui utilise la V_STAND du prédicteur figé
   est entraîné avec CONFIG puis avec TUNED["hybrid"], aux mêmes graines et avec le même
   nombre d'épisodes. On retient le réglage dont le gain exact moyen de la politique
   gloutonne finale est le plus élevé ; une égalité conserve CONFIG.
2. Test, graines 0 à 19, qui ne servent à aucun choix : les deux réglages, plus le
   Q-learning standard et l'hybride avec V_STAND exacte comme références.

Le gain exact J utilise la récurrence de Wassim et la vraie loi du croupier. C'est un
outil d'évaluation : l'agent supervisé ne l'utilise jamais pendant l'apprentissage.
"""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
from joblib import Parallel, delayed

from src.environment import HIT, STAND
from src.evaluate import J, J_STAR, PI_STAR, REACH, V_TRUE, solve
from src.hybrid_agent import ROOT, file_sha256
from src.q_learning import TUNED, greedy_policy, train
from src.supervised.data import require_integer
from src.supervised.model import DEFAULT_MODEL, load_model
from src.train_hybrid import PRESETS, RULES

VALIDATION_SEEDS = tuple(range(200, 220))
TEST_SEEDS = tuple(range(20))
METHODS = ("hybrid_supervised", "hybrid_exact", "standard")
METHOD_NAMES = {"hybrid_supervised": "Hybride, V_STAND supervisée", "hybrid_exact": "Hybride, V_STAND exacte",
                "standard": "Q-learning standard"}
HANDS_COUNTED = {"train_hands": "entraînement", "all_hands": "toutes", "none": "aucune"}
CHECKPOINTS = (10_000, 20_000, 50_000, 100_000)
LEGACY_AGENT = ROOT / "models/hybrid/agent_config.npz"  # CONFIG, graine 0, sources du commit 0780a4f
DEFAULT_OUTPUT = ROOT / "results/hybrid/config_vs_tuned"


def preset_config(method: str, preset: str, episodes: int) -> dict:
    """CONFIG est commun ; TUNED dépend du type d'agent (eps_min=0.4 standard, 1.0 hybride)."""
    base = TUNED["standard"] if method == "standard" and preset == "tuned" else PRESETS[preset]
    return {**base, "episodes": episodes, "eval_every": min(base["eval_every"], episodes)}


def train_run(method: str, preset: str, seed: int, episodes: int, v_supervised: np.ndarray) -> dict:
    cfg = {**preset_config(method, preset, episodes), "seeds": [seed]}
    V = v_supervised if method == "hybrid_supervised" else V_TRUE
    return train("standard" if method == "standard" else "hybrid", seed, V=V, cfg=cfg)


def run_row(phase, method, preset, seed, episodes, result, v_supervised, limit) -> dict:
    Q, N = result["Q"], result["N"]
    if method != "standard" and (np.any(N[STAND]) or not np.array_equal(
            Q[STAND], v_supervised if method == "hybrid_supervised" else V_TRUE)):
        raise RuntimeError("La valeur de STAND a été modifiée pendant l'apprentissage.")
    policy = greedy_policy(Q)
    exact = J(policy)
    # Politique limite : optimale pour la V_STAND fournie (estimée pour l'hybride supervisé).
    limit_policy, limit_reward = limit if method == "hybrid_supervised" else (PI_STAR, J_STAR)
    cfg = preset_config(method, preset, episodes)
    return {
        "phase": phase, "method": method, "preset": preset, "seed": seed, "episodes": episodes,
        "eps_min": cfg["eps_min"], "omega": cfg["omega"],
        "exact_expected_reward": exact, "gap_to_optimal": J_STAR - exact, "gap_to_limit": limit_reward - exact,
        "states_differing_from_optimal": int(np.count_nonzero((policy != PI_STAR) & REACH)),
        "states_differing_from_limit": int(np.count_nonzero((policy != limit_policy) & REACH)),
        "visited_hit_states": int(np.count_nonzero(N[HIT][REACH])),
        "training_seconds": result["seconds"],
    }


def retained_preset(validation_rows: list[dict]) -> str:
    """Règle fixée à l'avance : meilleur gain exact moyen de l'hybride supervisé ; égalité → CONFIG."""
    mean = {preset: np.mean([row["exact_expected_reward"] for row in validation_rows
                             if row["method"] == "hybrid_supervised" and row["preset"] == preset])
            for preset in PRESETS}
    return "tuned" if mean["tuned"] > mean["config"] else "config"


def sign_flip_p_value(differences) -> float:
    """p bilatéral exact du test de permutation des signes sur des différences appariées."""
    d = np.asarray(differences, dtype=float)
    if d.size > 24:
        raise ValueError("Le test exact énumère 2^n signes : 24 graines au plus.")
    observed, count = abs(d.sum()), 0
    for start in range(0, 2 ** d.size, 2 ** 16):
        codes = np.arange(start, min(start + 2 ** 16, 2 ** d.size))
        signs = 1 - 2 * ((codes[:, None] >> np.arange(d.size)) & 1)
        count += int(np.count_nonzero(np.abs(signs @ d) >= observed - 1e-12))
    return count / 2 ** d.size


def paired_difference(rows: list[dict], reference: tuple, candidate: tuple) -> dict:
    """Gain exact du candidat moins celui de la référence, graine par graine ; bras = (méthode, réglage)."""
    values = {arm: {row["seed"]: row["exact_expected_reward"] for row in rows
                    if (row["method"], row["preset"]) == arm} for arm in (reference, candidate)}
    d = np.array([values[candidate][seed] - values[reference][seed] for seed in sorted(values[reference])])
    return {"reference": list(reference), "candidate": list(candidate), "seeds": int(d.size),
            "mean": float(d.mean()), "standard_error": float(d.std(ddof=1) / np.sqrt(d.size)),
            "candidate_better": int(np.sum(d > 0)), "equal": int(np.sum(d == 0)),
            "candidate_worse": int(np.sum(d < 0)), "sign_flip_p_value": sign_flip_p_value(d)}


def arm_summary(rows: list[dict], method: str, preset: str) -> dict:
    selected = [row for row in rows if row["method"] == method and row["preset"] == preset]

    def column(name: str) -> np.ndarray:
        return np.array([row[name] for row in selected], dtype=float)

    return {"method": method, "preset": preset, "eps_min": selected[0]["eps_min"], "seeds": len(selected),
            "exact_expected_reward_mean": float(column("exact_expected_reward").mean()),
            "exact_expected_reward_sd": float(column("exact_expected_reward").std(ddof=1)),
            "gap_to_optimal_mean": float(column("gap_to_optimal").mean()),
            "gap_to_limit_mean": float(column("gap_to_limit").mean()),
            "states_differing_from_optimal_mean": float(column("states_differing_from_optimal").mean()),
            "states_differing_from_limit_mean": float(column("states_differing_from_limit").mean())}


def mean_gap_at(curves: dict, method: str, preset: str, episodes: int, eval_every: int) -> float | None:
    """Écart moyen à l'optimum de la politique gloutonne après `episodes` épisodes, sur les graines de test."""
    if episodes <= 0 or episodes % eval_every:
        return None
    index = episodes // eval_every - 1
    gaps = [J_STAR - curve[index] for (phase, m, p, _), curve in curves.items()
            if (phase, m, p) == ("test", method, preset) and index < len(curve)]
    return float(np.mean(gaps)) if gaps else None


def legacy_reproduced(results: dict, model_path: Path, episodes: int) -> bool | None:
    """L'ancien agent est-il retrouvé à l'identique par le run CONFIG de même graine ? None si non comparable."""
    if not LEGACY_AGENT.exists():
        return None
    with np.load(LEGACY_AGENT, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        key = ("test", "hybrid_supervised", "config", metadata["seed"])
        if (key not in results or metadata["training_config"]["episodes"] != episodes
                or metadata["dealer_model_sha256"] != file_sha256(model_path)):
            return None
        return bool(np.array_equal(archive["q_hit"], results[key]["Q"][HIT])
                    and np.array_equal(archive["visits"], results[key]["N"][HIT]))


def compare(model_path=DEFAULT_MODEL, episodes=100_000, output_dir=DEFAULT_OUTPUT,
            validation_seeds=VALIDATION_SEEDS, test_seeds=TEST_SEEDS, jobs=-1) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choisir un dossier de résultats vide pour préserver les expériences.")
    episodes = require_integer(episodes, "episodes", 1)
    validation_seeds = [require_integer(seed, "seed", 0) for seed in validation_seeds]
    test_seeds = [require_integer(seed, "seed", 0) for seed in test_seeds]
    if (min(len(validation_seeds), len(test_seeds)) < 2 or set(validation_seeds) & set(test_seeds)
            or len(set(validation_seeds)) < len(validation_seeds) or len(set(test_seeds)) < len(test_seeds)):
        raise ValueError("Validation et test exigent chacun au moins deux graines, toutes distinctes.")
    predictor = load_model(model_path)
    if predictor.metadata.get("rules") != RULES:
        raise ValueError("Les règles du prédicteur ne correspondent pas à l'environnement RL.")
    V = predictor.stand_values()
    q_limit = solve(V)[0]
    limit = (q_limit > V, J(q_limit > V))
    started = time.perf_counter()

    def run_phase(phase: str, arms: list[tuple]) -> dict:
        print(f"{phase} : {len(arms)} entraînements de {episodes:,} épisodes…", flush=True)
        results = Parallel(n_jobs=jobs)(delayed(train_run)(m, p, s, episodes, V) for m, p, s in arms)
        return {(phase, *arm): result for arm, result in zip(arms, results, strict=True)}

    validation = run_phase("validation", [("hybrid_supervised", p, s) for s in validation_seeds for p in PRESETS])
    validation_rows = [run_row(*key, episodes, result, V, limit) for key, result in validation.items()]
    retained = retained_preset(validation_rows)
    print(f"Réglage retenu sur la validation : {retained}", flush=True)
    test = run_phase("test", [(m, p, s) for m in METHODS for p in PRESETS for s in test_seeds])
    test_rows = [run_row(*key, episodes, result, V, limit) for key, result in test.items()]
    curves = {key: result["curve"] for key, result in {**validation, **test}.items()}

    eval_every = preset_config("hybrid_supervised", "config", episodes)["eval_every"]
    hands = {name: split["samples"] for name, split in predictor.metadata["splits"].items()}
    budgets = {"train_hands": episodes - hands["train"], "all_hands": episodes - sum(hands.values())}
    manifest = json.loads((ROOT / "docs/wassim_source.json").read_text(encoding="utf-8"))
    summary = {
        "question": "Hybride supervisé : CONFIG (précédent) ou TUNED['hybrid'] (58f5ce3), à budget RL égal ?",
        "protocol": {
            "episodes_per_run": episodes, "validation_seeds": validation_seeds, "test_seeds": test_seeds,
            "decision_rule": "meilleur gain exact moyen de l'hybride supervisé sur les graines de validation ; "
                             "égalité → config ; le test ne sert à aucun choix.",
            "metric": "gain exact J de la politique gloutonne finale (récurrence de Wassim, vraie loi du croupier).",
            "paired_statistics": "différences par graine ; test exact de permutation des signes, bilatéral.",
            "presets": {"config": preset_config("hybrid_supervised", "config", episodes),
                        "tuned_hybrid": preset_config("hybrid_supervised", "tuned", episodes),
                        "tuned_standard": preset_config("standard", "tuned", episodes)},
        },
        "dealer_model": {"sha256": file_sha256(model_path), "simulated_hands": hands},
        "optimal_exact_expected_reward": J_STAR,
        "supervised_limit": {
            "exact_expected_reward": limit[1], "gap_to_optimal": J_STAR - limit[1],
            "states_differing_from_optimal": int(np.count_nonzero((limit[0] != PI_STAR) & REACH)),
            "reachable_states": int(REACH.sum())},
        "validation": {
            "arms": [arm_summary(validation_rows, "hybrid_supervised", p) for p in PRESETS],
            "tuned_minus_config": paired_difference(
                validation_rows, ("hybrid_supervised", "config"), ("hybrid_supervised", "tuned")),
            "retained_preset": retained},
        "test": {
            "arms": [arm_summary(test_rows, m, p) for m in METHODS for p in PRESETS],
            "comparisons": {
                "supervised_tuned_minus_config": paired_difference(
                    test_rows, ("hybrid_supervised", "config"), ("hybrid_supervised", "tuned")),
                "exact_tuned_minus_config": paired_difference(
                    test_rows, ("hybrid_exact", "config"), ("hybrid_exact", "tuned")),
                "supervised_minus_exact_retained": paired_difference(
                    test_rows, ("hybrid_exact", retained), ("hybrid_supervised", retained)),
                "supervised_retained_minus_standard_tuned": paired_difference(
                    test_rows, ("standard", "tuned"), ("hybrid_supervised", retained))},
            "gap_to_optimal_by_rl_episodes": [
                {"method": m, "preset": p, **{str(n): mean_gap_at(curves, m, p, n, eval_every)
                                              for n in CHECKPOINTS if n <= episodes}}
                for m in METHODS for p in PRESETS],
            "total_budget": {
                "total_simulations": episodes, "supervised_rl_episodes": budgets,
                "gap_to_optimal_mean": [
                    {"method": "hybrid_supervised", "preset": p, "dealer_hands_counted": counted,
                     "rl_episodes": n, "value": mean_gap_at(curves, "hybrid_supervised", p, n, eval_every)}
                    for counted, n in budgets.items() for p in PRESETS] + [
                    {"method": m, "preset": p, "dealer_hands_counted": "none", "rl_episodes": episodes,
                     "value": mean_gap_at(curves, m, p, episodes, eval_every)}
                    for m in ("standard", "hybrid_exact") for p in PRESETS]}},
        "checks": {"stand_frozen_in_hybrid_runs": True,
                   "legacy_agent_reproduced": legacy_reproduced(test, model_path, episodes)},
        "provenance": {
            "wassim_source_commit": manifest["commit"],
            "rl_source_sha256": {name: file_sha256(ROOT / name) for name in (
                "src/environment.py", "src/q_learning.py", "src/stand_value.py", "src/evaluate.py")},
            "versions": {"python": platform.python_version(), **{
                name: version(name) for name in ("numpy", "gymnasium", "scikit-learn", "joblib")}},
            "wall_seconds": time.perf_counter() - started},
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = validation_rows + test_rows
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "learning_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "method", "preset", "seed", "episodes", "exact_expected_reward"])
        writer.writerows((*key, (i + 1) * eval_every, f"{score:.10f}")
                         for key, curve in curves.items() for i, score in enumerate(curve))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(summary, output_dir)
    print(f"Comparaison écrite dans {output_dir}", flush=True)
    return summary


def write_report(summary: dict, output: Path) -> None:
    protocol, validation, test = summary["protocol"], summary["validation"], summary["test"]
    cfg, tuned = protocol["presets"]["config"], protocol["presets"]["tuned_hybrid"]
    limit, hands = summary["supervised_limit"], summary["dealer_model"]["simulated_hands"]
    retained = validation["retained_preset"]

    def seeds(values: list[int]) -> str:
        consecutive = values == list(range(values[0], values[-1] + 1))
        return f"{values[0]}–{values[-1]}" if consecutive else ", ".join(map(str, values))

    def paired_line(name: str, comparison: dict) -> str:
        return (f"- {name} : {comparison['mean']:+.6f} en moyenne (erreur type {comparison['standard_error']:.6f}) ; "
                f"meilleur sur {comparison['candidate_better']}, égal sur {comparison['equal']}, moins bon sur "
                f"{comparison['candidate_worse']} des {comparison['seeds']} graines ; p exact = "
                f"{comparison['sign_flip_p_value']:.2g}.")

    def arm_table(arms: list[dict]) -> list[str]:
        return ["| Méthode | Réglage | eps_min | J moyen | écart-type | écart à l'optimum | écart à la limite "
                "| états ≠ π* |", "|---|---|---:|---:|---:|---:|---:|---:|",
                *[f"| {METHOD_NAMES[arm['method']]} | {arm['preset']} | {arm['eps_min']:g} | "
                  f"{arm['exact_expected_reward_mean']:+.6f} | {arm['exact_expected_reward_sd']:.6f} | "
                  f"{arm['gap_to_optimal_mean']:.6f} | {arm['gap_to_limit_mean']:.6f} | "
                  f"{arm['states_differing_from_optimal_mean']:.1f} |" for arm in arms]]

    supervised_test = test["comparisons"]["supervised_tuned_minus_config"]
    lines = [
        "# Comparaison CONFIG / TUNED de l'hybride supervisé", "",
        f"Sources RL de Wassim : commit `{summary['provenance']['wassim_source_commit']}`.",
        f"Prédicteur supervisé figé : `{summary['dealer_model']['sha256']}`.", "",
        "## Question et réglages", "",
        "À budget RL égal, l'agent hybride qui utilise notre V_STAND estimée apprend-il mieux",
        "avec la configuration précédente `CONFIG` ou avec `TUNED[\"hybrid\"]` du commit 58f5ce3 ?", "",
        "| Réglage | eps0 | eps_min | decay | omega | gamma |", "|---|---:|---:|---:|---:|---:|",
        *[f"| {name} | {c['eps0']:g} | {c['eps_min']:g} | {c['decay']:g} | {c['omega']:g} | {c['gamma']:g} |"
          for name, c in (("CONFIG (précédent)", cfg), ("TUNED[\"hybrid\"]", tuned))], "",
        "Avec eps_min = 1, le comportement pendant l'apprentissage est entièrement aléatoire ; la politique",
        "évaluée reste gloutonne, le Q-learning étant hors politique. Wassim a réglé TUNED avec la V_STAND",
        "exacte (graines 100–102) : ce réglage n'était pas encore vérifié avec notre V_STAND estimée.", "",
        "## Protocole", "",
        f"- Budget : {protocol['episodes_per_run']:,} épisodes RL par entraînement, identique pour tous les bras.",
        "- Les deux réglages de l'hybride supervisé partagent le même prédicteur, donc le même coût de",
        f"  simulation du croupier : {hands['train']:,} mains d'entraînement, plus {hands['validation']:,} de",
        f"  validation et {hands['test']:,} de test pour le choisir et l'évaluer.",
        f"- Validation : graines {seeds(protocol['validation_seeds'])}, hybride supervisé seulement.",
        f"  Règle fixée avant l'exécution : {protocol['decision_rule']}.",
        f"- Test : graines {seeds(protocol['test_seeds'])}, sans aucun choix. Références au même budget RL :",
        "  Q-learning standard (CONFIG ou TUNED[\"standard\"], eps_min = 0.4) et hybride avec V_STAND exacte.",
        f"- Métrique : {protocol['metric']}",
        "  Elle est sans bruit pour une politique donnée ; la variabilité vient de l'entraînement RL.",
        f"- Statistiques : {protocol['paired_statistics']}.", "",
        "## Validation et choix", "",
        *arm_table(validation["arms"]), "",
        paired_line("TUNED − CONFIG", validation["tuned_minus_config"]), "",
        f"**Réglage retenu : `{retained}`.**", "",
        "## Test", "",
        *arm_table(test["arms"]), "",
        "Différences appariées sur les graines de test (candidat − référence) :", "",
        paired_line("hybride supervisé, TUNED − CONFIG", supervised_test),
        paired_line("hybride exact, TUNED − CONFIG", test["comparisons"]["exact_tuned_minus_config"]),
        paired_line(f"réglage {retained}, hybride supervisé − hybride exact",
                    test["comparisons"]["supervised_minus_exact_retained"]),
        paired_line(f"hybride supervisé ({retained}) − standard (TUNED)",
                    test["comparisons"]["supervised_retained_minus_standard_tuned"]), "",
        "Ces quatre comparaisons sont descriptives ; seule la validation a servi à décider.", "",
        "### D'où vient l'écart de l'hybride supervisé", "",
        f"La politique limite, optimale pour notre V_STAND estimée, a un gain exact de "
        f"{limit['exact_expected_reward']:+.9f}.",
        f"Elle ne diffère de π* que sur {limit['states_differing_from_optimal']} des {limit['reachable_states']} "
        f"états atteignables ; l'écart dû au prédicteur vaut {limit['gap_to_optimal']:.6f}.",
        "La colonne « écart à la limite » est la différence de gain réel entre cette politique limite",
        "et la politique apprise. Ce terme n'est pas garanti positif : l'optimum du modèle estimé",
        "n'est pas une borne supérieure du gain dans le vrai environnement.", "",
        "### Écart moyen à l'optimum selon le nombre d'épisodes RL (test)", "",
    ]
    checkpoints = [key for key in test["gap_to_optimal_by_rl_episodes"][0] if key not in ("method", "preset")]
    lines.extend([
        "| Méthode | Réglage | " + " | ".join(f"{int(n):,}" for n in checkpoints) + " |",
        "|---|---|" + "---:|" * len(checkpoints),
        *[f"| {METHOD_NAMES[row['method']]} | {row['preset']} | "
          + " | ".join(f"{row[n]:.6f}" if row[n] is not None else "—" for n in checkpoints) + " |"
          for row in test["gap_to_optimal_by_rl_episodes"]], "",
        "### Budget total de simulations", "",
        f"Si l'on compte les mains simulées pour le prédicteur, un budget total de "
        f"{test['total_budget']['total_simulations']:,} simulations laisse à l'hybride supervisé "
        f"{test['total_budget']['supervised_rl_episodes']['train_hands']:,} épisodes RL (mains d'entraînement "
        f"seules) ou {test['total_budget']['supervised_rl_episodes']['all_hands']:,} (avec validation et test).", "",
        "| Méthode | Réglage | Mains du croupier comptées | Épisodes RL | Écart moyen à l'optimum |",
        "|---|---|---|---:|---:|",
        *[f"| {METHOD_NAMES[row['method']]} | {row['preset']} | {HANDS_COUNTED[row['dealer_hands_counted']]} | "
          f"{row['rl_episodes']:,} | " + (f"{row['value']:.6f}" if row["value"] is not None else "—") + " |"
          for row in test["total_budget"]["gap_to_optimal_mean"]], "",
        "L'hybride exact utilise la loi analytique du croupier : son coût de simulation n'est pas compté,",
        "c'est une référence avec probabilités exactes, pas une borne de performance à budget RL fini.",
        "Seule la politique optimale π* fournit la borne supérieure du gain espéré.", "",
        "## Vérifications", "",
        "- STAND est resté figé dans tous les entraînements hybrides.",
    ])
    legacy = summary["checks"]["legacy_agent_reproduced"]
    lines.append({True: "- Le run CONFIG de graine 0 reproduit bit à bit l'ancien agent `models/hybrid/agent_config.npz`.",
                  False: "- Le run CONFIG de graine 0 ne reproduit pas l'ancien agent `models/hybrid/agent_config.npz`.",
                  None: "- Reproduction de l'ancien agent non vérifiable avec ces paramètres."}[legacy])
    lines.extend([
        "", "## Limites", "",
        f"- {len(protocol['test_seeds'])} graines de test : les conclusions portent sur ce budget de "
        f"{protocol['episodes_per_run']:,} épisodes et ce prédicteur.",
        "- Un seul prédicteur supervisé (une graine de données) : la variabilité due aux données",
        "  supervisées n'est pas mesurée ici.",
        "- Le test énumère exactement les permutations de signes des différences appariées. Sa validité",
        "  suppose l'échangeabilité des deux traitements au sein des paires sous H0 (ou la symétrie des",
        "  différences autour de zéro). L'égalité des moyennes seule ne suffit pas. Les p ne corrigent",
        "  pas la multiplicité ; ces comparaisons sont descriptives et ne prouvent aucune équivalence.",
        "- J utilise la vraie loi du croupier pour évaluer. L'hybride supervisé ne l'utilise pas pour",
        "  ses mises à jour ; l'hybride exact, par définition, la reçoit pour calculer STAND.",
        "- Le budget total additionne mains du croupier et épisodes RL : ce sont des unités de simulation",
        "  différentes, pas une égalisation du temps de calcul. Le coût du réglage n'est pas inclus.", "",
        "## Fichiers", "",
        "- `runs.csv` : une ligne par entraînement (phase, méthode, réglage, graine, métriques).",
        f"- `learning_curves.csv` : gain exact tous les {cfg['eval_every']:,} épisodes pour chaque entraînement.",
        "- `summary.json` : protocole, résumés, tests appariés, versions et empreintes.", "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=-1, help="Processus parallèles (joblib) ; -1 : tous les cœurs.")
    args = parser.parse_args()
    try:
        compare(args.model, args.episodes, args.output_dir, jobs=args.jobs)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
