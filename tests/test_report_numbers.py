"""Les tableaux du rapport et des annexes recopient bien les fichiers de résultats.

Chaque ligne citée est recalculée depuis results/ ; chaque nombre décimal affiché doit être à moins
d'une unité de sa dernière décimale (un arrondi prudent vers l'extérieur reste donc admis), chaque
entier doit être exact.
"""
import csv
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT, ANNEX = "docs/rapport_scientifique.md", "docs/annexes.md"
NUMBER = re.compile(r"[+−-]?\d+(?:,\d+)?")


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def numbers(line, columns):
    cells = line.strip("|").split("|")[1:]
    text = " ".join(cells if columns is None else [cells[c] for c in columns])
    return [(float(m.replace("−", "-").replace(",", ".")), len(m.split(",")[1]) if "," in m else 0)
            for m in NUMBER.findall(text)]


def check(doc, label, expected, columns=None):
    """Un même libellé peut figurer dans plusieurs tableaux : on garde la ligne qui a autant de nombres."""
    lines = [line for line in (ROOT / doc).read_text(encoding="utf-8").splitlines() if line.startswith(f"| {label} |")]
    candidates = [numbers(line, columns) for line in lines if len(numbers(line, columns)) == len(expected)]
    assert len(candidates) == 1, f"{doc} : {len(candidates)} ligne(s) « {label} » à {len(expected)} nombres"
    shown = candidates[0]
    for (value, decimals), true in zip(shown, expected, strict=True):
        tolerance = 10 ** -decimals if decimals else 0
        assert abs(value - true) <= tolerance + 1e-12, f"{doc} « {label} » : {value} affiché, {true} attendu"


def ci(entry, key="ci95"):
    return [entry["mean"], *entry[key]]


@pytest.fixture(scope="module")
def curve_regret():
    """Regret moyen par méthode et budget sur les 9 jeux secondaires, recalculé depuis les courbes brutes."""
    sums = {}
    with (ROOT / "results/2_campagne_principale/entrainements/curves.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["group"]) < 9 and r["n"] in ("0", "20000"):
                key = (r["method"], int(r["episodes"]))
                s = sums.setdefault(key, [0., 0])
                s[0] += float(r["regret"]); s[1] += 1
    return {k: v[0] / v[1] for k, v in sums.items()}


def test_main_criterion():
    primary = load("results/2_campagne_principale/analyse/summary.json")["primary"]
    check(REPORT, "J(logistique) − J(standard réglé), 28 jeux", [primary["mean"]])
    check(REPORT, "IC 95 % Student ; bootstrap", [*primary["ci95"], *primary["bootstrap_cluster_ci95"]])
    check(REPORT, "Étendue des 28 différences", [primary["min"], primary["max"]])


def test_regret_tables(curve_regret):
    e9 = load("results/4_autres_methodes/autres_algorithmes/summary.json")["contrasts"]
    main, annex = [1000, 10000, 40000, 100000, 200000], [1000, 5000, 10000, 20000, 40000, 100000, 200000]
    learned = {"RL à modèle appris": "model_based", "Dyna-Q": "dyna_q", "Expected SARSA": "expected_sarsa",
               "Contrôle Monte-Carlo": "monte_carlo"}
    for label, key in learned.items():
        for doc, budgets in ((REPORT, main), (ANNEX, annex)):
            check(doc, label, [e9[key]["regret_by_budget_secondary"][str(b)] for b in budgets])
    curves = {"**Hybride logistique**": "logistic", "Q-learning standard réglé": "standard"}
    for label, method in curves.items():
        for doc, budgets in ((REPORT, main), (ANNEX, annex)):
            check(doc, label, [curve_regret[method, b] for b in budgets])
    check(REPORT, "Hybride à STAND exacte", [curve_regret["exact", b] for b in main])
    check(ANNEX, "Hybride, STAND exacte (oracle)", [curve_regret["exact", b] for b in annex])


def test_contrasts_with_other_algorithms():
    e9 = load("results/4_autres_methodes/autres_algorithmes/summary.json")["contrasts"]
    double_q = load("results/4_autres_methodes/planification_double_q/summary.json")["double_q"]
    for label, key in {"RL à modèle appris": "model_based", "Dyna-Q": "dyna_q", "Expected SARSA": "expected_sarsa",
                       "Contrôle Monte-Carlo": "monte_carlo"}.items():
        check(ANNEX, label, [v for b in ("10000", "40000", "100000") for v in ci(e9[key][b]["hybrid_minus"])])
    check(ANNEX, "Double Q-learning", [v for b in ("10000", "40000", "100000")
                                       for v in ci(double_q[b]["hybrid_minus_double"])])


def test_finite_shoe_tables():
    shoe = load("results/5_sabot_fini/summary.json")
    sizes = [1000, 5000, 20000, 100000]
    for label, key in {"Carte": "carte", "Table": "table", "**Logistique**": "logistique", "Boosting": "boosting"}.items():
        check(REPORT, label, [shoe["supervised"][f"{key}_{n}"]["excess_kl"]["mean"] for n in sizes])

    def gains(agent, budgets):
        return [shoe["agents"][agent][str(b)]["minus_basic"]["difference"]["mean"] for b in budgets]

    short, full = [10000, 200000, 1000000, 3000000], [10000, 50000, 200000, 1000000, 3000000]
    for label, agent in {"Standard tabulaire": "standard", "Hybride carte": "hybride_carte",
                         "Hybride table": "hybride_table", "Hybride logistique (Q_HIT tabulaire)": "hybride_logistique",
                         "Standard linéaire": "standard_lineaire",
                         "**Hybride logistique linéaire**": "hybride_logistique_lineaire"}.items():
        check(REPORT, label, gains(agent, short))
    for label, agent in {"Standard": "standard", "Hybride carte": "hybride_carte", "Hybride table": "hybride_table",
                         "**Hybride logistique**": "hybride_logistique", "Standard (tabulaire, rappel)": "standard",
                         "Hybride logistique (rappel)": "hybride_logistique", "Standard linéaire": "standard_lineaire",
                         "**Hybride logistique linéaire**": "hybride_logistique_lineaire"}.items():
        check(ANNEX, label, gains(agent, full))

    def contrast(key, budgets):
        return [v for b in budgets for d in [shoe["contrasts"][key][str(b)]]
                for v in (*ci(d["difference"], "ci95_with_evaluation"), d["pairs_better"])]

    for label, key in {"Hybride logistique − standard tabulaire": "hybride_logistique-standard",
                       "Hybride table − standard tabulaire": "hybride_table-standard",
                       "Hybride logistique − standard linéaire": "hybride_logistique-standard_lineaire",
                       "Hybride logistique linéaire − standard linéaire":
                           "hybride_logistique_lineaire-standard_lineaire"}.items():
        check(REPORT, label, contrast(key, [200000, 3000000]))
    for label, key in {"Logistique − standard": "hybride_logistique-standard",
                       "Logistique − table": "hybride_logistique-hybride_table",
                       "Logistique − carte": "hybride_logistique-hybride_carte",
                       "Table − standard": "hybride_table-standard", "Carte − standard": "hybride_carte-standard"}.items():
        check(ANNEX, label, contrast(key, [200000, 1000000, 3000000]))
    for label, key in {"Standard linéaire − standard": "standard_lineaire-standard",
                       "Hybride logistique − standard linéaire": "hybride_logistique-standard_lineaire",
                       "Hybride logistique linéaire − standard linéaire": "hybride_logistique_lineaire-standard_lineaire",
                       "Hybride logistique linéaire − hybride logistique":
                           "hybride_logistique_lineaire-hybride_logistique"}.items():
        check(ANNEX, label, contrast(key, [10000, 200000, 1000000, 3000000]))


def test_confirmation_table():
    verdicts = load("results/6_confirmation_sabot/mesures/summary.json")["verdicts"]
    lines = [line for line in (ROOT / REPORT).read_text(encoding="utf-8").splitlines() if re.match(r"\| H\d\. ", line)]
    assert len(lines) == len(verdicts) == 4
    for line in lines:
        label = line.split("|")[1].strip()
        v = verdicts[label[:2]]
        check(REPORT, label, [v["mean"], *v["ci"]], columns=[0])
        assert line.split("|")[4].strip() == ("confirmée" if v["confirmee"] else "non confirmée")
