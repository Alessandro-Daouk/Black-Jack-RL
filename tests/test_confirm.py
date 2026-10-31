import json

from joblib import parallel_backend

from src.shoe.confirm import run


def test_confirmation_runs_on_a_tiny_plan(tmp_path):
    plan = json.loads(open("results/6_confirmation_sabot/plan.json", encoding="utf-8").read())
    plan.update(groupes=2, mains_modele1=2500, validation=800, test_modele1=50, sabots_evaluation=[9000000, 9000040],
                budgets=[100, 200])
    for h in plan["hypotheses"]:
        if "parties" in h:
            h["parties"] = 200
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with parallel_backend("threading", n_jobs=4):  # les processus ne verraient pas le plan réduit
        run(path, tmp_path / "out")
    verdicts = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))["verdicts"]
    assert set(verdicts) == {"H1", "H2", "H3", "H4"}
    assert all(v["ci"][0] <= v["mean"] <= v["ci"][1] and isinstance(v["confirmee"], bool) for v in verdicts.values())
