import csv
import json

import pytest

from src.compare_hybrid import compare, preset_config, retained_preset, sign_flip_p_value


def rows(config, tuned):
    return [{"method": "hybrid_supervised", "preset": preset, "seed": seed, "exact_expected_reward": value}
            for preset, values in (("config", config), ("tuned", tuned)) for seed, value in enumerate(values)]


def test_retained_preset_uses_validation_mean_and_tie_keeps_config():
    assert retained_preset(rows([-0.05, -0.04], [-0.05, -0.05])) == "config"
    assert retained_preset(rows([-0.05, -0.05], [-0.05, -0.04])) == "tuned"
    assert retained_preset(rows([-0.05, -0.04], [-0.04, -0.05])) == "config"


def test_sign_flip_p_value_is_exact():
    assert sign_flip_p_value([1.0, 2.0, 3.0]) == pytest.approx(2 / 8)
    assert sign_flip_p_value([1.0, -1.0]) == 1
    assert sign_flip_p_value([0.0, 0.0]) == 1


def test_tuned_depends_on_agent_type_and_keeps_omega():
    assert preset_config("hybrid_supervised", "config", 1_000)["eps_min"] == 0.05
    assert preset_config("hybrid_exact", "tuned", 1_000)["eps_min"] == 1.0
    assert preset_config("standard", "tuned", 1_000)["eps_min"] == 0.4
    assert all(preset_config(m, p, 1_000)["omega"] == 1.0 for m in ("standard", "hybrid_supervised")
               for p in ("config", "tuned"))


def test_small_comparison_writes_consistent_outputs(tmp_path):
    summary = compare(episodes=400, output_dir=tmp_path, validation_seeds=(1, 2), test_seeds=(3, 4), jobs=1)
    with (tmp_path / "runs.csv").open() as stream:
        runs = list(csv.DictReader(stream))
    validation = [dict(row, exact_expected_reward=float(row["exact_expected_reward"]))
                  for row in runs if row["phase"] == "validation"]
    assert len(validation) == 4 and len(runs) == 4 + 6 * 2
    assert summary["validation"]["retained_preset"] == retained_preset(validation)
    assert {row["seed"] for row in validation} == {"1", "2"}
    assert len(summary["test"]["arms"]) == 6
    assert summary["checks"]["legacy_agent_reproduced"] is None
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# Comparaison CONFIG / TUNED")


def test_validation_and_test_seeds_must_differ(tmp_path):
    with pytest.raises(ValueError, match="distinctes"):
        compare(episodes=10, output_dir=tmp_path, validation_seeds=(1, 2), test_seeds=(2, 3), jobs=1)
