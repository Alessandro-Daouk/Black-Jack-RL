import csv
import json

import numpy as np
import pytest

from src.compare_frequencies import run as ablation
from src.experiment_utils import describe
from src.supervised.learning_curve import run as learning_curve


def test_uncertainty_is_between_independent_replicates():
    stats = describe([1., 2., 3.])
    assert stats["mean"] == 2 and stats["sd"] == 1
    assert stats["mean_approx_95pct_interval"] == pytest.approx([-0.4841377117, 4.4841377117])


def test_learning_curve_records_independent_splits_and_validation_selection(tmp_path):
    summary = learning_curve(tmp_path, sizes=(800, 1600), repeats=2,
                             validation_size=400, test_size=500, jobs=1)
    fits = json.loads((tmp_path / "fits.json").read_text())
    seeds = [s for fit in fits for s in fit["seeds"].values()]
    assert len(set(seeds)) == len(seeds)
    for fit in fits:
        for size in fit["fits"]:
            best = min(size["candidates"], key=lambda c: c["validation_log_loss"])
            assert size["selected_C"] == best["C"]
            np.testing.assert_allclose(np.sum(size["logistic_probabilities"], axis=1), 1)
    rows = list(csv.DictReader((tmp_path / "runs.csv").open()))
    assert len(rows) == 8 and len(summary["arms"]) == 4
    assert all(float(r["exact_excess_log_loss"]) >= 0 for r in rows)
    with pytest.raises(ValueError, match="vide"):
        learning_curve(tmp_path)


def test_ablation_uses_all_methods_and_consistent_paired_scores(tmp_path):
    summary = ablation(tmp_path, episodes=200, seeds=(700, 701), jobs=1)
    rows = list(csv.DictReader((tmp_path / "runs.csv").open()))
    assert len(rows) == 8
    lookup = {(r["method"], int(r["seed"])): float(r["exact_expected_reward"]) for r in rows}
    expected = np.mean([lookup['logistic', s] - lookup['frequencies', s] for s in (700, 701)])
    assert summary["paired"]["logistic_minus_frequencies"]["mean"] == pytest.approx(expected)
    assert all(float(r["gap_to_optimal"]) >= -1e-12 for r in rows)
    with pytest.raises(ValueError, match="vide"):
        ablation(tmp_path)
