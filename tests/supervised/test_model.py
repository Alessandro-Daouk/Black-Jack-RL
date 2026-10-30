import joblib
import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.supervised.constants import CLASSES
from src.supervised.data import empirical_probabilities, generate_dataset
from src.supervised.evaluate import metrics, sample_losses
from src.supervised.model import DealerPredictor, fit_classifier, load_model
from src.supervised.reference import exact_probabilities
from src.supervised.train import run_experiment


@pytest.fixture(scope="module")
def trained():
    X, y = generate_dataset(6_000, seed=5)
    return DealerPredictor(fit_classifier(X, y), {"empirical_probabilities": empirical_probabilities(X, y).tolist()})


def test_all_cards_and_serialization(trained, tmp_path):
    path = tmp_path / "model.joblib"
    trained.save(path)
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.probability_matrix(), trained.probability_matrix())
    for card in range(1, 11):
        probabilities = loaded.predict_dealer_probabilities(card)
        assert tuple(probabilities) == CLASSES
        assert all(0 <= p <= 1 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 1e-12
        np.testing.assert_allclose(loaded.probability_vectors()[card], list(probabilities.values()))


@pytest.mark.parametrize("card", [0, 11, -1, 1.5, 10.0, True, "10", float("nan")])
def test_invalid_cards_rejected(trained, card):
    with pytest.raises(ValueError):
        trained.predict_dealer_probabilities(card)


def test_class_order_is_checked(trained, tmp_path):
    payload = {"schema_version": 1, "pipeline": trained.pipeline, "metadata": {}}
    path = tmp_path / "wrong.joblib"
    joblib.dump(payload, path)
    altered = joblib.load(path)
    altered["pipeline"].named_steps["classifier"].classes_ = np.array([22, 21, 20, 19, 18, 17])
    joblib.dump(altered, path)
    with pytest.raises(ValueError, match="Ordre des classes"):
        load_model(path)


def test_incomplete_training_data_rejected():
    with pytest.raises(ValueError):
        fit_classifier(np.ones((6, 1), dtype=int), np.array(CLASSES))
    with pytest.raises(ValueError):
        empirical_probabilities(np.ones((6, 1), dtype=int), np.array(CLASSES))


def test_metrics_match_sklearn_and_oracle_has_zero_error(trained):
    X, y = generate_dataset(1_000, seed=61)
    probabilities = trained.probability_matrix()
    np.testing.assert_allclose(sample_losses(probabilities, X, y).mean(),
                               log_loss(y, probabilities[X[:, 0] - 1], labels=list(CLASSES)))
    exact = exact_probabilities()
    result = metrics(exact, X, y, exact)
    assert result["max_stand_error"] == 0
    assert result["max_total_variation"] == 0
    assert result["exact_excess_log_loss"] == 0
    assert metrics(probabilities, X, y, exact)["exact_excess_log_loss"] > 0


def test_experiment_rejects_reused_seeds_before_writing(tmp_path):
    with pytest.raises(ValueError, match="distinctes"):
        run_experiment(train_seed=42, validation_seed=42, model_dir=tmp_path / "model")
    assert not (tmp_path / "model").exists()
