import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pytest

from src.environment import HIT, STAND
from src.evaluate_hybrid import evaluate_agent
from src.hybrid_agent import DEFAULT_AGENT, ROOT, HybridAgent, file_sha256, load_agent, results_dir_for
from src.supervised.model import DEFAULT_MODEL
from src.supervised.values import stand_value
from src.train_hybrid import PRESETS, train_hybrid

LEGACY_AGENT = ROOT / "models/hybrid/agent.npz"
MANIFEST = json.loads((ROOT / "docs/wassim_source.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hybrid")
    agent = train_hybrid(episodes=800, seed=11, agent_path=folder / "agent.npz", results_dir=folder / "results",
                         preset="config")
    return agent, folder


def test_training_uses_supervised_stand_and_preserves_it(trained):
    agent, folder = trained
    assert agent.metadata["stand_frozen"]
    assert agent.metadata["dealer_model_sha256"] == file_sha256(DEFAULT_MODEL)
    assert agent.metadata["training_config"]["episodes"] == 800
    assert agent.visits.sum() > 0
    np.testing.assert_array_equal(agent.stand_values, agent.predictor.stand_values())
    metadata = json.loads((folder / "results/training.json").read_text(encoding="utf-8"))
    assert metadata["training_config"]["gamma"] == 1


def test_decision_compares_both_values_and_ties_stand(trained):
    agent, _ = trained
    state = (15, 10, 0)
    q = agent.stand_values.copy()
    tied = HybridAgent(agent.predictor, q, agent.visits, agent.metadata)
    assert tied.act(state) == STAND
    q[state] += 0.1
    hit = HybridAgent(agent.predictor, q, agent.visits, agent.metadata)
    assert hit.act(state) == HIT
    q[state] -= 0.2
    stand = HybridAgent(agent.predictor, q, agent.visits, agent.metadata)
    assert stand.act(state) == STAND
    decision = stand.decide(state)
    assert decision.stand_value == pytest.approx(stand_value(15, stand.dealer_probabilities(10)))


@pytest.mark.parametrize("state", [(22, 10, 0), (11, 10, 1), (20, 11, 0), (3, 1, 0),
                                   (20.0, 10, 0), (20, 10, 2), (20, 10), np.array(1), None])
def test_invalid_states_rejected(trained, state):
    with pytest.raises(ValueError):
        trained[0].act(state)


def test_boolean_usable_ace_is_supported(trained):
    agent, _ = trained
    assert agent.act((18, 5, True)) == agent.act((18, 5, 1))


def test_save_and_load_keep_policy(trained, tmp_path):
    agent, _ = trained
    path = tmp_path / "agent.npz"
    agent.save(path)
    loaded = load_agent(path)
    np.testing.assert_array_equal(loaded.policy(), agent.policy())
    np.testing.assert_array_equal(loaded.visits, agent.visits)
    assert loaded.decide((15, 10, 0)) == agent.decide((15, 10, 0))


def test_changed_supervised_model_requires_matching_rl(trained, tmp_path):
    _, folder = trained
    changed = joblib.load(DEFAULT_MODEL)
    changed["metadata"]["different_experiment"] = True
    other_model = tmp_path / "other.joblib"
    joblib.dump(changed, other_model)
    with pytest.raises(ValueError, match="ne correspond pas"):
        load_agent(folder / "agent.npz", other_model)


@pytest.mark.parametrize("field", ["stand_values", "q_hit", "visits"])
def test_inconsistent_saved_tables_rejected(trained, tmp_path, field):
    _, folder = trained
    with np.load(folder / "agent.npz", allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    if field == "stand_values":
        payload[field][15, 10, 0] += 0.1
    elif field == "q_hit":
        payload[field][15, 10, 0] = np.nan
    else:
        payload[field][15, 10, 0] = -1
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        load_agent(path)


def test_evaluation_is_reproducible_without_learning(trained):
    agent, _ = trained
    q_before, visits_before = agent.q_hit.copy(), agent.visits.copy()
    result = evaluate_agent(agent, games=100, seed=90)
    assert result == evaluate_agent(agent, games=100, seed=90)
    assert result["wins"] + result["draws"] + result["losses"] == 100
    assert result["mean_reward"] == (result["wins"] - result["losses"]) / 100
    assert result["gap_to_optimal_exact"] >= -1e-12
    np.testing.assert_array_equal(agent.q_hit, q_before)
    np.testing.assert_array_equal(agent.visits, visits_before)
    with pytest.raises(ValueError, match="graine"):
        evaluate_agent(agent, games=100, seed=11)


def test_cannot_overwrite_supervised_model_with_rl_archive(tmp_path):
    path = tmp_path / "same.joblib"
    with pytest.raises(ValueError, match="fichiers distincts"):
        train_hybrid(model_path=path, agent_path=path, preset="config")
    assert not path.exists()


@pytest.mark.parametrize(("preset", "eps_min"), [("config", 0.05), ("tuned", 1.0)])
def test_preset_is_recorded_with_omega(tmp_path, preset, eps_min):
    agent = train_hybrid(episodes=300, seed=5, agent_path=tmp_path / "agent.npz",
                         results_dir=tmp_path / "results", preset=preset)
    saved = json.loads((tmp_path / "results/training.json").read_text(encoding="utf-8"))
    for metadata in (agent.metadata, saved, load_agent(tmp_path / "agent.npz").metadata):
        assert metadata["training_preset"] == preset
        assert metadata["training_config"]["eps_min"] == eps_min
        assert metadata["training_config"]["omega"] == 1.0
        assert metadata["training_config"]["gamma"] == 1.0
        assert metadata["wassim_source_commit"] == MANIFEST["commit"]
    np.testing.assert_array_equal(agent.stand_values, agent.predictor.stand_values())


def test_unknown_preset_rejected(tmp_path):
    with pytest.raises(ValueError, match="Réglage inconnu"):
        train_hybrid(agent_path=tmp_path / "agent.npz", results_dir=tmp_path / "results", preset="standard")
    assert not (tmp_path / "agent.npz").exists()


def test_local_rl_sources_match_manifest():
    for entry in MANIFEST["files"]:
        assert file_sha256(ROOT / entry["local_path"]) == entry["sha256"], entry["local_path"]


def test_legacy_agent_is_preserved_and_still_loads():
    legacy_evaluation = json.loads((ROOT / "results/hybrid/evaluation.json").read_text(encoding="utf-8"))
    assert file_sha256(LEGACY_AGENT) == legacy_evaluation["agent_sha256"]
    agent = load_agent(LEGACY_AGENT)
    assert agent.metadata["wassim_source_commit"].startswith("0780a4f")
    decision = agent.decide((15, 10, 0))
    assert decision.action == HIT
    assert decision.stand_value == pytest.approx(-0.582872, abs=1e-6)
    assert decision.hit_value == pytest.approx(-0.546372, abs=1e-6)
    # Les sorties par défaut ne visent plus l'ancien agent ni ses résultats.
    assert DEFAULT_AGENT != LEGACY_AGENT and results_dir_for(DEFAULT_AGENT) != ROOT / "results/hybrid"


def test_default_agent_uses_new_rl_code_and_plays_complete_games():
    agent = load_agent()
    assert agent.metadata["wassim_source_commit"] == MANIFEST["commit"]
    assert agent.metadata["training_preset"] in PRESETS
    assert all(agent.act((21, dealer, 0)) == STAND for dealer in range(1, 11))
    result = evaluate_agent(agent, games=200, seed=7)
    assert result["wins"] + result["draws"] + result["losses"] == 200
    assert result["decisions"] >= 200


@pytest.mark.parametrize("args", [["--state", "15", "10", "0"], ["--games", "2", "--seed", "2030", "--trace"]])
def test_play_output_fits_windows_code_page(args):
    # Sous Windows, une sortie redirigée (fichier, pipe) est encodée en cp1252, pas en UTF-8.
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    run = subprocess.run([sys.executable, "-m", "src.play", *args], cwd=ROOT, env=env, capture_output=True)
    assert run.returncode == 0, run.stderr.decode("cp1252", "replace")


@pytest.mark.parametrize("preset", ["config", "tuned"])
def test_training_requires_explicit_output(preset):
    with pytest.raises(ValueError, match="explicitement"):
        train_hybrid(preset=preset)


def test_overwrite_is_explicit_and_invalidates_previous_evaluation(tmp_path):
    path, output = tmp_path / "agent.npz", tmp_path / "results"
    train_hybrid(agent_path=path, results_dir=output, episodes=200, seed=1, preset="config")
    original = path.read_bytes()
    (output / "evaluation.json").write_text('{"stale": true}', encoding="utf-8")
    (output / "report.md").write_text("Ancien rapport", encoding="utf-8")
    with pytest.raises(ValueError, match="déjà existante"):
        train_hybrid(agent_path=path, results_dir=output, episodes=200, seed=2, preset="tuned")
    assert path.read_bytes() == original
    assert (output / "evaluation.json").exists()
    train_hybrid(agent_path=path, results_dir=output, episodes=200, seed=2, preset="tuned", overwrite=True)
    assert load_agent(path).metadata["seed"] == 2
    assert not (output / "evaluation.json").exists()
    assert not (output / "report.md").exists()


def test_result_directory_collision_is_rejected_before_training(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "report.md").write_text("À conserver", encoding="utf-8")
    with pytest.raises(ValueError, match="déjà existante"):
        train_hybrid(agent_path=tmp_path / "new.npz", results_dir=output, preset="config")
    assert not (tmp_path / "new.npz").exists()
