from pathlib import Path
import shutil
import subprocess

import pytest

from src.export_web import OUTPUT, render

ROOT = Path(__file__).resolve().parents[1]


def test_exported_models_are_up_to_date():
    assert OUTPUT.read_text(encoding="utf-8") == render(), "Relancer : python -m src.export_web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js absent")
def test_game_rules_in_javascript():
    # dont : le gain simulé de chaque modèle en JavaScript retrouve son gain exact calculé en Python
    result = subprocess.run(["node", "--test", "web/game.test.js"], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
