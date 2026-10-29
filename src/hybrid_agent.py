"""Agent hybride : probabilités supervisées pour STAND, Q_HIT appris pour HIT."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from src.environment import HIT, SHAPE, STAND
from src.supervised.constants import CLASSES
from src.supervised.data import require_integer
from src.supervised.model import DEFAULT_MODEL, DealerPredictor, load_model

ROOT = Path(__file__).resolve().parents[1]
# Agent retenu par la comparaison CONFIG/TUNED ; l'ancien agent reste dans models/hybrid/agent.npz.
DEFAULT_AGENT = ROOT / "models/hybrid/agent_tuned.npz"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def results_dir_for(agent_path: Path) -> Path:
    """Dossier de résultats par défaut d'un agent : results/hybrid/<nom du fichier>/."""
    return ROOT / "results/hybrid" / Path(agent_path).stem


def validate_state(state) -> tuple[int, int, int]:
    if (not isinstance(state, (tuple, list, np.ndarray))
            or (isinstance(state, np.ndarray) and state.ndim != 1) or len(state) != 3):
        raise ValueError("État requis : (total_joueur, carte_visible, as_utilisable).")
    total = require_integer(state[0], "total_joueur", 4, 21)
    dealer = require_integer(state[1], "carte_visible", 1, 10)
    ace = int(state[2]) if isinstance(state[2], (bool, np.bool_)) else require_integer(state[2], "as_utilisable", 0, 1)
    if ace and total < 12:
        raise ValueError("Un as utilisable nécessite un total d'au moins 12.")
    return total, dealer, ace


@dataclass(frozen=True)
class Decision:
    state: tuple[int, int, int]
    action: int
    stand_value: float
    hit_value: float
    hit_visits: int

    @property
    def action_name(self) -> str:
        return "HIT" if self.action == HIT else "STAND"


class HybridAgent:
    """Charge les deux composants une fois ; aucune exploration pendant le jeu.

    Seules dix cartes peuvent être visibles : les probabilités et V_STAND sont
    précalculées au chargement. À chaque état, l'agent compare V_STAND et Q_HIT.
    Une égalité conduit à STAND, conformément au code de Wassim.
    """

    def __init__(self, predictor: DealerPredictor, q_hit: np.ndarray,
                 visits: np.ndarray, metadata: dict):
        self.predictor = predictor
        self.metadata = dict(metadata)
        q_hit = np.asarray(q_hit, dtype=float)
        visits = np.asarray(visits)
        if q_hit.shape != SHAPE or not np.all(np.isfinite(q_hit)):
            raise ValueError(f"Q_HIT doit être un tableau fini de forme {SHAPE}.")
        if np.any(np.abs(q_hit) > 1 + 1e-8):
            raise ValueError("Q_HIT doit rester entre -1 et +1 pour ces récompenses et gamma=1.")
        if visits.shape != SHAPE or not np.issubdtype(visits.dtype, np.integer) or np.any(visits < 0):
            raise ValueError("Les nombres de visites doivent être des entiers positifs ou nuls.")
        self.q_hit = q_hit.copy()
        self.visits = visits.copy()
        self.probabilities = predictor.probability_matrix()
        self.stand_values = predictor.stand_values()
        for array in (self.q_hit, self.visits, self.probabilities, self.stand_values):
            array.setflags(write=False)

    def decide(self, state) -> Decision:
        state = validate_state(state)
        v_stand = float(self.stand_values[state])
        q_hit = float(self.q_hit[state])
        return Decision(state, HIT if q_hit > v_stand else STAND, v_stand, q_hit, int(self.visits[state]))

    def act(self, state) -> int:
        return self.decide(state).action

    def dealer_probabilities(self, dealer_showing: int) -> dict[int, float]:
        dealer_showing = require_integer(dealer_showing, "carte_visible", 1, 10)
        return dict(zip(CLASSES, map(float, self.probabilities[dealer_showing - 1]), strict=True))

    def policy(self) -> np.ndarray:
        return self.q_hit > self.stand_values

    def save(self, path: Path = DEFAULT_AGENT) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, q_hit=self.q_hit, visits=self.visits,
                                stand_values=self.stand_values,
                                metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)))


def load_agent(agent_path: Path = DEFAULT_AGENT, model_path: Path = DEFAULT_MODEL) -> HybridAgent:
    """Charge une Q-table associée exactement au prédicteur qui a servi à son entraînement."""
    with np.load(agent_path, allow_pickle=False) as archive:
        required = {"q_hit", "visits", "stand_values", "metadata"}
        if not required.issubset(archive.files):
            raise ValueError("Archive hybride incomplète.")
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema_version") != 1 or metadata.get("kind") != "hybrid_supervised":
            raise ValueError("Format d'agent hybride incompatible.")
        if metadata.get("dealer_model_sha256") != file_sha256(model_path):
            raise ValueError("Ce prédicteur ne correspond pas à celui utilisé pour entraîner Q_HIT. "
                             "Utiliser le modèle associé ou réentraîner l'hybride.")
        if metadata.get("training_config", {}).get("gamma") != 1.0:
            raise ValueError("Cet agent requiert gamma=1 pour comparer des gains non actualisés.")
        predictor = load_model(model_path)
        agent = HybridAgent(predictor, archive["q_hit"], archive["visits"], metadata)
        if archive["stand_values"].shape != SHAPE or not np.allclose(
                archive["stand_values"], agent.stand_values, atol=1e-12, rtol=0):
            raise ValueError("La table STAND sauvegardée ne correspond pas au prédicteur.")
    return agent
