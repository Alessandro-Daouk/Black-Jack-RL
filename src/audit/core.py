"""Instrumentation v2 sans changer les trajectoires du Q-learning historique."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.stats import t

from src.environment import HIT, SHAPE, STAND, make_env
from src.q_learning import choose_action, update
from src.supervised.data import conditional_counts, draw_card, hand_value
from src.supervised.model import fit_classifier
from src.supervised.values import build_stand_table


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(config):
    def git(*args):
        return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()
    return {"config": config, "commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {n: version(n) for n in ("numpy", "scipy", "scikit-learn", "gymnasium", "joblib")},
            "thread_environment": {n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "source_sha256": {str(p): digest(p) for p in sorted(Path("src").rglob("*.py"))},
            "protocol_sha256": digest("docs/protocole_v2.md"),
            "protocol_addendum_sha256": digest("docs/protocole_v2_addendum.md") if Path("docs/protocole_v2_addendum.md").exists() else None,
            "lock_sha256": digest("uv.lock")}


def new_output(path, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Sortie non vide, refus d'écraser : {path}")
    dump(path / "provenance.json", provenance(config))
    return path


def interval(values):
    a = np.asarray(values, float)
    if len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Au moins deux unités indépendantes finies requises.")
    half = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return {"n": len(a), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "ci95": [float(a.mean()-half), float(a.mean()+half)],
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def categorical_risk(probabilities, reference, weights):
    """KL non écrêté : renvoie +inf lorsqu'une probabilité vraie positive manque."""
    p, q = np.asarray(reference), np.asarray(probabilities)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0)
    return float(np.asarray(weights) @ terms.sum(axis=1))


def smoothed(counts, alpha=.5):
    counts = np.asarray(counts)
    if counts.shape != (10, 6) or np.any(counts < 0) or alpha <= 0:
        raise ValueError("Effectifs (10,6) non négatifs et alpha positif requis.")
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + 6*alpha)


def dataset(n, seed, card_probabilities=None):
    """Même flux que generate_dataset historique ; comptes et temps par préfixe."""
    rng = np.random.default_rng(seed)
    data = np.empty((n, 2), dtype=np.int64)
    draws = np.empty(n, dtype=np.int64)
    elapsed = np.empty(n)
    count = 0
    start = time.perf_counter()
    def draw():
        return draw_card(rng) if card_probabilities is None else int(rng.choice(np.arange(1, 11), p=card_probabilities))
    for i in range(n):
        hand = [draw(), draw()]
        visible = hand[0]
        while hand_value(hand) < 17:
            hand.append(draw())
        data[i] = visible, min(hand_value(hand), 22)
        count += len(hand)
        draws[i] = count
        elapsed[i] = time.perf_counter()-start
    return data[:, :1], data[:, 1], draws, elapsed


def fit_estimators(X, y, Xv, yv):
    candidates = []
    best = None
    for c in (.01, .1, 1., 10., 100.):
        start = time.perf_counter()
        model = fit_classifier(X, y, c)
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        p = model.predict_proba(np.arange(1, 11)[:, None])
        loss = float(-np.log(p[Xv[:, 0]-1, yv-17]).mean())
        scoring_seconds = time.perf_counter()-start
        candidates.append({"C": c, "validation_log_loss": loss, "fit_seconds": fit_seconds,
                           "scoring_seconds": scoring_seconds})
        if best is None or loss < best[0]:
            best = loss, p, c, fit_seconds
    start = time.perf_counter()
    counts = conditional_counts(X, y)
    totals = counts.sum(axis=1, keepdims=True)
    # Cas non observé explicitement uniforme, sans créer une probabilité nulle supplémentaire.
    freq = np.divide(counts, totals, out=np.full((10, 6), 1/6), where=totals > 0)
    frequency_seconds = time.perf_counter()-start
    start = time.perf_counter()
    jeffreys = smoothed(counts)
    smoothing_seconds = time.perf_counter()-start
    matrices = {"logistic": best[1], "frequencies": freq, "jeffreys": jeffreys}
    tables, table_times = {}, {}
    for method, p in matrices.items():
        start = time.perf_counter()
        tables[method] = build_stand_table(p)
        table_times[method] = time.perf_counter()-start
    # Monte-Carlo direct de R(p,D), sans construire la loi : identité algébrique indépendante.
    start = time.perf_counter()
    direct = np.zeros((18, 10))
    for total in range(4, 22):
        rewards = np.where(y == 22, 1, np.sign(total-y))
        sums = np.bincount(X[:, 0]-1, weights=rewards, minlength=10)
        direct[total-4] = np.divide(sums, totals[:, 0], out=np.zeros(10), where=totals[:, 0] > 0)
    direct_seconds = time.perf_counter()-start
    mask = totals[:, 0] > 0
    max_direct_error = float(np.max(np.abs(direct[:, mask]-tables['frequencies'][4:, 1:, 0][:, mask])))
    return matrices, tables, {"selected_C": best[2], "selected_fit_seconds": best[3],
        "candidates": candidates, "selection_seconds": sum(c['fit_seconds']+c['scoring_seconds'] for c in candidates),
        "counts": counts.tolist(), "frequency_seconds": frequency_seconds,
        "smoothing_seconds": smoothing_seconds, "stand_build_seconds": table_times,
        "direct_stand_seconds": direct_seconds, "direct_stand_max_error": max_direct_error}


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except ImportError:
        return None


def train(seed, stand, checkpoints, evaluate, eps_min=1., omega=1., standard=False, env_factory=make_env):
    """Même ordre RNG et mises à jour que train historique ; évaluation hors temps RL."""
    checkpoints = sorted(set(checkpoints))
    if not checkpoints or checkpoints[0] < 0 or checkpoints[-1] < 1 or not 0 < eps_min <= 1 or not .5 < omega <= 1:
        raise ValueError("Budgets, epsilon ou omega invalides.")
    Q = np.zeros((2, *SHAPE))
    N = np.zeros((2, *SHAPE), dtype=np.int64)
    visits = np.zeros_like(N)
    if not standard:
        Q[STAND] = stand
    rng = np.random.default_rng(seed)
    curve = []
    rl_seconds = eval_seconds = 0.
    cards = decisions = 0
    def record(episode):
        nonlocal eval_seconds
        start = time.perf_counter()
        score = evaluate(Q[HIT] > Q[STAND])
        eval_seconds += time.perf_counter()-start
        curve.append({"episodes": episode, "j": score, "rl_seconds": rl_seconds,
                      "evaluation_seconds": eval_seconds, "cards": cards, "decisions": decisions})
    if 0 in checkpoints:
        record(0)
    targets = set(checkpoints)
    start = time.perf_counter()
    with env_factory() as env:
        for episode in range(checkpoints[-1]):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            epsilon = max(eps_min, .99995 ** episode)
            while True:
                action = choose_action(Q, state, epsilon, rng)
                visits[action][state] += 1
                nxt, reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise RuntimeError("Troncature inattendue ; aucune partie incomplète n'est apprise.")
                if standard or action == HIT:
                    update(Q, N, state, action, reward, nxt, terminated, gamma=1., omega=omega)
                decisions += 1
                state = nxt
                if terminated:
                    break
            cards += len(env.unwrapped.player)+len(env.unwrapped.dealer)
            if episode+1 in targets:
                rl_seconds += time.perf_counter()-start
                record(episode+1)
                start = time.perf_counter()
    if not standard:
        np.testing.assert_array_equal(Q[STAND], stand)
    return {"Q": Q, "N": N, "visits": visits, "curve": curve, "peak_process_rss_bytes": peak_rss_bytes(),
            "table_bytes": Q.nbytes+N.nbytes+visits.nbytes}
