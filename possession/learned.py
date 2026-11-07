"""Learned emission model: a gradient-boosting classifier gives P(H, A, D | frame features).

The classifier replaces the hand-written ball-control, stillness and shape scores with what the
external Metrica games teach. The Viterbi decoder and the Smart Tagging terms are unchanged:
emissions are T * (log P(state | x) - log prior(state)), so the classifier's view of a frame becomes
a log-likelihood ratio, and tag, restart and kick-off evidence is added on top with its usual
fixed weights.

Symmetry: the model is trained on every game twice, once as recorded and once with the teams
swapped (labels H <-> A), and at prediction time the two views are averaged. P(H) for a frame is
therefore exactly P(A) for the swapped frame, so the model cannot carry a home or away bias learnt
on other games.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import params as p
from possession.data import Tracking
from possession.features import FEATURE_NAMES, frame_features

# Params
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "learned_model.joblib"  # v7
CONFIG_PATH = HERE / "learned_v7.json"

SUBSAMPLE = 5  # Neighbouring frames are near-duplicates; every fifth keeps training fast.
GBM = dict(  # Moderate capacity: ~500k rows, 31 features, three classes.
    max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
    l2_regularization=1.0, early_stopping=False, random_state=0,
)
LABEL_CODES = {"H": p.HOME, "A": p.AWAY, "D": p.DEAD}


def paths(version="v7"):
    """(model, config) files of a learned version; v7 keeps its original file names."""
    if version == "v7":
        return MODEL_PATH, CONFIG_PATH
    return HERE / "learned_model_{}.joblib".format(version), HERE / "learned_{}.json".format(version)


def swap_teams(tracking):
    """The same tracking with home and away exchanged."""
    return Tracking(tracking.players, tracking.ball, (1 - tracking.team_of_slot).astype(np.int8),
                    tracking.pitch, tracking.players_xy64, tracking.ball_xy64)


def training_rows(tracking, blocks, truth, *, fps=p.FPS):
    """(X, y) for both team orientations, every SUBSAMPLE-th graded frame."""
    keep = np.isin(truth, list(LABEL_CODES))
    keep[np.arange(len(keep)) % SUBSAMPLE != 0] = False
    y = np.array([LABEL_CODES[label] for label in truth[keep]])
    swapped_y = np.where(y == p.HOME, p.AWAY, np.where(y == p.AWAY, p.HOME, y))
    X = frame_features(tracking, blocks, fps=fps)[keep]
    Xs = frame_features(swap_teams(tracking), blocks, fps=fps)[keep]
    return np.vstack((X, Xs)), np.concatenate((y, swapped_y))


def fit(X, y):
    model = HistGradientBoostingClassifier(**GBM)
    model.fit(X, y)
    if list(model.classes_) != [p.HOME, p.AWAY, p.DEAD]:
        raise ValueError("training data must contain H, A and D")
    return model


def predict_proba(model, tracking, blocks, *, fps=p.FPS):
    """Symmetrised (n_frames, 3) probabilities in H, A, D order; NaN outside blocks."""
    direct = model.predict_proba(frame_features(tracking, blocks, fps=fps))
    swapped = model.predict_proba(frame_features(swap_teams(tracking), blocks, fps=fps))
    probs = (direct + swapped[:, [p.AWAY, p.HOME, p.DEAD]]) / 2.
    inside = np.zeros(len(probs), dtype=bool)
    for _, start, end in blocks:
        inside[max(0, start):min(len(probs), end + 1)] = True
    probs[~inside] = np.nan
    return probs


def emissions(probs, prior, temperature):
    """T * log-likelihood ratio per state; 0 where there is no prediction."""
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = temperature * (np.log(np.clip(probs, 1e-6, 1.)) - np.log(prior))
    scores[~np.isfinite(scores)] = 0.
    return scores


def save(model, config, version="v7"):
    model_path, config_path = paths(version)
    joblib.dump(model, model_path, compress=3)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def load(version="v7"):
    model_path, config_path = paths(version)
    config = json.loads(config_path.read_text())
    if tuple(config["features"]) != FEATURE_NAMES:
        raise ValueError("learned model was trained on different features")
    return joblib.load(model_path), config
