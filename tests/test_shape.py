"""Team-shape features must be antisymmetric so the model cannot favour either side."""

import numpy as np

from possession.data import Tracking
from possession.shape import FEATURES, fit_logistic, shape_features


def synthetic(seed=0, frames=200):
    rng = np.random.default_rng(seed)
    players = np.full((frames, 22, 3), np.nan, dtype=np.float32)
    players[..., :2] = rng.uniform(0.1, 0.9, (frames, 22, 2))
    players[:, :11, 1] = np.clip(players[:, :11, 1] * 1.3 - 0.15, 0, 1)  # home wider
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    ball = np.full((frames, 3), np.nan, dtype=np.float32)
    return Tracking(players, ball, team, (105.0, 68.0))


def test_swapping_teams_negates_every_feature():
    tracking = synthetic()
    swapped = Tracking(tracking.players, tracking.ball, 1 - tracking.team_of_slot, tracking.pitch)
    blocks = [(1, 0, 199)]
    a, b = shape_features(tracking, blocks), shape_features(swapped, blocks)
    finite = np.isfinite(a)
    assert finite.any()
    np.testing.assert_allclose(a[finite], -b[finite], atol=1e-9)
    assert a.shape[1] == len(FEATURES)


def test_too_few_players_gives_nan():
    tracking = synthetic()
    tracking.players[:, 5:11] = np.nan  # five home players left
    # A row with any missing feature is ignored by the model, so no row may be complete.
    assert not np.isfinite(shape_features(tracking, [(1, 0, 199)])).all(axis=1).any()


def test_logistic_learns_sign_without_intercept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 1))
    y = (X[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(float)
    w, _ = fit_logistic(X, y)
    assert w[0] > 1
