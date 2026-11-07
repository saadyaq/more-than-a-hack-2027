"""The learned model must treat the two teams symmetrically and decode deterministically."""

import numpy as np
import pytest

from possession import learned, params as p
from possession.data import Tracking


def synthetic(frames=300, seed=0):
    rng = np.random.default_rng(seed)
    players = np.full((frames, 22, 3), np.nan, dtype=np.float32)
    players[..., :2] = rng.uniform(0.1, 0.9, (frames, 22, 2))
    players[..., 2] = rng.uniform(0, 5, (frames, 22))
    ball = np.full((frames, 3), np.nan, dtype=np.float32)
    ball[: frames // 2, :2] = players[: frames // 2, 3, :2] + 0.005
    ball[: frames // 2, 2] = 2.
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return Tracking(players, ball, team, (105.0, 68.0))


@pytest.fixture(scope="module")
def model_and_config():
    if not learned.MODEL_PATH.exists():
        pytest.skip("learned model not trained")
    return learned.load()


def test_predictions_are_symmetric(model_and_config):
    model, _ = model_and_config
    tracking, blocks = synthetic(), [(1, 0, 299)]
    probs = learned.predict_proba(model, tracking, blocks)
    swapped = learned.predict_proba(model, learned.swap_teams(tracking), blocks)
    np.testing.assert_allclose(probs, swapped[:, [p.AWAY, p.HOME, p.DEAD]], atol=1e-12)
    np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-9)


def test_emissions_zero_without_prediction():
    probs = np.full((4, 3), np.nan)
    probs[1] = [0.5, 0.3, 0.2]
    scores = learned.emissions(probs, np.array([1 / 3] * 3), 0.4)
    assert np.all(scores[[0, 2, 3]] == 0)
    assert scores[1, p.HOME] > 0 > scores[1, p.DEAD]


def test_config_respects_fake_stoppage_constraint(model_and_config):
    _, config = model_and_config
    p.check_no_cheap_fake_stoppage(config["penalties"])
