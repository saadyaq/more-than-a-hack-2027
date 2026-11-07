"""Synthetic regression cases for distance precision, missing observations, and voting."""

import numpy as np
import pytest

from possession.baseline import majority_smooth, nearest_team
from possession.data import Tracking
from possession.io import write_submission
from possession.run import possession_count
from starter import baseline as official


def tracking(players, ball, teams=(0, 1), pitch=(105.3, 68.0)):
    return Tracking(np.asarray(players, dtype=np.float32), np.asarray(ball, dtype=np.float32),
                    np.asarray(teams, dtype=np.int8), pitch)


def test_nearest_metric_distance_and_slot_ties():
    observations = tracking(
        [[[0.1, 0, 0], [0, 0.11, 0]], [[0.25, 0, 0], [0.75, 0, 0]]],
        [[0, 0, 0], [0.5, 0, 0]], teams=(1, 0), pitch=(100., 64.))
    # Pitch scaling reverses the first normalised-distance choice; first slot wins the tie.
    codes = nearest_team(observations)
    np.testing.assert_array_equal(codes, [0, 1])
    assert codes.dtype == np.int8


def test_nearest_missing_players_ball_and_speed():
    nan = np.nan
    observations = tracking(
        [[[nan, nan, nan], [0.2, 0.3, nan]],
         [[nan, nan, nan], [nan, nan, nan]],
         [[0.2, 0.3, 1], [0.5, 0.5, 1]],
         [[0.2, nan, 1], [0.5, 0.5, 1]]],
        [[0.2, 0.3, nan], [0.2, 0.3, 1], [nan, nan, nan], [0.5, 0.5, 1]])
    np.testing.assert_array_equal(nearest_team(observations), [1, -1, -1, 1])


def test_nearest_empty_frames_and_slots():
    observations = Tracking(np.empty((0, 2, 3), np.float32), np.empty((0, 3), np.float32),
                            np.array([0, 1], np.int8), (105.3, 68.0))
    assert nearest_team(observations).shape == (0,)
    observations = Tracking(np.empty((2, 0, 3), np.float32), np.zeros((2, 3), np.float32),
                            np.array([], np.int8), (105.3, 68.0))
    np.testing.assert_array_equal(nearest_team(observations), [-1, -1])


def test_float64_path_preserves_nearest_decision():
    observations = tracking([[[0.50000002, 0, 0], [0.50000001, 0, 0]]], [[0, 0, 0]])
    np.testing.assert_array_equal(nearest_team(observations), [0])  # float32 collapses the gap.
    observations.players_xy64 = np.array([[[0.50000002, 0], [0.50000001, 0]]])
    observations.ball_xy64 = np.array([[0., 0.]])
    np.testing.assert_array_equal(nearest_team(observations), [1])


@pytest.mark.parametrize("codes,window,expected", [
    ([0, 1, -1], 0, "HAD"), ([0, 1, -1], -1, "HAD"),
    ([0, -1, 1], 1, "HHA"), ([0, -1, 1], 20, "HHH"),
    ([-1, -1, -1], 2, "DDD"), ([], 50, ""), ([1], 50, "A"),
])
def test_majority_smooth(codes, window, expected):
    labels = majority_smooth(np.array(codes, np.int8), window)
    np.testing.assert_array_equal(labels, np.array(list(expected), dtype="<U1"))


@pytest.mark.parametrize("window", [-1, 0, 1, 25, 50, 125, 1000])
def test_smoothing_matches_official(window):
    codes = np.random.default_rng(12).choice(np.array([-1, 0, 1], np.int8), size=257)
    np.testing.assert_array_equal(majority_smooth(codes, window), official.smooth(codes, window))


def test_submission_matches_official(tmp_path):
    labels = np.array(list("HHADXA"))
    ours = write_submission(tmp_path / "nested" / "ours.csv", labels)
    reference = official.write_submission(tmp_path / "reference.csv", labels)
    assert ours.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("labels,count", [("", 0), ("HHADAAHXXH", 5), ("DDXX", 0)])
def test_possession_count(labels, count):
    assert possession_count(np.array(list(labels), dtype="<U1")) == count
