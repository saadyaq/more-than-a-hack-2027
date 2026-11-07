"""Synthetic evidence checks exercise team semantics and timing, never the public GT."""

import numpy as np
import pandas as pd
import pytest

from possession import params as p
from possession.data import Tracking
from possession.evidence import (build_evidence, control_evidence, restart_evidence,
                                 stillness_evidence, tag_evidence)


def observations(n=10, player_speed=2.):
    players = np.tile([[[.49, .5, player_speed], [.6, .5, player_speed]]], (n, 1, 1))
    ball = np.tile([.5, .5, 0.], (n, 1))
    return Tracking(players, ball, np.array([0, 1], np.int8), (100., 50.))


def event(code, team="Barcelona", start=2, end=6, **tags):
    return dict(code=code, Team=team, start_frame=start, end_frame=end, **tags)


@pytest.mark.parametrize("code,team,state,weight", [
    (" BUILD UP ", "bArCeLoNa", 0, p.W_TAG_ATTACK),
    ("Progression", "ATLETIC CLUB", 1, p.W_TAG_ATTACK),
    ("DEFENDING IN MIDDLE THIRD", "Barcelona", 1, p.W_TAG_DEFEND),
    ("defensive transition", "Atletic Club", 0, p.W_TAG_DEFEND),
])
def test_attack_and_defend_conventions(code, team, state, weight):
    evidence = tag_evidence(pd.DataFrame([event(code, team)]), 8)
    expected = np.zeros((8, 3))
    expected[2:6, state] = weight
    np.testing.assert_array_equal(evidence, expected)


def test_tag_overlaps_cap_per_state_unknown_teams_and_clipping():
    events = pd.DataFrame([event("BUILD UP", start=-3), event("PROGRESSION"),
                           event("LONG BALL"), event("BUILD UP", "Atletic Club"),
                           event("BUILD UP", "N/A"), event("UNKNOWN")])
    evidence = tag_evidence(events, 8)
    np.testing.assert_allclose(evidence[2:6, 0], p.TAG_CAP)
    np.testing.assert_allclose(evidence[2:6, 1], p.W_TAG_ATTACK)
    np.testing.assert_array_equal(evidence[6:], 0.)
    assert not evidence[:, 2].any()


def test_control_contested_and_interpolated_weights():
    tracking = observations(4)
    tracking.players[:, 0, 0] = .495  # Half a metre from the ball.
    tracking.players[:, 1, 0] = [.505, .51, .515, .52]
    evidence = control_evidence(tracking, tracking.ball[:, :2], np.zeros(4),
                                 np.array([False, False, False, True]))
    np.testing.assert_allclose(evidence[:, 0], [0., p.W_CONTROL / 2, p.W_CONTROL,
                                               p.W_CONTROL * p.INTERPOLATED_CONTROL_WEIGHT], atol=1e-12)
    assert not evidence[:, 1:].any()


def test_fast_far_and_missing_ball_give_no_control():
    tracking = observations(4)
    tracking.players[1, :, 0] = [0., 1.]
    tracking.ball[2, :2] = np.nan
    evidence = control_evidence(tracking, tracking.ball[:, :2], np.array([8., 0., 0., np.nan]),
                                 np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(evidence, 0.)


def test_missing_player_and_absent_opponent():
    tracking = observations(2)
    tracking.players[0, 1] = np.nan
    tracking.players[1, 0] = np.nan
    tracking.players[1, 1, 0] = .505
    evidence = control_evidence(tracking, tracking.ball[:, :2], np.zeros(2),
                                 np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(evidence, [[p.W_CONTROL, 0., 0.], [0., p.W_CONTROL, 0.]])


def test_stillness_and_absent_ball_rolling_window():
    tracking = observations(9, player_speed=.5)
    seen = np.zeros(9, dtype=bool)
    seen[4] = True
    evidence = stillness_evidence(tracking, seen, fps=1)
    np.testing.assert_array_equal(evidence[:, 2], [1.5, 1.5, 1., 1., 1., 1., 1., 1.5, 1.5])
    tracking.players[:] = np.nan
    np.testing.assert_array_equal(stillness_evidence(tracking, seen, fps=1), 0.)


def test_stillness_uses_median_then_rolling_mean():
    tracking = observations(5, player_speed=0.)
    tracking.players[2, :, 2] = 10.
    evidence = stillness_evidence(tracking, np.ones(5, dtype=bool), fps=1)
    np.testing.assert_array_equal(evidence, 0.)  # Window means exceed 1.2 despite still centres.


def test_restart_lead_alive_and_unknown_team():
    events = pd.DataFrame([event("set pieces", "BARCELONA", start=5, end=9),
                           event("SET PIECES", "N/A", start=12, end=14)])
    evidence = restart_evidence(events, 16, [(1, 0, 15)], fps=1)
    expected = np.zeros((16, 3))
    expected[2:5, 2] = p.W_RESTART_DEAD
    expected[5:7, 0] = p.W_RESTART_ALIVE
    expected[9:12, 2] = p.W_RESTART_DEAD
    np.testing.assert_array_equal(evidence, expected)


def test_half_kickoff_and_after_goal_are_different():
    events = pd.DataFrame([
        event("SET PIECES", start=8, end=10, Type="Kick Off (Start)", Half="1st Half"),
        event("SET PIECES", start=20, end=22, Type="Kick Off (After Goal)", Half="1st Half"),
        event("SET PIECES", "Atletic Club", start=35, end=36,
              Type="kick off (start)", Half="2ND HALF"),
    ])
    evidence = restart_evidence(events, 40, [(1, 2, 25), (2, 30, 39)], fps=1)
    np.testing.assert_array_equal(evidence[2:5, 2], p.W_PREKICKOFF)
    np.testing.assert_array_equal(evidence[5:8, 2], p.W_PREKICKOFF + p.W_RESTART_DEAD)
    np.testing.assert_array_equal(evidence[10:17, 2], 0.)
    np.testing.assert_array_equal(evidence[17:20, 2], p.W_RESTART_DEAD)
    np.testing.assert_array_equal(evidence[30:32, 2], p.W_PREKICKOFF)
    np.testing.assert_array_equal(evidence[35:37, 1], p.W_RESTART_ALIVE)


def test_out_evidence_requires_observed_ball_and_outside_halves_zero():
    tracking = observations(7)
    tracking.ball[:, 0] = -.01
    tracking.ball[3, :2] = np.nan  # It will interpolate, but cannot supply W_OUT.
    tracking.players[..., 0] = .5
    evidence = build_evidence(tracking, pd.DataFrame(), blocks=[(1, 1, 5)])
    np.testing.assert_array_equal(evidence[:, 2], [0., p.W_OUT, p.W_OUT, 0., p.W_OUT, p.W_OUT, 0.])
    assert not evidence[:, :2].any()
