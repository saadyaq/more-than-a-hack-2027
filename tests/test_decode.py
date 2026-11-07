"""Toy possession paths test transition costs independently of match data."""

import itertools

import numpy as np
import pytest

from possession import params as p
from possession.decode import decode, transition_scores, viterbi


def test_short_opposing_blip_is_absorbed():
    evidence = np.zeros((30, 3))
    evidence[:, 0] = p.W_CONTROL
    evidence[15] = [0., p.W_CONTROL, 0.]
    path = viterbi(evidence)
    assert path[0] == p.DEAD
    np.testing.assert_array_equal(path[1:], p.HOME)


def test_sustained_evidence_switches():
    evidence = np.zeros((31, 3))
    evidence[:15, 0] = p.W_CONTROL
    evidence[15:, 1] = p.W_CONTROL
    path = viterbi(evidence)
    np.testing.assert_array_equal(path[1:15], p.HOME)
    np.testing.assert_array_equal(path[15:], p.AWAY)


def test_ball_out_causes_dead_state():
    evidence = np.zeros((31, 3))
    evidence[:15, 0] = p.W_CONTROL
    evidence[15:, 2] = p.W_OUT
    path = viterbi(evidence)
    np.testing.assert_array_equal(path[1:15], p.HOME)
    np.testing.assert_array_equal(path[15:], p.DEAD)


def test_uniform_evidence_stays_dead_and_halves_reset():
    np.testing.assert_array_equal(viterbi(np.zeros((20, 3))), p.DEAD)
    evidence = np.tile([p.W_CONTROL, 0., 0.], (30, 1))
    labels = decode(evidence, blocks=[(1, 2, 10), (2, 18, 26)])
    np.testing.assert_array_equal(labels[[2, 18]], "D")
    np.testing.assert_array_equal(labels[3:11], "H")
    np.testing.assert_array_equal(labels[19:27], "H")
    np.testing.assert_array_equal(labels[np.r_[0:2, 11:18, 27:30]], "X")


def test_ties_prefer_staying_then_home_order():
    # Entering H on the final frame exactly repays the transition cost; D should stay.
    np.testing.assert_array_equal(viterbi(np.array([[0., 0., 0.], [p.P_FROM_DEAD, 0., 0.]])),
                                  [p.DEAD, p.DEAD])
    evidence = np.tile([p.W_CONTROL, p.W_CONTROL, 0.], (20, 1))
    np.testing.assert_array_equal(viterbi(evidence)[1:], p.HOME)


def test_viterbi_score_matches_exhaustive_search():
    evidence = np.random.default_rng(42).normal(size=(7, 3)) * 5
    transitions = transition_scores()

    def score(path):
        return (sum(evidence[t, state] for t, state in enumerate(path))
                + sum(transitions[a, b] for a, b in zip(path[:-1], path[1:])))

    optimum = max(score((p.DEAD,) + rest) for rest in itertools.product(range(3), repeat=6))
    assert score(viterbi(evidence)) == pytest.approx(optimum)


def test_empty_and_invalid_evidence():
    assert viterbi(np.empty((0, 3))).size == 0
    assert decode(np.empty((0, 3)), blocks=[]).size == 0
    with pytest.raises(ValueError):
        viterbi(np.array([[0., np.nan, 0.]]))
