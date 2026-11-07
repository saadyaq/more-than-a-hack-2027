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
    path = viterbi(evidence, penalties=p.V1_TRANSITION_PENALTIES)
    np.testing.assert_array_equal(path[1:15], p.HOME)
    np.testing.assert_array_equal(path[15:], p.AWAY)


@pytest.mark.parametrize("penalties", [None, p.V1_TRANSITION_PENALTIES])
def test_ball_out_causes_dead_state(penalties):
    boundary = 3 * p.FPS  # Establish sustained control before testing the stoppage.
    evidence = np.zeros((2 * boundary, 3))
    evidence[:boundary, 0] = p.W_CONTROL
    evidence[boundary:, 2] = p.W_OUT
    path = viterbi(evidence, penalties=penalties)
    np.testing.assert_array_equal(path[1:boundary], p.HOME)
    np.testing.assert_array_equal(path[boundary:], p.DEAD)


def test_uniform_evidence_stays_dead_and_halves_reset():
    np.testing.assert_array_equal(viterbi(np.zeros((20, 3))), p.DEAD)
    evidence = np.tile([p.W_CONTROL, 0., 0.], (30, 1))
    # v2's cheap restart keeps these blocks short; this test is about the per-half reset.
    labels = decode(evidence, blocks=[(1, 2, 10), (2, 18, 26)],
                    penalties=p.V2_TRANSITION_PENALTIES)
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


def test_scaled_penalties_and_frozen_v1():
    assert p.P_SWITCH == p.W_CONTROL * p.FPS * p.MIN_CONTROL_S == 50.
    assert p.P_TO_DEAD == p.W_STILL * p.FPS * p.MIN_STOPPAGE_S == 25.
    assert p.P_FROM_DEAD == p.P_SWITCH - p.P_TO_DEAD == 25.
    np.testing.assert_array_equal(transition_scores(p.V1_TRANSITION_PENALTIES),
                                  [[0., -8., -6.], [-8., 0., -6.], [-4., -4., 0.]])
    np.testing.assert_array_equal(transition_scores(p.V2_TRANSITION_PENALTIES),
                                  [[0., -50., -25.], [-50., 0., -25.], [-4., -4., 0.]])
    np.testing.assert_array_equal(transition_scores(),
                                  [[0., -50., -25.], [-50., 0., -25.], [-25., -25., 0.]])


def test_fake_stoppage_constraint():
    with pytest.raises(ValueError):
        p.check_no_cheap_fake_stoppage(p.V2_TRANSITION_PENALTIES)
    p.check_no_cheap_fake_stoppage(
        {"P_SWITCH": p.P_SWITCH, "P_TO_DEAD": p.P_TO_DEAD, "P_FROM_DEAD": p.P_FROM_DEAD})


def test_registered_methods_enforce_constraint_and_only_historical_exemptions(monkeypatch):
    from possession.run import METHODS, validate_method_penalties

    validate_method_penalties(METHODS)
    # A new name cannot inherit v2's exemption even with identical parameters.
    monkeypatch.setitem(p.METHOD_TRANSITION_PENALTIES, "new_method", p.V2_TRANSITION_PENALTIES)
    with pytest.raises(ValueError):
        validate_method_penalties({**METHODS, "new_method": None})
    with pytest.raises(AssertionError):
        validate_method_penalties({**METHODS, "undeclared_method": None})
    # The actual registered v3 configuration must also be checked, not just the defaults.
    monkeypatch.setitem(p.METHOD_TRANSITION_PENALTIES, "v3", p.V2_TRANSITION_PENALTIES)
    with pytest.raises(ValueError):
        validate_method_penalties(METHODS)


def test_v3_turnover_without_dead_evidence_is_direct():
    context = 3 * p.FPS
    evidence = np.zeros((2 * context, 3))
    evidence[:context, p.HOME] = p.W_CONTROL
    evidence[context:, p.AWAY] = p.W_CONTROL
    path = viterbi(evidence)
    assert not np.any(path[1:] == p.DEAD)
    assert path[context - 1] == p.HOME and path[-1] == p.AWAY
    # v2 hid the same turnover behind a phantom stoppage, or switched no cheaper than v3.
    v2_path = viterbi(evidence, penalties=p.V2_TRANSITION_PENALTIES)
    assert v2_path[-1] == p.AWAY


def test_v3_real_stoppage_still_dead():
    context = 3 * p.FPS
    stoppage = 4 * p.FPS
    evidence = np.zeros((2 * context + stoppage, 3))
    evidence[:context, p.HOME] = p.W_CONTROL
    evidence[context:context + stoppage, p.DEAD] = p.W_OUT + p.W_STILL
    evidence[context + stoppage:, p.AWAY] = p.W_CONTROL
    path = viterbi(evidence)
    assert np.all(path[context + p.FPS:context + stoppage - p.FPS] == p.DEAD)
    assert path[-1] == p.AWAY


@pytest.mark.parametrize("seconds,should_switch", [(.5, False), (2., True)])
def test_v2_control_duration(seconds, should_switch):
    context = 3 * p.FPS
    # Round up the half-second case: at 25 fps it lies between 12 and 13 frames.
    duration = int(np.ceil(seconds * p.FPS))
    evidence = np.zeros((2 * context + duration, 3))
    evidence[:, p.HOME] = p.W_CONTROL
    evidence[context:context + duration] = [0., p.W_CONTROL, 0.]
    path = viterbi(evidence, penalties=p.V2_TRANSITION_PENALTIES)
    if should_switch:
        np.testing.assert_array_equal(path[context + 1:context + duration - 1], p.AWAY)
        assert path[-1] == p.HOME
    else:
        np.testing.assert_array_equal(path[1:], p.HOME)
    # The same half-second control previously caused a v1 turnover.
    if not should_switch:
        original = viterbi(evidence, penalties=p.V1_TRANSITION_PENALTIES)
        np.testing.assert_array_equal(original[context:context + duration], p.AWAY)


def test_penalty_override_does_not_mutate_defaults():
    expected = transition_scores().copy()
    viterbi(np.zeros((5, 3)), penalties=p.V1_TRANSITION_PENALTIES)
    np.testing.assert_array_equal(transition_scores(), expected)
