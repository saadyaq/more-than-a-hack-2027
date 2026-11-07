"""Decode possession independently in each graded half, with a dead initial state."""

import numpy as np

from games import GAME, half_video_blocks
from possession import params as p


def transition_scores(penalties=None):
    """Rows are the previous state, columns the next; entering and leaving dead differ."""
    if penalties is None:
        penalties = {"P_SWITCH": p.P_SWITCH, "P_TO_DEAD": p.P_TO_DEAD,
                     "P_FROM_DEAD": p.P_FROM_DEAD}
    switch, to_dead, from_dead = (penalties[key] for key in
                                  ("P_SWITCH", "P_TO_DEAD", "P_FROM_DEAD"))
    return np.array([[0., -switch, -to_dead],
                     [-switch, 0., -to_dead],
                     [-from_dead, -from_dead, 0.]])


def viterbi(evidence, penalties=None):
    """Prefer a self predecessor on ties, then H/A/D; first frame is fixed to D.

    At a tied terminal score, prefer a state whose selected last step stayed, then
    H/A/D. The zero common emission baseline is uniform and does not affect paths.
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 3 or not np.isfinite(evidence).all():
        raise ValueError("evidence must be a finite (n_frames, 3) array")
    if not len(evidence):
        return np.empty(0, dtype=np.int8)
    transition = transition_scores(penalties)
    states = np.arange(3)
    previous = np.empty((len(evidence), 3), dtype=np.int8)
    previous[0] = states
    scores = np.array([-np.inf, -np.inf, 0.]) + evidence[0] + p.EMISSION_BASELINE
    for frame in range(1, len(evidence)):
        candidates = scores[:, None] + transition
        predecessor = np.argmax(candidates, axis=0)
        best = candidates[predecessor, states]
        previous[frame] = np.where(candidates[states, states] == best, states, predecessor)
        scores = best + evidence[frame] + p.EMISSION_BASELINE
    tied = scores == scores.max()
    staying = tied & (previous[-1] == states)
    state = int(np.flatnonzero(staying if staying.any() else tied)[0])
    path = np.empty(len(evidence), dtype=np.int8)
    path[-1] = state
    for frame in range(len(evidence) - 1, 0, -1):
        path[frame - 1] = previous[frame, path[frame]]
    return path


def decode(evidence, blocks=None, penalties=None):
    """Graded block ends are inclusive; every other video frame is X."""
    if blocks is None:
        blocks = half_video_blocks(GAME)
    labels = np.full(len(evidence), "X", dtype="<U1")
    alphabet = np.array(["H", "A", "D"])
    for _, start, end in blocks:
        start, stop = max(0, start), min(len(labels), end + 1)
        if start < stop:
            labels[start:stop] = alphabet[viterbi(evidence[start:stop], penalties=penalties)]
    return labels
