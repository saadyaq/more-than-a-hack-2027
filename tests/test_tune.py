"""The tuner must never propose a decoder that breaks the fake-stoppage constraint."""

import itertools

from possession import params as p
from possession.tune import TRANSITION_GRID, penalties_of, start_point


def test_start_point_is_v3():
    assert penalties_of(start_point()) == p.METHOD_TRANSITION_PENALTIES["v3"]


def test_every_grid_point_respects_fake_stoppage_constraint():
    for switch, to_dead, extra in itertools.product(*TRANSITION_GRID.values()):
        penalties = penalties_of(dict(P_SWITCH=switch, P_TO_DEAD=to_dead,
                                      P_FROM_DEAD_EXTRA=extra))
        assert penalties["P_TO_DEAD"] + penalties["P_FROM_DEAD"] >= penalties["P_SWITCH"]
        assert penalties["P_FROM_DEAD"] >= 0
