"""Known synthetic possessions establish aggregate denominators and overlap handling."""

import numpy as np
import pandas as pd
import pytest

from possession.diagnostics import summarise


def test_aggregate_diagnostics():
    labels = np.array(list("XHHADDHHAX"))
    events = pd.DataFrame([
        dict(code="BUILD UP", Team="Barcelona", start_frame=1, end_frame=9),
        dict(code="PROGRESSION", Team="Barcelona", start_frame=1, end_frame=4),
    ])
    report, = summarise(labels, events, blocks=[(1, 1, 8)], fps=2)
    assert report["possessions"] == 4
    assert report["mean_s"] == report["median_s"] == .75
    assert report["dead_share"] == .25
    assert report["home_share"] == pytest.approx(4 / 6)
    assert report["direct_switches"] == 2
    assert report["tagged_frames"] == 8
    assert report["tag_agreement"] == .5
    assert report["short_dead_runs"] == 0  # Exactly one second is not shorter than one second.


def test_no_alive_or_tagged_frames():
    report, = summarise(np.array(list("DDD")), pd.DataFrame(), blocks=[(1, 0, 2)])
    assert report["possessions"] == 0
    assert report["dead_share"] == 1.
    assert np.isnan(report["home_share"])
    assert np.isnan(report["tag_agreement"])


@pytest.mark.parametrize("length,expected", [(1, 1), (24, 1), (25, 0), (26, 0)])
def test_short_dead_run_strict_one_second_boundary(length, expected):
    labels = np.array(list("H" + "D" * length + "A"))
    report, = summarise(labels, pd.DataFrame(), blocks=[(1, 0, len(labels) - 1)], fps=25)
    assert report["short_dead_runs"] == expected


def test_short_dead_runs_counted_separately_per_half():
    labels = np.array(list("XDHDDADXDDHADDX"))
    reports = summarise(labels, pd.DataFrame(), blocks=[(1, 1, 6), (2, 8, 13)], fps=2)
    # Half 1 has two one-frame D runs and one exactly-one-second run; half 2 has two long runs.
    assert [report["short_dead_runs"] for report in reports] == [2, 0]
