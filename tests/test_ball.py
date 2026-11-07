"""Ball cleanup limits use synthetic motion in a known metric pitch."""

import numpy as np
import pytest

from possession.ball import ball_speed_mps, interpolate_short_gaps, out_of_pitch


@pytest.mark.parametrize("gap,filled", [(1, True), (12, True), (13, False)])
def test_interpolation_gap_limit(gap, filled):
    xy = np.full((gap + 2, 2), np.nan)
    xy[0], xy[-1] = [0.4, 0.5], [0.41, 0.5]
    result, mask = interpolate_short_gaps(xy, pitch=(100., 50.))
    assert mask.sum() == (gap if filled else 0)
    if filled:
        np.testing.assert_allclose(result[:, 0], np.linspace(0.4, 0.41, gap + 2))
    else:
        assert np.isnan(result[1:-1]).all()
    assert np.isnan(xy[1:-1]).all()  # Input observations are never overwritten.


@pytest.mark.parametrize("speed,filled", [(40., True), (40.1, False)])
def test_interpolation_speed_limit(speed, filled):
    xy = np.array([[0., 0.], [np.nan, np.nan], [speed * 2 / 25 / 100, 0.]])
    _, mask = interpolate_short_gaps(xy, pitch=(100., 50.))
    assert bool(mask[1]) == filled


def test_interpolation_leading_trailing_and_partial_gaps():
    xy = np.array([[np.nan, np.nan], [0.4, 0.5], [np.nan, 0.5],
                   [0.41, 0.5], [np.nan, np.nan]])
    result, mask = interpolate_short_gaps(xy, pitch=(100., 50.))
    np.testing.assert_array_equal(mask, [False, False, True, False, False])
    assert np.isnan(result[[0, 4]]).all()
    np.testing.assert_allclose(result[2], [0.405, 0.5])


@pytest.mark.parametrize("length", [0, 1, 5])
def test_cleanup_without_observations(length):
    xy = np.full((length, 2), np.nan)
    result, mask = interpolate_short_gaps(xy)
    assert np.isnan(result).all()
    assert not mask.any()
    assert np.isnan(ball_speed_mps(xy, (100., 50.))).all()
    assert not out_of_pitch(xy).any()


def test_centred_speed_and_median_filter():
    xy = np.column_stack([np.arange(21) * .001, np.zeros(21)])
    xy[10, 0] += .4  # One position error contaminates two finite differences.
    speed = ball_speed_mps(xy, (100., 50.))
    np.testing.assert_allclose(speed[1:-1], 2.5)
    assert np.isnan(speed[[0, -1]]).all()
    xy[10] = np.nan
    speed = ball_speed_mps(xy, (100., 50.))
    assert np.isnan(speed[9:12]).all()


def test_out_of_pitch_metric_margin_and_missing():
    xy = np.array([[-.005, .5], [-.0051, .5], [1.0051, .5],
                   [.5, -.01], [.5, -.0101], [.5, 1.0101], [.5, .5], [np.nan, 2.]])
    np.testing.assert_array_equal(out_of_pitch(xy, pitch=(100., 50.)),
                                  [False, True, True, False, True, True, False, False])
