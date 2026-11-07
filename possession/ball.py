"""Clean brief ball occlusions without turning long absences into invented control."""

import numpy as np

from possession import params as p


def interpolate_short_gaps(ball_xy, max_gap=p.MAX_GAP_FRAMES, *, pitch=p.PITCH, fps=p.FPS,
                           max_speed_mps=p.MAX_INTERPOLATION_SPEED_MPS):
    """Fill bounded gaps only when endpoint motion is physically plausible in metres."""
    filled = np.array(ball_xy, dtype=np.float64, copy=True)
    observed = np.isfinite(filled).all(axis=1)
    filled[~observed] = np.nan
    interpolated = np.zeros(len(filled), dtype=bool)
    present = np.flatnonzero(observed)
    if len(present) < 2:
        return filled, interpolated
    left, right = present[:-1], present[1:]
    gaps = right - left - 1
    speed = np.linalg.norm((filled[right] - filled[left]) * pitch, axis=1) * fps / (gaps + 1)
    eligible = (gaps > 0) & (gaps <= max_gap) & (speed <= max_speed_mps)
    # Each missing frame finds its bracketing observations; no frame-by-frame Python loop.
    index = np.arange(len(filled))
    interval = np.searchsorted(present, index, side="right") - 1
    bounded = (~observed) & (interval >= 0) & (interval < len(left))
    candidates = np.flatnonzero(bounded)
    candidates = candidates[eligible[interval[candidates]]]
    brackets = interval[candidates]
    weight = (candidates - left[brackets]) / (right[brackets] - left[brackets])
    filled[candidates] = (filled[left[brackets]] * (1 - weight[:, None])
                          + filled[right[brackets]] * weight[:, None])
    interpolated[candidates] = True
    return filled, interpolated


def ball_speed_mps(ball_xy, pitch, *, fps=p.FPS, median_frames=p.SPEED_MEDIAN_FRAMES):
    """Centred differences need the centre and both neighbours; filtering preserves gaps."""
    xy = np.asarray(ball_xy, dtype=np.float64)
    speed = np.full(len(xy), np.nan)
    observed = np.isfinite(xy).all(axis=1)
    if len(xy) >= 3:
        available = observed[:-2] & observed[1:-1] & observed[2:]
        centred = np.linalg.norm((xy[2:] - xy[:-2]) * pitch, axis=1) * fps / 2
        speed[1:-1] = np.where(available, centred, np.nan)
    if not len(speed):
        return speed
    if median_frames < 1 or median_frames % 2 != 1:
        raise ValueError("median_frames must be a positive odd integer")
    radius = median_frames // 2
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(speed, (radius, radius), constant_values=np.nan), median_frames)
    filtered = np.full(len(speed), np.nan)
    valid = np.isfinite(speed)
    # Select valid centres first: no all-NaN median warnings and no synthetic availability.
    filtered[valid] = np.nanmedian(windows[valid], axis=1)
    return filtered


def out_of_pitch(ball_xy, margin_m=p.OUT_MARGIN_M, *, pitch=p.PITCH):
    """Missing positions are not out; both axes use their own metre-to-coordinate margin."""
    xy = np.asarray(ball_xy, dtype=np.float64)
    margin = margin_m / np.asarray(pitch)
    return np.isfinite(xy).all(axis=1) & ((xy < -margin) | (xy > 1 + margin)).any(axis=1)
