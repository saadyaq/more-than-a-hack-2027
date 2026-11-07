"""Per-frame features for the learned emission model, computable on any tracking source.

Everything here uses only positions, speeds and team membership, so the same features exist on
the external Metrica games (where the model is trained) and on the competition match (where it is
applied). Smart Tagging is deliberately absent: the external games have none, so tag evidence is
added separately with fixed weights.

The labelling is offline, so features may look into the future as well as the past. During a
ball blackout, "which team was closest when the ball was last seen" and "which team is closest
when it reappears" are the strongest clues there are.

Missing values are NaN; the gradient-boosting model handles them natively.
"""

import numpy as np

from possession import params as p
from possession.ball import ball_speed_mps, interpolate_short_gaps, out_of_pitch
from possession.evidence import nearest_team_distances, rolling_sum
from possession.shape import shape_features

# Params
GAP_CLIP_S = 60.0  # Beyond a minute, time to the ball's last or next sighting carries no extra signal.
INTERPOLATE_FRAMES = 100  # Bridge up to 4 s of missing ball for the interpolated-distance feature.
NEAR_RADIUS_M = 3.0  # Players this close to the ball can play it within a stride.
CONTROL_RADIUS_M = 2.0  # A player this close to a slow ball is plausibly controlling it.
CONTROL_SPEED_MPS = 12.0  # Faster balls are passes or shots, not controlled.
CONTROL_WINDOWS_S = (1.0, 3.0, 8.0)  # Control balance at touch, phase and possession scale.
STILL_WINDOWS_S = (1.0, 3.0, 6.0)  # Stillness at restart and stoppage scales.

FEATURE_NAMES = (
    "ball_seen", "d_home", "d_away", "d_diff", "ball_speed", "ball_x_attack", "ball_edge_m",
    "ball_out", "near_home", "near_away",
    "since_seen_s", "until_seen_s", "last_seen_d_diff", "next_seen_d_diff",
    "last_seen_x_attack", "next_seen_x_attack", "last_seen_out", "next_seen_out",
    "interp_d_diff",
    "shape_territory", "shape_depth", "shape_width", "shape_move",
    "median_speed", "still_1s", "still_3s", "still_6s", "tracked_players",
    "control_1s", "control_3s", "control_8s",
)


def _attack_direction(tracking, blocks):
    """Per frame +1 when home attacks towards x = 1 in that half, -1 otherwise."""
    direction = np.ones(len(tracking.ball))
    x = tracking.players[..., 0].astype(np.float64)
    home, away = tracking.team_of_slot == p.HOME, tracking.team_of_slot == p.AWAY
    for _, start, end in blocks:
        part = slice(max(0, start), min(len(direction), end + 1))
        home_x, away_x = x[part][:, home], x[part][:, away]
        if np.isfinite(home_x).any() and np.isfinite(away_x).any():
            direction[part] = 1. if np.nanmean(home_x) <= np.nanmean(away_x) else -1.
    return direction


def _carry(values, seen, forward=True):
    """Value at the last (forward) or next (backward) frame where `seen` holds; NaN if none."""
    index = np.arange(len(values))
    if forward:
        source = np.maximum.accumulate(np.where(seen, index, -1))
    else:
        source = np.minimum.accumulate(np.where(seen, index, len(values))[::-1])[::-1]
    valid = (source >= 0) & (source < len(values))
    out = np.full(len(values), np.nan)
    out[valid] = values[source[valid]]
    distance = np.where(valid, np.abs(index - source), np.nan)
    return out, distance


def _rolling_mean(values, radius):
    finite = np.isfinite(values)
    counts = rolling_sum(finite, radius)
    return np.divide(rolling_sum(np.where(finite, values, 0.), radius), counts,
                     out=np.full(len(values), np.nan), where=counts > 0)


def frame_features(tracking, blocks, *, fps=p.FPS):
    """(n_frames, len(FEATURE_NAMES)) float array; NaN outside blocks or where undefined."""
    length, width = tracking.pitch
    n = len(tracking.ball)
    ball = tracking.ball_xy64 if tracking.ball_xy64 is not None else tracking.ball[:, :2].astype(np.float64)
    seen = np.isfinite(ball).all(axis=1)
    direction = _attack_direction(tracking, blocks)

    distances = nearest_team_distances(tracking, ball)
    distances[~np.isfinite(distances)] = np.nan
    distances[~seen] = np.nan
    d_diff = distances[:, p.HOME] - distances[:, p.AWAY]
    speed = ball_speed_mps(ball, tracking.pitch, fps=fps)
    x_attack = direction * (ball[:, 0] - 0.5) * 2.
    edge = np.minimum.reduce([ball[:, 0] * length, (1 - ball[:, 0]) * length,
                              ball[:, 1] * width, (1 - ball[:, 1]) * width])
    out = out_of_pitch(ball, pitch=tracking.pitch).astype(float)
    out[~seen] = np.nan

    players = tracking.players[..., :2].astype(np.float64)
    gaps = np.linalg.norm((players - ball[:, None, :]) * tracking.pitch, axis=2)
    near = gaps <= NEAR_RADIUS_M
    near_home = np.where(seen, near[:, tracking.team_of_slot == p.HOME].sum(axis=1), np.nan)
    near_away = np.where(seen, near[:, tracking.team_of_slot == p.AWAY].sum(axis=1), np.nan)

    clip = GAP_CLIP_S * fps
    last_diff, since = _carry(d_diff, seen & np.isfinite(d_diff), forward=True)
    next_diff, until = _carry(d_diff, seen & np.isfinite(d_diff), forward=False)
    last_x, _ = _carry(x_attack, seen, forward=True)
    next_x, _ = _carry(x_attack, seen, forward=False)
    last_out, _ = _carry(out, seen, forward=True)
    next_out, _ = _carry(out, seen, forward=False)
    since, until = np.minimum(np.nan_to_num(since, nan=clip), clip), np.minimum(np.nan_to_num(until, nan=clip), clip)

    filled, interpolated = interpolate_short_gaps(ball, INTERPOLATE_FRAMES, pitch=tracking.pitch,
                                                  fps=fps, max_speed_mps=40.)
    interp = nearest_team_distances(tracking, filled)
    interp_diff = np.where(interpolated, interp[:, p.HOME] - interp[:, p.AWAY], np.nan)
    interp_diff[~np.isfinite(interp_diff)] = np.nan

    shape = shape_features(tracking, blocks, fps=fps)

    player_speed = np.where(np.isfinite(tracking.players).all(axis=2), tracking.players[..., 2], np.nan)
    tracked = np.isfinite(player_speed).sum(axis=1).astype(float)
    median = np.full(n, np.nan)
    has = tracked > 0
    median[has] = np.nanmedian(player_speed[has], axis=1)
    still = [_rolling_mean(median, round(s * fps)) for s in STILL_WINDOWS_S]

    nearest = np.fmin(distances[:, p.HOME], distances[:, p.AWAY])
    slow = np.isfinite(speed) & (speed <= CONTROL_SPEED_MPS) & (nearest <= CONTROL_RADIUS_M)
    home_ctrl = (slow & (distances[:, p.HOME] < distances[:, p.AWAY])).astype(float)
    away_ctrl = (slow & (distances[:, p.AWAY] < distances[:, p.HOME])).astype(float)
    control = [(rolling_sum(home_ctrl, round(s * fps)) - rolling_sum(away_ctrl, round(s * fps)))
               / (2 * round(s * fps) + 1) for s in CONTROL_WINDOWS_S]

    columns = [
        seen.astype(float), distances[:, p.HOME], distances[:, p.AWAY], d_diff, speed, x_attack,
        np.where(seen, edge, np.nan), out, near_home, near_away,
        since / fps, until / fps, last_diff, next_diff, last_x, next_x, last_out, next_out,
        interp_diff,
        shape[:, 0], shape[:, 1], shape[:, 2], shape[:, 3],
        median, still[0], still[1], still[2], tracked,
        control[0], control[1], control[2],
    ]
    features = np.column_stack(columns).astype(np.float64)
    inside = np.zeros(n, dtype=bool)
    for _, start, end in blocks:
        inside[max(0, start):min(n, end + 1)] = True
    features[~inside] = np.nan
    return features
