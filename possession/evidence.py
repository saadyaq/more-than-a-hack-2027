"""Combine direct ball control with weak automatic phases and stoppage evidence.

Scores are additive log-evidence, not probabilities. Missing observations supply zero
evidence; no ground truth is used. Interval endpoints follow the starter's half-open rule.
"""

import numpy as np

from games import GAME, half_video_blocks
from possession import params as p
from possession.ball import ball_speed_mps, interpolate_short_gaps, out_of_pitch


def normalise(value):
    return str(value).strip().casefold()


def phase_intervals(events, n_frames, offensive_only=False):
    """Yield clipped team evidence intervals, ignoring unknown teams and unrelated codes."""
    for event in events.to_dict("records"):
        team = p.TEAM_NAMES.get(normalise(event.get("Team")))
        code = normalise(event.get("code"))
        if team is None:
            continue
        if code in p.ATTACK_CODES:
            weight = p.W_TAG_ATTACK
        elif not offensive_only and code in p.DEFEND_CODES:
            team, weight = 1 - team, p.W_TAG_DEFEND
        else:
            continue
        start, stop = max(0, int(event["start_frame"])), min(n_frames, int(event["end_frame"]))
        if start < stop:
            yield team, start, stop, weight


def tag_evidence(events, n_frames):
    """Cap each team's overlapping phase prior; restart terms are separate evidence."""
    # Integer coverage counts leave exact zeros after intervals, without float-sum residue.
    changes = np.zeros((n_frames + 1, 2, 2), dtype=np.int64)
    for team, start, stop, weight in phase_intervals(events, n_frames):
        kind = 0 if weight == p.W_TAG_ATTACK else 1
        changes[start, team, kind] += 1
        changes[stop, team, kind] -= 1
    evidence = np.zeros((n_frames, 3))
    evidence[:, :2] = np.minimum(
        np.cumsum(changes[:-1], axis=0) @ np.array([p.W_TAG_ATTACK, p.W_TAG_DEFEND]), p.TAG_CAP)
    return evidence


def nearest_team_distances(tracking, ball_xy):
    """Missing players cannot win a nearest-player comparison; absent teams stay infinite."""
    players = tracking.players_xy64
    if players is None:
        players = tracking.players[..., :2].astype(np.float64)
    delta = (players - ball_xy[:, None, :]) * tracking.pitch
    squared = np.sum(delta * delta, axis=2)
    squared[~np.isfinite(squared)] = np.inf
    distances = np.full((len(ball_xy), 2), np.inf)
    for team in (p.HOME, p.AWAY):
        slots = tracking.team_of_slot == team
        if slots.any():
            distances[:, team] = np.sqrt(np.min(squared[:, slots], axis=1))
    return distances


def control_evidence(tracking, ball_xy, speed, interpolated):
    """Exact contests give neither team evidence; uncontested control gets full weight."""
    distances = nearest_team_distances(tracking, ball_xy)
    closest = np.argmin(distances, axis=1)
    frames = np.arange(len(ball_xy))
    nearest = distances[frames, closest]
    eligible = ((nearest <= p.CONTROL_RADIUS_M) & np.isfinite(speed)
                & (speed <= p.CONTROL_SPEED_MPS))
    frames = frames[eligible]
    advantage = distances[frames, 1 - closest[frames]] - nearest[frames]
    strength = p.W_CONTROL * np.clip(advantage / p.CONTEST_MARGIN_M, 0, 1)
    strength *= np.where(interpolated[frames], p.INTERPOLATED_CONTROL_WEIGHT, 1.)
    evidence = np.zeros((len(ball_xy), 3))
    evidence[frames, closest[frames]] = strength
    return evidence


def rolling_sum(values, radius):
    """Clipped centred windows share one definition for stillness and ball absence."""
    cumulative = np.concatenate(([0.], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    return cumulative[np.minimum(len(values), index + radius + 1)] - cumulative[np.maximum(0, index - radius)]


def stillness_evidence(tracking, observed, *, fps=p.FPS):
    """Average finite per-frame player medians; untracked frames cannot imply stillness."""
    speeds = np.where(np.isfinite(tracking.players).all(axis=2), tracking.players[..., 2], np.nan)
    available = np.isfinite(speeds).any(axis=1)
    medians = np.full(len(speeds), np.nan)
    medians[available] = np.nanmedian(speeds[available], axis=1)
    radius = round(p.STILL_WINDOW_S * fps)
    counts = rolling_sum(available, radius)
    means = np.divide(rolling_sum(np.nan_to_num(medians), radius), counts,
                      out=np.full(len(medians), np.nan), where=counts > 0)
    still = means < p.STILL_SPEED_MPS
    no_ball = rolling_sum(observed, radius) == 0
    evidence = np.zeros((len(observed), 3))
    evidence[:, p.DEAD] = still * (p.W_STILL + no_ball * p.W_STILL_NO_BALL)
    return evidence


def restart_evidence(events, n_frames, blocks, *, fps=p.FPS):
    """Set-piece starts mark restarts; only half-start tags extend D back to a half boundary."""
    evidence = np.zeros((n_frames, 3))
    half_bounds = {half: (start, end + 1) for half, start, end in blocks}
    kickoff_halves = set()
    lead, alive = round(p.RESTART_LEAD_S * fps), round(p.RESTART_ALIVE_S * fps)
    for event in events.to_dict("records"):
        if normalise(event.get("code")) != p.SET_PIECES_CODE:
            continue
        start = int(event["start_frame"])
        before, stop = max(0, start - lead), min(n_frames, start)
        if before < stop:
            evidence[before:stop, p.DEAD] += p.W_RESTART_DEAD
        team = p.TEAM_NAMES.get(normalise(event.get("Team")))
        after, stop = max(0, start), min(n_frames, start + alive)
        if team is not None and after < stop:
            evidence[after:stop, team] += p.W_RESTART_ALIVE
        if normalise(event.get("Type")) != p.KICKOFF_START_TYPE:
            continue  # After-goal kickoffs get only the ordinary restart lead window.
        half = p.HALF_NAMES.get(normalise(event.get("Half")))
        if half not in half_bounds:
            continue
        if half in kickoff_halves:
            raise ValueError("multiple half-start kickoff tags for one half")
        kickoff_halves.add(half)
        lower, upper = half_bounds[half]
        before, stop = max(0, lower), min(n_frames, upper, start)
        if before < stop:
            evidence[before:stop, p.DEAD] += p.W_PREKICKOFF
    return evidence


def build_evidence(tracking, events, blocks=None, *, fps=p.FPS):
    """Compute all fixed v1 terms, then discard evidence outside the graded halves."""
    if blocks is None:
        blocks = half_video_blocks(GAME)
    ball_xy = tracking.ball_xy64
    if ball_xy is None:
        ball_xy = tracking.ball[:, :2].astype(np.float64)
    observed = np.isfinite(ball_xy).all(axis=1)
    filled, interpolated = interpolate_short_gaps(ball_xy, pitch=tracking.pitch, fps=fps)
    speed = ball_speed_mps(filled, tracking.pitch, fps=fps)
    evidence = control_evidence(tracking, filled, speed, interpolated)
    evidence += tag_evidence(events, len(ball_xy))
    evidence[:, p.DEAD] += p.W_OUT * out_of_pitch(ball_xy, pitch=tracking.pitch)
    evidence += stillness_evidence(tracking, observed, fps=fps)
    evidence += restart_evidence(events, len(ball_xy), blocks, fps=fps)
    graded = np.zeros(len(ball_xy), dtype=bool)
    for _, start, end in blocks:
        graded[max(0, start):max(0, min(len(graded), end + 1))] = True
    evidence[~graded] = 0
    return evidence
