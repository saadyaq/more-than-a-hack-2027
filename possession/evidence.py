"""Combine direct ball control with weak automatic phases and stoppage evidence.

Scores are additive log-evidence, not probabilities. Missing observations supply zero
evidence; no ground truth is used. Interval endpoints follow the starter's half-open rule.
"""

from types import SimpleNamespace

import numpy as np

from games import GAME, half_video_blocks
from possession import params as p
from possession.shape import shape_logit
from possession.ball import ball_speed_mps, interpolate_short_gaps, out_of_pitch


# Parameters exposed for external-data searches; competition defaults stay untouched.
EVIDENCE_PARAMS = frozenset({
    'MAX_GAP_FRAMES', 'MAX_INTERPOLATION_SPEED_MPS', 'SPEED_MEDIAN_FRAMES', 'OUT_MARGIN_M',
    'CONTROL_RADIUS_M', 'CONTROL_SPEED_MPS', 'CONTEST_MARGIN_M', 'W_CONTROL',
    'INTERPOLATED_CONTROL_WEIGHT', 'W_TAG_ATTACK', 'W_TAG_DEFEND', 'TAG_CAP', 'W_OUT',
    'STILL_WINDOW_S', 'STILL_SPEED_MPS', 'W_STILL', 'W_STILL_NO_BALL', 'RESTART_LEAD_S',
    'W_RESTART_DEAD', 'RESTART_ALIVE_S', 'W_RESTART_ALIVE', 'W_PREKICKOFF',
    'W_SHAPE', 'SHAPE_BALL_FACTOR',
})


def evidence_config(params=None):
    """Copy overrides locally so grid searches cannot mutate competition defaults."""
    params = {} if params is None else dict(params)
    unknown = set(params) - EVIDENCE_PARAMS
    if unknown:
        raise ValueError('Unknown evidence parameters: {}'.format(sorted(unknown)))
    return SimpleNamespace(**{name: params.get(name, getattr(p, name)) for name in EVIDENCE_PARAMS})


def normalise(value):
    return str(value).strip().casefold()


def phase_intervals(events, n_frames, offensive_only=False, *, config=p):
    """Yield clipped team evidence intervals, ignoring unknown teams and unrelated codes."""
    for event in events.to_dict("records"):
        team = p.TEAM_NAMES.get(normalise(event.get("Team")))
        code = normalise(event.get("code"))
        if team is None:
            continue
        if code in p.ATTACK_CODES:
            weight = config.W_TAG_ATTACK
        elif not offensive_only and code in p.DEFEND_CODES:
            team, weight = 1 - team, config.W_TAG_DEFEND
        else:
            continue
        start, stop = max(0, int(event["start_frame"])), min(n_frames, int(event["end_frame"]))
        if start < stop:
            yield team, start, stop, weight


def tag_evidence(events, n_frames, *, config=p):
    """Cap each team's overlapping phase prior; restart terms are separate evidence."""
    # Integer coverage counts leave exact zeros after intervals, without float-sum residue.
    changes = np.zeros((n_frames + 1, 2, 2), dtype=np.int64)
    for team, start, stop, weight in phase_intervals(events, n_frames, config=config):
        kind = 0 if weight == config.W_TAG_ATTACK else 1
        changes[start, team, kind] += 1
        changes[stop, team, kind] -= 1
    evidence = np.zeros((n_frames, 3))
    evidence[:, :2] = np.minimum(
        np.cumsum(changes[:-1], axis=0) @ np.array([config.W_TAG_ATTACK, config.W_TAG_DEFEND]), config.TAG_CAP)
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


def control_evidence(tracking, ball_xy, speed, interpolated, *, config=p):
    """Exact contests give neither team evidence; uncontested control gets full weight."""
    distances = nearest_team_distances(tracking, ball_xy)
    closest = np.argmin(distances, axis=1)
    frames = np.arange(len(ball_xy))
    nearest = distances[frames, closest]
    eligible = ((nearest <= config.CONTROL_RADIUS_M) & np.isfinite(speed)
                & (speed <= config.CONTROL_SPEED_MPS))
    frames = frames[eligible]
    advantage = distances[frames, 1 - closest[frames]] - nearest[frames]
    strength = config.W_CONTROL * np.clip(advantage / config.CONTEST_MARGIN_M, 0, 1)
    strength *= np.where(interpolated[frames], config.INTERPOLATED_CONTROL_WEIGHT, 1.)
    evidence = np.zeros((len(ball_xy), 3))
    evidence[frames, closest[frames]] = strength
    return evidence


def rolling_sum(values, radius):
    """Clipped centred windows share one definition for stillness and ball absence."""
    cumulative = np.concatenate(([0.], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    return cumulative[np.minimum(len(values), index + radius + 1)] - cumulative[np.maximum(0, index - radius)]


def stillness_evidence(tracking, observed, *, fps=p.FPS, config=p):
    """Average finite per-frame player medians; untracked frames cannot imply stillness."""
    speeds = np.where(np.isfinite(tracking.players).all(axis=2), tracking.players[..., 2], np.nan)
    available = np.isfinite(speeds).any(axis=1)
    medians = np.full(len(speeds), np.nan)
    medians[available] = np.nanmedian(speeds[available], axis=1)
    radius = round(config.STILL_WINDOW_S * fps)
    counts = rolling_sum(available, radius)
    means = np.divide(rolling_sum(np.nan_to_num(medians), radius), counts,
                      out=np.full(len(medians), np.nan), where=counts > 0)
    still = means < config.STILL_SPEED_MPS
    no_ball = rolling_sum(observed, radius) == 0
    evidence = np.zeros((len(observed), 3))
    evidence[:, p.DEAD] = still * (config.W_STILL + no_ball * config.W_STILL_NO_BALL)
    return evidence


def restart_evidence(events, n_frames, blocks, *, fps=p.FPS, config=p):
    """Set-piece starts mark restarts; only half-start tags extend D back to a half boundary."""
    evidence = np.zeros((n_frames, 3))
    half_bounds = {half: (start, end + 1) for half, start, end in blocks}
    kickoff_halves = set()
    lead, alive = round(config.RESTART_LEAD_S * fps), round(config.RESTART_ALIVE_S * fps)
    for event in events.to_dict("records"):
        if normalise(event.get("code")) != p.SET_PIECES_CODE:
            continue
        start = int(event["start_frame"])
        before, stop = max(0, start - lead), min(n_frames, start)
        if before < stop:
            evidence[before:stop, p.DEAD] += config.W_RESTART_DEAD
        team = p.TEAM_NAMES.get(normalise(event.get("Team")))
        after, stop = max(0, start), min(n_frames, start + alive)
        if team is not None and after < stop:
            evidence[after:stop, team] += config.W_RESTART_ALIVE
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
            evidence[before:stop, p.DEAD] += config.W_PREKICKOFF
    return evidence


def build_evidence(tracking, events, blocks=None, *, fps=p.FPS, params=None):
    """Build evidence with local overrides; empty tags and explicit blocks are valid.

    Omitted blocks retain the competition timeline. No overrides preserves the
    accepted v1/v2/v3 evidence exactly; frames outside the selected blocks are zero.
    """
    config = evidence_config(params)
    if blocks is None:
        blocks = half_video_blocks(GAME)
    ball_xy = tracking.ball_xy64
    if ball_xy is None:
        ball_xy = tracking.ball[:, :2].astype(np.float64)
    observed = np.isfinite(ball_xy).all(axis=1)
    filled, interpolated = interpolate_short_gaps(
        ball_xy, config.MAX_GAP_FRAMES, pitch=tracking.pitch, fps=fps,
        max_speed_mps=config.MAX_INTERPOLATION_SPEED_MPS)
    speed = ball_speed_mps(filled, tracking.pitch, fps=fps,
                           median_frames=config.SPEED_MEDIAN_FRAMES)
    evidence = control_evidence(tracking, filled, speed, interpolated, config=config)
    evidence += tag_evidence(events, len(ball_xy), config=config)
    evidence[:, p.DEAD] += config.W_OUT * out_of_pitch(ball_xy, config.OUT_MARGIN_M, pitch=tracking.pitch)
    evidence += stillness_evidence(tracking, observed, fps=fps, config=config)
    evidence += restart_evidence(events, len(ball_xy), blocks, fps=fps, config=config)
    if config.W_SHAPE:
        # Team shape speaks for the ball when the tracker has lost it; split the log-odds
        # evenly so H and A move apart symmetrically and D is untouched.
        weight = config.W_SHAPE * np.where(observed, config.SHAPE_BALL_FACTOR, 1.)
        half_logit = 0.5 * weight * shape_logit(tracking, blocks, fps=fps)
        evidence[:, p.HOME] += half_logit
        evidence[:, p.AWAY] -= half_logit
    graded = np.zeros(len(ball_xy), dtype=bool)
    for _, start, end in blocks:
        graded[max(0, start):max(0, min(len(graded), end + 1))] = True
    evidence[~graded] = 0
    return evidence
