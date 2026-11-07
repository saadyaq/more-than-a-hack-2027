"""Vectorised official baseline, preserving metric distances and deterministic ties."""

import numpy as np
from possession.params import HOME, AWAY, NOBODY


def nearest_team(tracking):
    """First slot wins exact ties, as in the starter's strict-less-than loop."""
    players = tracking.players_xy64
    ball = tracking.ball_xy64
    if players is None:
        players = tracking.players[..., :2].astype(np.float64)
    if ball is None:
        ball = tracking.ball[..., :2].astype(np.float64)
    codes = np.full(len(ball), NOBODY, dtype=np.int8)
    if players.shape[1] == 0:
        return codes
    # Multiply before subtracting: (player - ball) * pitch can round differently.
    delta = players * tracking.pitch - (ball * tracking.pitch)[:, None, :]
    distances = delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
    distances[np.isnan(distances)] = np.inf
    nearest = np.argmin(distances, axis=1)
    seen = np.isfinite(distances[np.arange(len(ball)), nearest])
    codes[seen] = tracking.team_of_slot[nearest[seen]]
    return codes


def majority_smooth(codes, window):
    """Vote within +/- window frames; ignore nobody votes and give home exact ties."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels
