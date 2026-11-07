"""Team-shape evidence for possession when the tracker has lost the ball.

The competition tracker loses the ball in long blackouts (median ~11 s, 41 % of graded frames), so
for long stretches the only signal is the geometry of the 22 players. A team with the ball pushes
into the opponent's half, stretches wide and long to create passing lanes, and moves forward; the
defending team drops into a compact block. These features capture that, per frame:

    territory  where the whole block of players sits, along home's attacking direction
    depth      home minus away spread along the pitch length (std, metres)
    width      home minus away spread across the pitch (std, metres)
    move       mean forward velocity of all players along home's attacking direction (m/s)

Every feature flips sign when home and away are swapped, and the logistic model has no intercept,
so the model is symmetric between the teams: it cannot learn a home advantage from the external
games that would not transfer. It is trained on external Metrica games only (train_shape.py).
"""

import json
from pathlib import Path

import numpy as np

from possession import params as p

# Params
MIN_PLAYERS = 6  # Fewer tracked players per team make centroid and spread meaningless.
SMOOTH_S = 1.0  # Team shape changes over seconds; average out tracking noise over +/- 1 s.
MAX_PLAYER_SPEED_MPS = 12.0  # Faster apparent motion is a slot identity switch, not running.
FEATURES = ("territory", "depth", "width", "move")
MODEL_PATH = Path(__file__).resolve().parent / "shape_model.json"


def _rolling_nanmean(values, radius):
    """Centred mean over +/- radius rows, ignoring NaN; NaN where the window has no value."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.)
    pad = lambda a: np.concatenate((np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)))
    sums, counts = pad(filled), pad(finite.astype(float))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - radius), np.minimum(len(values), index + radius + 1)
    total, count = sums[high] - sums[low], counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _team_stats(xy, members):
    """Per-frame centroid (x, y), spread (x, y) and player count for one team, in metres."""
    team = xy[:, members]
    count = np.isfinite(team[..., 0]).sum(axis=1)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as quiet:
        quiet.filter(RuntimeWarning)
        centre = np.nanmean(team, axis=1)
        spread = np.nanstd(team, axis=1)
    enough = count >= MIN_PLAYERS
    centre[~enough] = np.nan
    spread[~enough] = np.nan
    return centre, spread


def shape_features(tracking, blocks, *, fps=p.FPS):
    """(n_frames, len(FEATURES)) array, NaN outside blocks or where a team is under-tracked."""
    length, width = tracking.pitch
    xy = tracking.players[..., :2].astype(np.float64) * (length, width)
    home, away = tracking.team_of_slot == p.HOME, tracking.team_of_slot == p.AWAY
    (home_c, home_s), (away_c, away_s) = _team_stats(xy, home), _team_stats(xy, away)

    velocity = np.full(xy.shape[:2], np.nan)
    velocity[1:-1] = (xy[2:, :, 0] - xy[:-2, :, 0]) * fps / 2.
    velocity[np.abs(velocity) > MAX_PLAYER_SPEED_MPS] = np.nan
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as quiet:
        quiet.filter(RuntimeWarning)
        mean_vx = np.nanmean(velocity, axis=1)

    features = np.full((len(xy), len(FEATURES)), np.nan)
    radius = int(round(SMOOTH_S * fps))
    for _, start, end in blocks:
        start, stop = max(0, start), min(len(xy), end + 1)
        if start >= stop:
            continue
        part = slice(start, stop)
        # Home attacks towards +x when its average position over the half is the lower one.
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as quiet:
            quiet.filter(RuntimeWarning)
            direction = 1. if np.nanmean(home_c[part, 0]) <= np.nanmean(away_c[part, 0]) else -1.
        block = np.column_stack((
            direction * ((home_c[part, 0] + away_c[part, 0]) / 2. - length / 2.) / (length / 2.),
            (home_s[part, 0] - away_s[part, 0]) / 10.,
            (home_s[part, 1] - away_s[part, 1]) / 10.,
            direction * mean_vx[part],
        ))
        features[part] = _rolling_nanmean(block, radius)
    return features


def fit_logistic(X, y, l2=1.0, iterations=50):
    """No-intercept logistic regression by Newton's method; returns (weights, scale).

    Features are scaled by their standard deviation but not centred: centring would add an
    implicit intercept and break the home/away symmetry.
    """
    scale = np.std(X, axis=0)
    scale[scale == 0] = 1.
    Z = X / scale
    w = np.zeros(Z.shape[1])
    for _ in range(iterations):
        prob = 1. / (1. + np.exp(-Z @ w))
        gradient = Z.T @ (prob - y) + l2 * w
        hessian = (Z * (prob * (1. - prob))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w, scale


def load_model(path=MODEL_PATH):
    model = json.loads(Path(path).read_text())
    if tuple(model["features"]) != FEATURES:
        raise ValueError("shape model was trained on different features")
    return np.array(model["weights"]), np.array(model["scale"])


def shape_logit(tracking, blocks, *, fps=p.FPS, model=None):
    """Log-odds that home has the ball from team shape alone; 0 where features are missing."""
    weights, scale = load_model() if model is None else model
    features = shape_features(tracking, blocks, fps=fps)
    logit = (features / scale) @ weights
    return np.where(np.isfinite(logit), logit, 0.)
