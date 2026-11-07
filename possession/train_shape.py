"""Train the team-shape possession model on external Metrica games only.

    .venv/bin/python -m possession.train_shape           # writes possession/shape_model.json

Training frames are every SUBSAMPLE-th alive frame of each external game, degraded with three
seeds like the tuner, so the model sees the same tracking noise it will meet in the competition.
Leave-one-game-out accuracy and log-loss are printed first, to show whether shape transfers to an
unseen game; the saved model is then fitted on all three games.
"""

import json

import numpy as np

from possession import params as p
from possession.external import degrade, load_metrica_events, load_metrica_game, possession_truth
from possession.shape import FEATURES, MODEL_PATH, fit_logistic, shape_features

# Params
GAMES = (1, 2, 3)  # Every public Metrica sample game with synchronised events.
SEEDS = (0, 1, 2)  # Same degradations as the tuner.
SUBSAMPLE = 5  # Neighbouring frames are near-duplicates; every fifth keeps the fit fast.
L2 = 1.0  # Mild ridge penalty; five standardised features need little regularisation.


def game_rows(game):
    """(X, y) over alive frames with complete features, y = 1 when home has the ball."""
    tracking, blocks, home = load_metrica_game(game)
    truth = possession_truth(load_metrica_events(game), len(tracking.ball), blocks, home)
    xs, ys = [], []
    for seed in SEEDS:
        features = shape_features(degrade(tracking, seed, blocks=blocks), blocks)
        keep = np.isin(truth, ["H", "A"]) & np.isfinite(features).all(axis=1)
        keep[np.arange(len(keep)) % SUBSAMPLE != 0] = False
        xs.append(features[keep])
        ys.append((truth[keep] == "H").astype(float))
    return np.vstack(xs), np.concatenate(ys)


def report(name, w, scale, X, y):
    prob = 1. / (1. + np.exp(-(X / scale) @ w))
    loss = -np.mean(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12))
    print("{:<22} accuracy {:.3f}  log-loss {:.3f}  (chance 0.693)  n {}".format(
        name, np.mean((prob > .5) == (y == 1)), loss, len(y)), flush=True)


def main():
    rows = {game: game_rows(game) for game in GAMES}
    for held_out in GAMES:
        X = np.vstack([rows[g][0] for g in GAMES if g != held_out])
        y = np.concatenate([rows[g][1] for g in GAMES if g != held_out])
        w, scale = fit_logistic(X, y, l2=L2)
        report("held-out game {}".format(held_out), w, scale, *rows[held_out])
    X = np.vstack([rows[g][0] for g in GAMES])
    y = np.concatenate([rows[g][1] for g in GAMES])
    w, scale = fit_logistic(X, y, l2=L2)
    report("all games (train)", w, scale, X, y)
    for name, weight in zip(FEATURES, w):
        print("  {:<10} {:+.3f}".format(name, weight))
    MODEL_PATH.write_text(json.dumps({
        "features": list(FEATURES), "weights": w.tolist(), "scale": scale.tolist(),
        "trained_on": "Metrica public sample games 1-3, degraded seeds 0-2 (external data only)",
    }, indent=2) + "\n")
    print("wrote {}".format(MODEL_PATH))


if __name__ == "__main__":
    main()
