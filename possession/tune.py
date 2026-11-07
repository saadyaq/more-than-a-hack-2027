"""Tune the v3 evidence and decoder on external Metrica games only, never on the competition match.

    .venv/bin/python -m possession.tune                 # writes possession/tuned_v5.json

The organisers require every tuned parameter to be tuned on external data. The objective is the
competition's own ranking metric, spellF1, averaged over the three Metrica sample games, each
degraded with three seeds to resemble the competition's automatic tracking (41 % ball absence).

Coordinate descent: one parameter at a time over a small grid, keeping a change only when it
improves the objective by more than MIN_GAIN, until a full pass changes nothing. Deterministic.
Leave-one-game-out folds are reported first, so the held-out score says whether the tuned values
generalise to a game they never saw, which is the situation the competition match is in.

Not tuned, because the external games have no Smart Tagging: W_TAG_ATTACK, W_TAG_DEFEND, TAG_CAP,
W_RESTART_*, RESTART_*_S and W_PREKICKOFF keep their reasoned v3 values.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from possession import params as p
from possession.decode import decode
from possession.evidence import build_evidence
from possession.external import FPS, degrade, load_metrica_events, load_metrica_game, possession_truth
from possession.external_eval import score

# Params
GAMES = (1, 2, 3)  # Every public Metrica sample game with synchronised events.
SEEDS = (0, 1, 2)  # Three degradations per game average out the simulator's randomness.
MAX_PASSES = 4  # Coordinate descent stops earlier once a full pass changes nothing.
MIN_GAIN = 1e-4  # Smaller improvements are ties; ties keep the current value.
OUTPUT = Path(__file__).resolve().parent / "tuned_v5.json"  # v4 used the old geometric degrader.

EVIDENCE_GRID = {  # Ranges span what is physically plausible, centred on the v3 values.
    # Values that ended on a grid edge in the v4 search were extended outwards.
    "CONTROL_RADIUS_M": [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "CONTROL_SPEED_MPS": [4.0, 7.0, 10.0, 14.0, 20.0, 30.0],
    "CONTEST_MARGIN_M": [0.25, 0.5, 1.0, 2.0],  # 0 would divide by zero in the contest scaling.
    "INTERPOLATED_CONTROL_WEIGHT": [0.0, 0.5, 1.0],
    "MAX_GAP_FRAMES": [0, 12, 25, 50, 100, 250],
    "W_OUT": [1.0, 3.0, 6.0, 12.0],
    "STILL_WINDOW_S": [1.0, 2.0, 3.0, 5.0],
    "STILL_SPEED_MPS": [0.8, 1.2, 1.6, 2.0, 2.5],
    "W_STILL": [0.25, 0.5, 1.0, 2.0],
    "W_STILL_NO_BALL": [0.0, 0.5, 1.0, 2.0, 4.0],
    "W_SHAPE": [0.0, 0.05, 0.1, 0.2, 0.4],  # Team shape when the ball is lost (shape.py).
    "SHAPE_BALL_FACTOR": [0.0, 0.25, 0.5, 1.0],
}
TRANSITION_GRID = {  # W_CONTROL stays at 2.0 and anchors the scale of every penalty.
    "P_SWITCH": [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0],
    "P_TO_DEAD": [5.0, 10.0, 15.0, 25.0, 40.0, 60.0],
    "P_FROM_DEAD_EXTRA": [0.0, 5.0, 15.0],  # Restart cost on top of the fake-stoppage floor.
}


def start_point():
    """The v3 values, so the search starts from the reasoned method."""
    point = {name: getattr(p, name) for name in EVIDENCE_GRID}
    point.update(P_SWITCH=p.P_SWITCH, P_TO_DEAD=p.P_TO_DEAD,
                 P_FROM_DEAD_EXTRA=p.P_FROM_DEAD - max(0., p.P_SWITCH - p.P_TO_DEAD))
    return point


def start_from(path):
    """A previous tuning result as the starting point; parameters it lacks keep v3 values."""
    previous = json.loads(Path(path).read_text())
    point = dict(start_point(), **previous["evidence"])
    penalties = previous["penalties"]
    point.update(P_SWITCH=penalties["P_SWITCH"], P_TO_DEAD=penalties["P_TO_DEAD"],
                 P_FROM_DEAD_EXTRA=penalties["P_FROM_DEAD"]
                 - max(0., penalties["P_SWITCH"] - penalties["P_TO_DEAD"]))
    return point


def penalties_of(point):
    """Leaving D costs at least the turnover minus the entry, so fake stoppages never pay."""
    switch, to_dead = point["P_SWITCH"], point["P_TO_DEAD"]
    penalties = {"P_SWITCH": switch, "P_TO_DEAD": to_dead,
                 "P_FROM_DEAD": max(0., switch - to_dead) + point["P_FROM_DEAD_EXTRA"]}
    p.check_no_cheap_fake_stoppage(penalties)
    return penalties


def load_cases(games, seeds, profile="v1"):
    """[(game, seed, tracking, blocks, truth)] degraded once, reused by every evaluation."""
    cases = []
    for game in games:
        tracking, blocks, home = load_metrica_game(game)
        truth = possession_truth(load_metrica_events(game), len(tracking.ball), blocks, home)
        for seed in seeds:
            cases.append((game, seed, degrade(tracking, seed, blocks=blocks, profile=profile),
                          blocks, truth))
    return cases


class Objective:
    """Mean spellF1 over cases, memoising evidence because transitions do not change it."""

    def __init__(self, cases):
        self.cases = cases
        self.evidence = {}

    def _evidence(self, index, point):
        key = (index,) + tuple(point[name] for name in EVIDENCE_GRID)
        if key not in self.evidence:
            _, _, tracking, blocks, _ = self.cases[index]
            self.evidence[key] = build_evidence(
                tracking, pd.DataFrame(), blocks=blocks, fps=FPS,
                params={name: point[name] for name in EVIDENCE_GRID})
        return self.evidence[key]

    def metrics(self, point):
        penalties = penalties_of(point)
        rows = []
        for index, (_, _, _, blocks, truth) in enumerate(self.cases):
            labels = decode(self._evidence(index, point), blocks=blocks, penalties=penalties)
            rows.append(score(truth, labels))
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    def __call__(self, point):
        return self.metrics(point)["spell_f1_mean"]


def coordinate_descent(objective, point, log):
    point, best = dict(point), objective(point)
    log("  start spellF1 {:.4f}".format(best))
    grid = dict(EVIDENCE_GRID, **TRANSITION_GRID)
    for sweep in range(1, MAX_PASSES + 1):
        changed = False
        for name, values in grid.items():
            # Evidence values first, so each tried evidence setting is computed once per pass.
            for value in values:
                if value == point[name]:
                    continue
                candidate = dict(point, **{name: value})
                result = objective(candidate)
                if result > best + MIN_GAIN:
                    point, best, changed = candidate, result, True
            log("  pass {} {:<28} -> {:<6} spellF1 {:.4f}".format(sweep, name, point[name], best))
        if not changed:
            break
    return point, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-folds", action="store_true", help="skip leave-one-game-out")
    parser.add_argument("--start", help="previous tuned_*.json to start from (default: v3)")
    parser.add_argument("--output", default=str(OUTPUT), help="where to write the result")
    args = parser.parse_args(argv)

    def log(line):
        print(line, flush=True)

    origin = (lambda: start_from(args.start)) if args.start else start_point
    cases = load_cases(GAMES, SEEDS)
    if not args.no_folds:
        for held_out in GAMES:
            log("fold: tune on {}, test on game {}".format(
                [g for g in GAMES if g != held_out], held_out))
            train = Objective([case for case in cases if case[0] != held_out])
            test = Objective([case for case in cases if case[0] == held_out])
            tuned, _ = coordinate_descent(train, origin(), log)
            log("  held-out game {}: start spellF1 {:.4f} -> tuned {:.4f}".format(
                held_out, test(origin()), test(tuned)))

    log("final: tune on all games")
    everything = Objective(cases)
    tuned, best = coordinate_descent(everything, origin(), log)
    before, after = everything.metrics(origin()), everything.metrics(tuned)
    for label, metrics in (("start", before), ("tuned", after)):
        log("{:<6} spellF1 {:.4f}  macroF1 {:.4f}  possessions {:.1f}/{:.1f}".format(
            label, metrics["spell_f1_mean"], metrics["macro_f1"],
            metrics["n_pred_spells"], metrics["n_gt_spells"]))
    result = {"evidence": {name: tuned[name] for name in EVIDENCE_GRID},
              "penalties": penalties_of(tuned),
              "objective": {"metric": "spell_f1_mean", "games": list(GAMES),
                            "seeds": list(SEEDS), "value": best}}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
