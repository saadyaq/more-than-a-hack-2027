"""Run registered possession methods and report only whole-video aggregates."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from possession.baseline import majority_smooth, nearest_team
from possession.data import load_smart_tagging, load_tracking
from possession.decode import decode
from possession.evidence import build_evidence
from possession.io import write_submission
from possession.params import (DEFAULT_WINDOW, HISTORICAL_METHODS, METHOD_EVIDENCE_PARAMS,
                               METHOD_TRANSITION_PENALTIES, check_no_cheap_fake_stoppage)

# Params
RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def baseline(tracking, window):
    return majority_smooth(nearest_team(tracking), window)


def _evidence_decode(tracking, penalties=None, evidence_params=None):
    """All versions share the evidence model and differ only in their parameter values."""
    start = perf_counter()
    evidence = build_evidence(tracking, load_smart_tagging(), params=evidence_params)
    print("evidence: {:.3f} s".format(perf_counter() - start))
    start = perf_counter()
    labels = decode(evidence, penalties=penalties)
    print("decoder: {:.3f} s".format(perf_counter() - start))
    return labels


def v1(tracking, window=None):
    """Preserve the accepted v1 penalties; the baseline smoothing window is unused."""
    return _evidence_decode(tracking, penalties=METHOD_TRANSITION_PENALTIES["v1"])


def v2(tracking, window=None):
    """Preserve the accepted v2 penalties; the baseline smoothing window is unused."""
    return _evidence_decode(tracking, penalties=METHOD_TRANSITION_PENALTIES["v2"])


def v3(tracking, window=None):
    """Default penalties: v2 plus no fake stoppage cheaper than a turnover."""
    return _evidence_decode(tracking, penalties=METHOD_TRANSITION_PENALTIES["v3"])


def v4(tracking, window=None):
    """v3's model with evidence and penalties tuned on external games (tuned_v4.json)."""
    return _evidence_decode(tracking, penalties=METHOD_TRANSITION_PENALTIES["v4"],
                            evidence_params=METHOD_EVIDENCE_PARAMS["v4"])


def v6(tracking, window=None):
    """v4 plus team-shape evidence, tuned on external games with realistic ball loss."""
    return _evidence_decode(tracking, penalties=METHOD_TRANSITION_PENALTIES["v6"],
                            evidence_params=METHOD_EVIDENCE_PARAMS["v6"])


def v7(tracking, window=None):
    """Learned emissions (gradient boosting, external games) plus Smart Tagging terms."""
    return _learned_decode(tracking, "v7")


def v8(tracking, window=None):
    """v7 retrained on external games degraded like the real tracker's player loss too."""
    return _learned_decode(tracking, "v8")


def _learned_decode(tracking, version):
    """Decode one learned version: its classifier, temperature and penalties."""
    from games import GAME, half_video_blocks
    from possession import learned
    from possession import params as p
    from possession.evidence import restart_evidence, tag_evidence

    start = perf_counter()
    model, config = learned.load(version)
    blocks = half_video_blocks(GAME)
    probs = learned.predict_proba(model, tracking, blocks)
    scores = learned.emissions(probs, np.array(config["prior"]), config["temperature"])
    # Tag and restart weights were set against clear ball control (W_CONTROL per frame). Clear
    # control in the learned model is about temperature * log(0.9 / (1/3)) per frame, so the
    # same ratio keeps their intended strength relative to the ball.
    tag_scale = config["temperature"] * np.log(0.9 * 3) / p.W_CONTROL
    events = load_smart_tagging()
    scores += tag_scale * tag_evidence(events, len(scores))
    scores += tag_scale * restart_evidence(events, len(scores), blocks)
    graded = np.zeros(len(scores), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    scores[~graded] = 0.
    print("learned evidence: {:.3f} s".format(perf_counter() - start))
    return decode(scores, penalties=METHOD_TRANSITION_PENALTIES[version])


METHODS = {"baseline": baseline, "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v6": v6, "v7": v7, "v8": v8}


def validate_method_penalties(methods):
    """Require declared, consistent penalties for every nonhistorical decoder method."""
    for name in methods:
        if name == "baseline":
            continue  # The nearest-team baseline has no transition model.
        assert name in METHOD_TRANSITION_PENALTIES, "registered method lacks transition penalties"
        if name not in HISTORICAL_METHODS:
            check_no_cheap_fake_stoppage(METHOD_TRANSITION_PENALTIES[name])


validate_method_penalties(METHODS)


def possession_count(labels):
    """Count maximal H/A runs; dead balls and abstentions break possessions."""
    if not len(labels):
        return 0
    starts = np.concatenate(([True], labels[1:] != labels[:-1]))
    return int(np.count_nonzero(starts & ((labels == "H") | (labels == "A"))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="centred smoothing half-width in video frames")
    parser.add_argument("--name", help="submission filename stem (default: method name)")
    args = parser.parse_args(argv)
    if args.name is None:
        args.name = "ours_baseline50" if args.method == "baseline" else args.method
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        parser.error("--name must be a filename stem without directories")
    start = perf_counter()
    tracking = load_tracking()
    print("tracking load: {:.3f} s".format(perf_counter() - start))
    labels = METHODS[args.method](tracking, window=args.window)
    path = write_submission(RESULTS_DIR / (args.name + ".csv"), labels)
    counts = {label: int(np.count_nonzero(labels == label)) for label in ("H", "A", "D", "X")}
    print("label counts: {}".format(counts))
    print("possessions: {}".format(possession_count(labels)))
    print("wrote {}".format(path))
    print("total pipeline: {:.3f} s".format(perf_counter() - start))


if __name__ == "__main__":
    main()
