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
from possession.params import DEFAULT_WINDOW, V1_TRANSITION_PENALTIES, V2_TRANSITION_PENALTIES

# Params
RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def baseline(tracking, window):
    return majority_smooth(nearest_team(tracking), window)


def _evidence_decode(tracking, penalties=None):
    """Both versions share unchanged evidence and differ only in transition penalties."""
    start = perf_counter()
    evidence = build_evidence(tracking, load_smart_tagging())
    print("evidence: {:.3f} s".format(perf_counter() - start))
    start = perf_counter()
    labels = decode(evidence, penalties=penalties)
    print("decoder: {:.3f} s".format(perf_counter() - start))
    return labels


def v1(tracking, window=None):
    """Preserve the accepted v1 penalties; the baseline smoothing window is unused."""
    return _evidence_decode(tracking, penalties=V1_TRANSITION_PENALTIES)


def v2(tracking, window=None):
    """Preserve the accepted v2 penalties; the baseline smoothing window is unused."""
    return _evidence_decode(tracking, penalties=V2_TRANSITION_PENALTIES)


def v3(tracking, window=None):
    """Default penalties: v2 plus no fake stoppage cheaper than a turnover."""
    return _evidence_decode(tracking)


METHODS = {"baseline": baseline, "v1": v1, "v2": v2, "v3": v3}


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
