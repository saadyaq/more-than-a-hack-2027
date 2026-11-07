"""Run registered possession methods and report only whole-video aggregates."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from possession.baseline import majority_smooth, nearest_team
from possession.data import load_tracking
from possession.io import write_submission

# Params
DEFAULT_WINDOW = 50  # Published baseline's +/- two seconds smooth tracking flicker at 25 fps.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def baseline(tracking, window):
    return majority_smooth(nearest_team(tracking), window)


METHODS = {"baseline": baseline}


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
    parser.add_argument("--name", default="ours_baseline50", help="submission filename stem")
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    main()
