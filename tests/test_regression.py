"""Whole-file hash comparison only: never expose tracking or frame labels on failure."""

import hashlib
from pathlib import Path
import tempfile

import pytest

from possession.baseline import majority_smooth, nearest_team
from possession.data import load_tracking
from possession.io import write_submission


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.digest()


@pytest.mark.data
def test_baseline50_byte_identical():
    results = Path(__file__).resolve().parents[1] / "evaluation" / "results"
    try:
        labels = majority_smooth(nearest_team(load_tracking()), 50)
        with tempfile.TemporaryDirectory(prefix="regression_", dir=results) as folder:
            actual = write_submission(Path(folder) / "baseline.csv", labels)
            matches = digest(actual) == digest(results / "baseline_smooth50.csv")
    except Exception:
        pytest.fail("Data regression could not complete; details suppressed to protect match data",
                    pytrace=False)
    if not matches:
        pytest.fail("Baseline SHA-256 comparison failed (contents suppressed)", pytrace=False)


@pytest.mark.data
@pytest.mark.parametrize("method", ["v1", "v2", "v3", "v4", "v6"])
def test_evidence_methods_valid_and_deterministic(method):
    from evaluation import possession_metrics as pm
    from games import source_frames
    from possession.run import METHODS

    results = Path(__file__).resolve().parents[1] / "evaluation" / "results"
    try:
        tracking = load_tracking()
        hashes, valid = [], True
        with tempfile.TemporaryDirectory(prefix=method + "_regression_", dir=results) as folder:
            for repeat in range(2):
                labels = METHODS[method](tracking)
                path = write_submission(Path(folder) / (str(repeat) + ".csv"), labels)
                hashes.append(digest(path))
                valid = valid and pm.validate_submission(pm.read_label_csv(path), source_frames()) is None
        matches = hashes[0] == hashes[1]
        if method in ("v1", "v2"):
            # Compare against accepted files; never regenerate them during this test.
            matches = matches and hashes[0] == digest(results / (method + ".csv"))
    except Exception:
        pytest.fail("Evidence-method regression could not complete; details suppressed to protect match data",
                    pytrace=False)
    if not valid or not matches:
        pytest.fail("Validation or deterministic/reference hash comparison failed (contents suppressed)",
                    pytrace=False)
