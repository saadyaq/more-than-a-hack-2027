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
