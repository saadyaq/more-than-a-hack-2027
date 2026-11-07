"""Keep synthetic tests usable without access to the gated match bundle."""

import os

import pytest


def pytest_collection_modifyitems(items):
    if not os.environ.get("MTAH_DATA_DIR"):
        skip = pytest.mark.skip(reason="MTAH_DATA_DIR is unset")
        for item in items:
            if item.get_closest_marker("data"):
                item.add_marker(skip)
