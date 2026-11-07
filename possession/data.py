"""Parse ATD once, retaining original coordinate precision for baseline regression.

The public arrays use float32 for later stages; the separate float64 xy arrays prevent
rounding from changing the nearest player. Missing video frames remain NaN. Only schema
counts are printed: tracking observations must never appear in logs.
"""

from collections import Counter
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from games import DATA_DIR, GAME, GAMES, require, source_frames
from starter.baseline import FIELD_SEP, GROUP_SEP, MISSING, team_of_each_track
from starter.load_smart_tagging import load as load_tags
from possession.params import CACHE_VERSION


@dataclass
class Tracking:
    """Video-indexed observations; xy64 preserves close contests for the ball."""

    players: np.ndarray = field(repr=False)
    ball: np.ndarray = field(repr=False)
    team_of_slot: np.ndarray = field(repr=False)
    pitch: Tuple[float, float]
    players_xy64: Optional[np.ndarray] = field(default=None, repr=False)
    ball_xy64: Optional[np.ndarray] = field(default=None, repr=False)


def _signature(raw_path, meta_path):
    """Team assignments matter too, so metadata changes also invalidate the cache."""
    raw, meta = Path(raw_path).stat(), Path(meta_path).stat()
    return dict(cache_version=CACHE_VERSION, raw_mtime_ns=raw.st_mtime_ns,
                raw_size=raw.st_size, meta_mtime_ns=meta.st_mtime_ns,
                meta_size=meta.st_size)


def _read_cache(path, signature):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        if any(key not in cache or int(cache[key]) != value
               for key, value in signature.items()):
            return None
        return Tracking(cache["players"], cache["ball"], cache["team_of_slot"],
                        tuple(cache["pitch"]), cache["players_xy64"], cache["ball_xy64"])


def _observation(group, counts):
    fields = group.split(",")
    counts[len(fields)] += 1
    if group.startswith(MISSING):
        return (np.nan, np.nan, np.nan)
    if len(fields) < 3:
        raise ValueError("ATD observation has fewer than three fields")
    try:
        return tuple(float(value) for value in fields[:3])
    except ValueError:
        # Conversion errors must not expose the offending tracking value.
        raise ValueError("ATD observation contains a nonnumeric field") from None


def _parse_tracking(raw_path, teams, n_frames, pitch):
    players = np.full((n_frames, len(teams), 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players_xy64 = np.full((n_frames, len(teams), 2), np.nan, dtype=np.float64)
    ball_xy64 = np.full((n_frames, 2), np.nan, dtype=np.float64)
    player_fields, ball_fields = Counter(), Counter()
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue  # Match the official parser's handling of non-record lines.
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                raise ValueError("ATD frame identifier is not an integer") from None
            if not 0 <= frame < n_frames:
                raise ValueError("ATD frame identifier is outside the source timeline")
            groups = parts[1].split(GROUP_SEP)
            if len(groups) != len(teams):
                raise ValueError("ATD player group count does not match metadata")
            for slot, group in enumerate(groups):
                values = _observation(group, player_fields)
                players[frame, slot] = values
                players_xy64[frame, slot] = values[:2]
            values = _observation(parts[2], ball_fields)
            ball[frame] = values
            ball_xy64[frame] = values[:2]
    print("player group fields: {}".format(dict(sorted(player_fields.items()))))
    print("ball group fields: {}".format(dict(sorted(ball_fields.items()))))
    return Tracking(players, ball, teams, pitch, players_xy64, ball_xy64)


def _write_cache(path, tracking, signature):
    """Atomic replacement keeps an interrupted parse from leaving a partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, players=tracking.players, ball=tracking.ball,
                     team_of_slot=tracking.team_of_slot, pitch=tracking.pitch,
                     players_xy64=tracking.players_xy64, ball_xy64=tracking.ball_xy64,
                     **signature)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_tracking(source="tactical", data_dir=None) -> Tracking:
    """Use the source's own timeline; broadcast is never aligned to tactical here."""
    root = Path(data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, root, source)
    signature = _signature(raw_path, meta_path)
    cache_path = root / "cache" / "{}_{}_tracking.npz".format(GAME, source)
    tracking = _read_cache(cache_path, signature)
    if tracking is None:
        tracking = _parse_tracking(raw_path, team_of_each_track(meta_path),
                                   source_frames(source), GAMES[GAME]["pitch"])
        _write_cache(cache_path, tracking, signature)
    return tracking


def load_smart_tagging(data_dir=None):
    """Keep the starter's rounded, half-open video intervals and title-case tag names."""
    root = data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    events = load_tags(data_dir=root)
    intervals = pd.DataFrame(
        [{key: event[key] for key in ("start_frame", "end_frame", "code")}
         for event in events], columns=["start_frame", "end_frame", "code"])
    intervals = intervals.astype({"start_frame": "int64", "end_frame": "int64"})
    tags = pd.DataFrame([event["tags"] for event in events])
    return pd.concat([intervals, tags], axis=1)
