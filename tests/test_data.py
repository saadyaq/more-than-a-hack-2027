"""Synthetic files exercise parsing and invalidation without exposing match assets."""

import os

import numpy as np
import pytest

from games import GAME, asset_paths
from possession import data
from possession.baseline import nearest_team
from starter.baseline import nearest_player_labels


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    paths = asset_paths(data_dir=tmp_path)
    (tmp_path / "tactical").mkdir()
    meta = '<Root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></Root>'
    with open(paths["atd_meta"], "w") as handle:
        handle.write(meta)
    raw = ("ignored header\n"
           "3:NaN,NaN,NaN;0.1,0.2,3:NaN,NaN,NaN\n"
           "1:0.50000002,0,1;0.50000001,0,2:0,0,3,extra\n")
    with open(paths["atd_raw"], "w") as handle:
        handle.write(raw)
    monkeypatch.setattr(data, "source_frames", lambda source: 4)
    return tmp_path, paths


def test_parse_schema_index_precision_and_cache(bundle, monkeypatch, capsys):
    root, paths = bundle
    observations = data.load_tracking(data_dir=root)
    assert observations.players.shape == (4, 2, 3)
    assert observations.ball.shape == (4, 3)
    assert observations.players.dtype == observations.ball.dtype == np.float32
    assert observations.team_of_slot.dtype == np.int8
    assert observations.pitch == data.GAMES[GAME]["pitch"]
    assert np.isnan(observations.players[1]).all()
    assert np.isnan(observations.ball[2]).all()
    assert observations.players[2, 1, 2] == 3
    np.testing.assert_array_equal(nearest_team(observations), [1, -1, -1, -1])
    expected = nearest_player_labels(paths["atd_raw"], observations.team_of_slot, 4,
                                     observations.pitch)
    np.testing.assert_array_equal(nearest_team(observations), expected)
    output = capsys.readouterr().out
    assert "player group fields: {3: 4}" in output
    assert "ball group fields: {3: 1, 4: 1}" in output
    cache_path = root / "cache" / (GAME + "_tactical_tracking.npz")
    with np.load(cache_path, allow_pickle=False) as cache:
        assert int(cache["raw_size"]) == os.stat(paths["atd_raw"]).st_size
        assert int(cache["raw_mtime_ns"]) == os.stat(paths["atd_raw"]).st_mtime_ns

    def unexpected_parse(*args):
        pytest.fail("warm cache unexpectedly reparsed tracking")

    monkeypatch.setattr(data, "_parse_tracking", unexpected_parse)
    cached = data.load_tracking(data_dir=root)
    for name in ("players", "ball", "team_of_slot", "players_xy64", "ball_xy64"):
        np.testing.assert_array_equal(getattr(cached, name), getattr(observations, name))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("change", ["mtime", "size", "metadata"])
def test_cache_invalidation(bundle, monkeypatch, change):
    root, paths = bundle
    data.load_tracking(data_dir=root)
    path = paths["atd_meta"] if change == "metadata" else paths["atd_raw"]
    previous = os.stat(path)
    if change == "size":
        with open(path, "a") as handle:
            handle.write("ignored footer\n")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    else:
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    original = data._parse_tracking
    calls = []

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(data, "_parse_tracking", counted)
    data.load_tracking(data_dir=root)
    assert len(calls) == 1


def test_broadcast_source_has_separate_cache(bundle, monkeypatch):
    root, paths = bundle
    (root / "broadcast").mkdir()
    broadcast = asset_paths(data_dir=root, source="broadcast")
    for key in ("atd_meta", "atd_raw"):
        with open(paths[key]) as source, open(broadcast[key], "w") as target:
            target.write(source.read())
    monkeypatch.setattr(data, "source_frames", lambda source: 5 if source == "broadcast" else 4)
    observations = data.load_tracking(source="broadcast", data_dir=root)
    assert observations.ball.shape == (5, 3)
    assert (root / "cache" / (GAME + "_broadcast_tracking.npz")).exists()


def test_smart_tagging_video_frames_and_tags(bundle):
    root, paths = bundle
    with open(paths["smart_tagging"], "w") as handle:
        handle.write('<Root><instance><start>0.1</start><end>0.3</end><code>PHASE</code>'
                     '<label><group>Team</group><text>Home</text></label>'
                     '<label><group>Type</group><text>Restart</text></label>'
                     '</instance></Root>')
    events = data.load_smart_tagging(data_dir=root)
    assert events["start_frame"].dtype == events["end_frame"].dtype == np.int64
    assert events.loc[0, "start_frame"] == 2
    assert events.loc[0, "end_frame"] == 8
    assert events.loc[0, "code"] == "PHASE"
    assert events.loc[0, "Team"] == "Home"
    assert events.loc[0, "Type"] == "Restart"


def test_empty_smart_tagging(bundle):
    root, paths = bundle
    with open(paths["smart_tagging"], "w") as handle:
        handle.write("<Root/>")
    events = data.load_smart_tagging(data_dir=root)
    assert events.empty
    assert events["start_frame"].dtype == events["end_frame"].dtype == np.int64
