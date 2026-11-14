"""Streaming adapters for external DFL/Sportec open matches, never competition data.

DFL open data, Bassek, Rein, Weber & Memmert, Scientific Data 2025,
doi 10.1038/s41597-025-04505-y, CC BY 4.0. Licence checked against
https://kloppy.pysport.org/reference/providers/sportec/ (2026-11-14).
Parsed external arrays live in the ignored .ext_cache/ directory.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from possession.data import Tracking
from possession.external import CACHE_DIR, with_speeds

# Params
DFL_GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
DFL_MIN_SPELL_FRAMES = 12  # 0.5 s: short touches flanked by the same team are deflections.
DFL_CACHE_VERSION = 1  # Invalidate caches when DFL timeline or truth semantics change.
PERIODS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
ATTRIBUTES = re.compile(r'(\w+)="([^"]*)"')
FRAME_NUMBER = re.compile(r'\bN="(\d+)"')


def merge_flicker(truth, min_frames=DFL_MIN_SPELL_FRAMES, blocks=None):
    """Remove only short opposing touches flanked by the same team within a half.

    Decisions use the original runs, so the result does not depend on traversal order.
    """
    result = np.array(truth, dtype='<U1', copy=True)
    if blocks is None:
        blocks = [(1, 0, len(truth) - 1)]
    for _, start, end in blocks:
        part = np.asarray(truth)[start:end + 1]
        edges = np.r_[0, np.flatnonzero(part[1:] != part[:-1]) + 1, len(part)]
        for i in range(1, len(edges) - 2):
            left, right = edges[i:i + 2]
            label, before, after = part[left], part[edges[i - 1]], part[right]
            if (label in ('H', 'A') and right - left < min_frames
                    and before == after and before in ('H', 'A') and before != label):
                result[start + left:start + right] = before
    return result


def _frames(path, teams, inventory=False):
    """Read one line at a time; skip referees before interpreting their frames."""
    context = None
    with path.open() as handle:
        for line in handle:
            stripped = line.lstrip()
            if stripped.startswith('<FrameSet '):
                attrs = dict(ATTRIBUTES.findall(line))
                team = attrs.get('TeamId')
                period = PERIODS.get(attrs.get('GameSection'))
                context = ((period, team, attrs.get('PersonId'))
                           if period and team in (*teams, 'BALL') else None)
            elif stripped.startswith('</FrameSet>'):
                context = None
            elif context is not None and stripped.startswith('<Frame '):
                if inventory:
                    yield context, int(FRAME_NUMBER.search(line).group(1))
                else:
                    yield context, dict(ATTRIBUTES.findall(line))


def _parse_game(info_path, positions_path):
    attrs = {}
    for element in ET.parse(info_path).iter():
        attrs.update({k: v for k, v in element.attrib.items()
                      if k in ('HomeTeamId', 'GuestTeamId', 'PitchX', 'PitchY')})
    teams = (attrs['HomeTeamId'], attrs['GuestTeamId'])
    pitch = (float(attrs['PitchX']), float(attrs['PitchY']))
    bounds, persons = {}, {team: set() for team in teams}
    for (period, team, person), frame in _frames(positions_path, teams, inventory=True):
        lo, hi = bounds.get(period, (frame, frame))
        bounds[period] = (min(lo, frame), max(hi, frame))
        if team in persons:
            persons[team].add(person)
    if not bounds:
        raise ValueError('DFL positions contain no recognised frames')
    slots = [(team, person) for team in teams for person in sorted(persons[team])]
    slot_index = {key: i for i, key in enumerate(slots)}
    blocks, offsets, n = [], {}, 0
    for period, (lo, hi) in sorted(bounds.items()):
        offsets[period] = n - lo
        blocks.append((period, n, n + hi - lo))
        n += hi - lo + 1
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    truth = np.full(n, 'X', dtype='<U1')
    ball_seen = np.zeros(n, dtype=bool)
    length, width = pitch
    for (period, team, person), frame in _frames(positions_path, teams):
        index = int(frame['N']) + offsets[period]
        xy = ((float(frame['X']) + length / 2) / length,
              (width / 2 - float(frame['Y'])) / width)
        if team == 'BALL':
            ball[index], ball_seen[index] = xy, True
            if frame.get('BallStatus') == '0':
                truth[index] = 'D'
            elif frame.get('BallPossession') in ('1', '2'):
                truth[index] = 'H' if frame['BallPossession'] == '1' else 'A'
        else:
            players[index, slot_index[team, person]] = xy
    print('DFL missing ball elements: {} / {} ({:.6%})'.format(
        np.count_nonzero(~ball_seen), n, np.mean(~ball_seen)))
    truth = merge_flicker(truth, blocks=blocks)
    team_of_slot = np.array([teams.index(team) for team, _ in slots], dtype=np.int8)
    tracking = Tracking(with_speeds(players, pitch=pitch, blocks=blocks),
                        with_speeds(ball, pitch=pitch, blocks=blocks), team_of_slot, pitch)
    return tracking, blocks, truth


def load_dfl_game(match_id, root=None):
    """Return Tracking, inclusive dense period blocks, and H/A/D/X possession truth."""
    if match_id not in DFL_GAMES:
        raise ValueError('Unknown DFL match id')
    root = Path(root or os.environ.get('DFL_DATA_DIR', Path.home() / 'ext_data/dfl'))
    paths = []
    for prefix in ('DFL_02_01_matchinformation', 'DFL_04_03_positions_raw_observed'):
        matches = sorted(root.glob(prefix + '_DFL-COM-*_DFL-MAT-' + match_id + '.xml'))
        if len(matches) != 1:
            raise FileNotFoundError('Expected one DFL file per match and input type')
        paths.append(matches[0])
    signature = json.dumps([(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                            for path in paths] + [('version', DFL_CACHE_VERSION)])
    key = hashlib.sha256(signature.encode()).hexdigest()
    cache_path = CACHE_DIR / ('dfl_{}_{}.npz'.format(match_id, key))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            tracking = Tracking(cached['players'], cached['ball'], cached['team_of_slot'],
                                tuple(cached['pitch']))
            blocks = [tuple(map(int, block)) for block in cached['blocks']]
            return tracking, blocks, cached['truth']
    tracking, blocks, truth = _parse_game(*paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, players=tracking.players, ball=tracking.ball,
                                team_of_slot=tracking.team_of_slot, pitch=tracking.pitch,
                                blocks=np.asarray(blocks, dtype=np.int64), truth=truth)
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return tracking, blocks, truth
