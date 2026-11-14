"""Streaming adapters for external DFL/Sportec open matches, never competition data.

DFL open data, Bassek, Rein, Weber & Memmert, Scientific Data 2025,
doi 10.1038/s41597-025-04505-y, CC BY 4.0. Licence checked against
https://kloppy.pysport.org/reference/providers/sportec/ (2026-11-14).
Parsed external arrays live in the ignored .ext_cache/ directory.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from possession.data import Tracking
from possession.external import CACHE_DIR, CHALLENGE_FOLLOWUP_S, FPS, with_speeds

# Params
DFL_GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
DFL_MIN_SPELL_FRAMES = 12  # 0.5 s: short touches flanked by the same team are deflections.
DFL_CACHE_VERSION = 2  # Invalidate caches when DFL timeline or truth semantics change.
SNAP_WINDOW_S = 3.0  # DFL event times are off by -0.4 ± 1.8 s (Bassek et al. 2025).
SNAP_MAX_DIST_M = 3.0  # A player touching the ball is within ~3 m in optical tracking.
SET_PIECES = frozenset({'KickOff', 'ThrowIn', 'FreeKick', 'CornerKick', 'GoalKick', 'Penalty'})
OWNING = SET_PIECES | {'Play', 'OtherBallAction', 'BallClaiming', 'ShotAtGoal'}
CONFIRMED = frozenset({'TacklingGame', 'OtherBallAction'})  # Own only with a same-team follow-up.
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
    alive = np.ones(n, dtype=bool)
    first_times = {}
    length, width = pitch
    for (period, team, person), frame in _frames(positions_path, teams):
        index = int(frame['N']) + offsets[period]
        xy = ((float(frame['X']) + length / 2) / length,
              (width / 2 - float(frame['Y'])) / width)
        if team == 'BALL':
            ball[index], ball_seen[index] = xy, True
            if period not in first_times and frame.get('T'):
                block_start = next(b[1] for b in blocks if b[0] == period)
                first_times[period] = _utc(frame['T']) - (index - block_start) / FPS
            alive[index] = frame.get('BallStatus') != '0'
            if frame.get('BallStatus') == '0':
                truth[index] = 'D'
            elif frame.get('BallPossession') in ('1', '2'):
                truth[index] = 'H' if frame['BallPossession'] == '1' else 'A'
        else:
            players[index, slot_index[team, person]] = xy
    print('DFL missing ball elements: {} / {} ({:.6%})'.format(
        np.count_nonzero(~ball_seen), n, np.mean(~ball_seen)))
    raw_flags = truth.copy()
    truth = merge_flicker(truth, blocks=blocks)
    team_of_slot = np.array([teams.index(team) for team, _ in slots], dtype=np.int8)
    tracking = Tracking(with_speeds(players, pitch=pitch, blocks=blocks),
                        with_speeds(ball, pitch=pitch, blocks=blocks), team_of_slot, pitch)
    metadata = dict(alive=alive, raw_flags=raw_flags,
                    first_times=np.array([first_times.get(b[0], np.nan) for b in blocks]),
                    player_ids=np.array([person for _, person in slots]), teams=np.array(teams))
    return tracking, blocks, truth, metadata


def _load_dfl(match_id, root=None):
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
            metadata = {key: cached[key] for key in
                        ('alive', 'raw_flags', 'first_times', 'player_ids', 'teams')}
            return tracking, blocks, cached['truth'], metadata
    tracking, blocks, truth, metadata = _parse_game(*paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, players=tracking.players, ball=tracking.ball,
                                team_of_slot=tracking.team_of_slot, pitch=tracking.pitch,
                                blocks=np.asarray(blocks, dtype=np.int64), truth=truth, **metadata)
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return tracking, blocks, truth, metadata


def _utc(value):
    """Offsets in event and tracking timestamps must refer to the same UTC instant."""
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('DFL timestamps must include a timezone')
    return parsed.astimezone(timezone.utc).timestamp()


def _parse_events(path, blocks, first_times, teams):
    records, dropped = [], 0
    for event in ET.parse(path).getroot():
        if not len(event):
            continue
        action = event[0]
        kind = action.tag
        if kind not in OWNING and kind != 'TacklingGame':
            continue
        time = _utc(event.attrib['EventTime'])
        frame = None
        for (_, start, end), first in zip(blocks, first_times):
            if first <= time <= first + (end - start) / FPS:
                frame = start + round((time - first) * FPS)
                break
        if frame is None:
            dropped += 1
            continue
        attrs = action.attrib
        nested = action.find('.//Play')
        team = attrs.get('WinnerTeam' if kind == 'TacklingGame' else 'Team')
        player = attrs.get('Winner' if kind == 'TacklingGame' else 'Player')
        if kind in SET_PIECES and nested is not None:
            team, player = team or nested.get('Team'), player or nested.get('Player')
        subtype = {}
        if kind == 'ShotAtGoal' and len(action):
            subtype['outcome'] = action[0].tag
        if kind in SET_PIECES:
            subtype['set_piece'] = kind
        if kind == 'TacklingGame':
            subtype = {key: attrs.get(key) for key in ('WinnerRole', 'PossessionChange')}
        records.append(dict(frame=frame, kind=kind,
                            team=list(teams).index(team) if team in teams else None,
                            player=player, subtype=subtype))
    return sorted(records, key=lambda event: event['frame']), dropped


def _events(match_id, root, blocks, metadata):
    root = Path(root or os.environ.get('DFL_DATA_DIR', Path.home() / 'ext_data/dfl'))
    paths = sorted(root.glob('DFL_03_02_events_raw_DFL-COM-*_DFL-MAT-' + match_id + '.xml'))
    if len(paths) != 1:
        raise FileNotFoundError('Expected one DFL events file per match')
    return _parse_events(paths[0], blocks, metadata['first_times'], metadata['teams'])


def load_dfl_events(match_id, root=None):
    """Return timestamp-aligned events before proximity snapping."""
    _, blocks, _, metadata = _load_dfl(match_id, root)
    return _events(match_id, root, blocks, metadata)[0]


def snap_events(events, tracking, alive, window_s=SNAP_WINDOW_S):
    """Use clean player proximity, never a neighbouring half, to align touches.

    DFL tracking carries player_ids in slot order and inclusive period blocks.
    Shifts include unchanged events; the share counts events whose frame moved.
    """
    if len(alive) != len(tracking.ball):
        raise ValueError('Alive mask and tracking lengths differ')
    ids = {person: slot for slot, person in enumerate(tracking.player_ids)}
    result, shifts = [], []
    for event in events:
        event = dict(event)
        original = event['frame']
        slot = ids.get(event['player'])
        block = next((b for b in tracking.blocks if b[1] <= original <= b[2]), None)
        if slot is not None and block is not None:
            start = max(block[1], original - round(window_s * FPS))
            stop = min(block[2], original + round(window_s * FPS)) + 1
            delta = (tracking.players[start:stop, slot, :2] - tracking.ball[start:stop, :2])
            distances = np.linalg.norm(delta * np.asarray(tracking.pitch), axis=1)
            distances[~np.isfinite(distances)] = np.inf
            best = int(np.argmin(distances))
            if distances[best] < SNAP_MAX_DIST_M:
                event['frame'] = start + best
        shifts.append(abs(event['frame'] - original) / FPS)
        result.append(event)
    stats = dict(snap_share=float(np.mean(np.asarray(shifts) > 0)) if shifts else 0.,
                 median_shift_s=float(np.median(shifts)) if shifts else 0.)
    return sorted(result, key=lambda event: event['frame']), stats


def event_truth(events, alive, blocks, n_frames, flags=None, return_stats=False):
    """Persist controlled possession inside alive runs; flags cover eventless runs.

    Dead set pieces can seed the immediately following alive run. Other dead
    events cannot transfer ownership. First alive owning events take precedence.
    """
    labels = np.full(n_frames, 'X', dtype='<U1')
    fallback = 0
    for _, start, end in blocks:
        block = sorted([e for e in events if start <= e['frame'] <= end],
                       key=lambda e: e['frame'])
        owning = []
        for index, event in enumerate(block):
            owns = event['kind'] in OWNING
            # A won duel or an unclassified touch (OtherBallAction: control, deflection or
            # clearance) owns only if its team follows up, as for Metrica's won challenges:
            # CHALLENGE.md says a deflection without control does not change possession.
            if event['kind'] in CONFIRMED:
                owns = any(e['kind'] in OWNING and e['team'] == event['team'] and
                           event['frame'] <= e['frame'] <= event['frame'] + CHALLENGE_FOLLOWUP_S * FPS
                           for e in block[index + 1:])
            if owns and event['team'] in (0, 1):
                owning.append(event)
        labels[start:end + 1] = 'D'
        mask = np.asarray(alive[start:end + 1], dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        previous_stop = start
        for left, right in zip(edges[::2], edges[1::2]):
            lower, stop = start + left, start + right
            actions = [e for e in owning if lower <= e['frame'] < stop]
            restarts = [e for e in owning if previous_stop <= e['frame'] < lower
                        and e['kind'] in SET_PIECES]
            if not actions and not restarts:
                fallback += 1
                if flags is None:
                    raise ValueError('Provider flags required for an eventless alive run')
                labels[lower:stop] = flags[lower:stop]
            else:
                first = actions[0] if actions else restarts[-1]
                labels[lower:stop] = 'H' if first['team'] == 0 else 'A'
                for event in actions:
                    labels[event['frame']:stop] = 'H' if event['team'] == 0 else 'A'
            previous_stop = stop
    stats = dict(fallback_runs=fallback)
    return (labels, stats) if return_stats else labels


def dfl_truth_details(match_id, root=None):
    """Share one clean cache load between event truth construction and diagnostics."""
    tracking, blocks, flags, metadata = _load_dfl(match_id, root)
    tracking.player_ids, tracking.blocks = metadata['player_ids'], blocks
    events, dropped = _events(match_id, root, blocks, metadata)
    events, stats = snap_events(events, tracking, metadata['alive'])
    truth, fallback = event_truth(events, metadata['alive'], blocks, len(flags),
                                  flags=metadata['raw_flags'], return_stats=True)
    stats.update(fallback, events_dropped=dropped)
    return tracking, blocks, flags, truth, metadata['alive'], stats


def load_dfl_game(match_id, root=None, truth='flags'):
    """Keep the public tuple stable while selecting provider or event possession."""
    if truth == 'events':
        tracking, blocks, _, labels, _, _ = dfl_truth_details(match_id, root)
    elif truth == 'flags':
        tracking, blocks, labels, metadata = _load_dfl(match_id, root)
        tracking.player_ids, tracking.blocks = metadata['player_ids'], blocks
    else:
        raise ValueError('Unknown DFL truth')
    return tracking, blocks, labels
