"""Metrica adapters and deterministic noise for external-only parameter experiments.

Source: Metrica Sports, https://github.com/metrica-sports/sample-data (checked
2026-11-07). Its README offers samples to the football analytics community and
requires acknowledgement for public use. There is no separate LICENSE file or
standard SPDX licence; this harness uses that stated permission, credits Metrica
Sports, and does not redistribute their raw data. These are EXTERNAL games, never
competition data. Parsed arrays live only in the repo's ignored .ext_cache/.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from possession.data import Tracking

# Params
FPS = 25  # Metrica samples share the competition's 25 Hz football timeline.
PITCH = (105.0, 68.0)  # Metrica documents a 105 by 68 metre playing surface.
CHALLENGE_FOLLOWUP_S = 2.0  # A won duel needs prompt controlled possession.
BALL_MISSING = 0.41  # Public DATA.md reports 41 percent ball absence during play.
PLAYER_MISSING = 0.07  # Requested observation loss, excluding inactive substitute slots.
BURST_MEAN_S = 1.5  # Occlusion persists across neighbouring football observations.
JITTER_M = 0.15  # Requested positional uncertainty in metres on each axis.
CROWD_RADIUS_M = 3.0  # Nearby bodies can hide the ball from tracking cameras.
CROWD_PLAYERS = 3  # Three nearby players define the requested crowded situation.
CROWD_WEIGHT = 4.0  # Untuned assumption: crowded frames get fourfold burst-start weight.
CACHE_VERSION = 1  # Invalidate parsing caches when timeline or slot semantics change.
CACHE_DIR = Path(__file__).resolve().parents[1] / '.ext_cache'
OWNING = frozenset({'PASS', 'CARRY', 'RECOVERY', 'SET PIECE', 'SHOT'})
PERIOD_NAMES = ('first_half', 'second_half', 'first_extra_half', 'second_extra_half')


def _folder(game_id, root):
    game_id = int(game_id)
    if game_id not in (1, 2, 3):
        raise ValueError('Metrica game_id must be 1, 2 or 3')
    root = Path(root or os.environ.get('EXT_DATA_DIR', Path.home() / 'ext_data/metrica'))
    return root / ('Sample_Game_{}'.format(game_id))


def _bounds(n, blocks):
    """Use inclusive block endpoints and keep adjacent periods distinct."""
    if blocks is None:
        blocks = [(1, 0, n - 1)]
    return [(period, max(0, start), min(n, end + 1)) for period, start, end in blocks
            if max(0, start) < min(n, end + 1)]


def with_speeds(xy, pitch=PITCH, blocks=None, fps=FPS):
    """Centred m/s differences require three finite observations in the same period.

    Endpoints, isolated samples and neighbours of gaps retain NaN speed; positions
    remain usable. No displacement bridges halftime, substitutions or missing data.
    """
    xy = np.asarray(xy, dtype=np.float64)
    result = np.full(xy.shape[:-1] + (3,), np.nan, dtype=np.float32)
    result[..., :2] = xy
    for _, start, stop in _bounds(len(xy), blocks):
        part = xy[start:stop]
        if len(part) < 3:
            continue
        observed = np.isfinite(part).all(axis=-1)
        valid = observed[:-2] & observed[1:-1] & observed[2:]
        speed = np.linalg.norm((part[2:] - part[:-2]) * pitch, axis=-1) * fps / 2
        result[start + 1:stop - 1, ..., 2] = np.where(valid, speed, np.nan)
    return result


def _csv_tracking(folder):
    frames = [pd.read_csv(next(folder.glob('*_RawTrackingData_{}_Team.csv'.format(team))),
                          skiprows=3, header=None) for team in ('Home', 'Away')]
    home, away = (frame.to_numpy(dtype=float) for frame in frames)
    if not np.array_equal(home[:, :3], away[:, :3]):
        raise ValueError('Metrica team timelines disagree')
    ids = home[:, 1].astype(np.int64)
    if np.any(np.diff(ids) <= 0):
        raise ValueError('Tracking frame ids must increase')
    origin, n = int(ids[0]), int(ids[-1] - ids[0] + 1)
    slots = [(frame.shape[1] - 5) // 2 for frame in (home, away)]
    players = np.full((n, sum(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    players[ids - origin] = np.concatenate(
        [frame[:, 3:-2].reshape(len(ids), count, 2)
         for frame, count in zip((home, away), slots)], axis=1)
    if not np.array_equal(home[:, -2:], away[:, -2:], equal_nan=True):
        raise ValueError('Metrica team ball coordinates disagree')
    ball[ids - origin] = home[:, -2:]
    blocks = [(int(period), int(ids[home[:, 0] == period][0] - origin),
               int(ids[home[:, 0] == period][-1] - origin))
              for period in np.unique(home[:, 0]) if period > 0]
    teams = np.repeat(np.array([0, 1], dtype=np.int8), slots)
    return players, ball, teams, blocks, 'Home', origin


def _epts_tracking(folder):
    """Resolve each format interval's channel refs, including substitution changes."""
    meta = ET.parse(next(folder.glob('*_metadata.xml'))).getroot()
    home_id = meta.find('.//Score').attrib['idLocalTeam']
    home = next(team.findtext('Name') for team in meta.findall('.//Teams/Team')
                if team.attrib['id'] == home_id)
    roster = meta.findall('.//Players/Player')
    slot = {player.attrib['id']: i for i, player in enumerate(roster)}
    teams = np.array([int(player.attrib['teamId'] != home_id) for player in roster], np.int8)
    channels = {ch.attrib['id']: ch.attrib for ch in meta.findall('.//PlayerChannel')}
    formats = []
    for spec in meta.findall('.//DataFormatSpecification'):
        refs = spec.findall('./SplitRegister/SplitRegister')
        indices = []
        for group in refs:
            pair = [channels[ref.attrib['playerChannelId']]
                    for ref in group.findall('PlayerChannelRef')]
            if (len(pair) != 2 or pair[0]['channelId'] != 'x'
                    or pair[1]['channelId'] != 'y' or pair[0]['playerId'] != pair[1]['playerId']):
                raise ValueError('Unsupported EPTS coordinate channels')
            indices.append(slot[pair[0]['playerId']])
        formats.append((int(spec.attrib['startFrame']), int(spec.attrib['endFrame']), indices))
    formats.sort()
    values = {item.findtext('Name'): item.findtext('Value')
              for item in meta.findall('.//GlobalConfig/ProviderGlobalParameters/ProviderParameter')}
    raw_blocks = [(i + 1, int(values[name + '_start']), int(values[name + '_end']))
                  for i, name in enumerate(PERIOD_NAMES) if values.get(name + '_start')]
    raw_path = next(folder.glob('*_tracking.txt'))
    with raw_path.open() as handle:
        origin = int(handle.readline().split(':', 1)[0])
    n = max(end for _, end, _ in formats) - origin + 1
    players, ball = np.full((n, len(roster), 2), np.nan), np.full((n, 2), np.nan)
    format_index, previous = 0, origin - 1
    with raw_path.open() as handle:
        for line in handle:
            raw_frame, player_text, ball_text = line.strip().split(':')
            frame = int(raw_frame)
            if frame <= previous:
                raise ValueError('Tracking frame ids must increase')
            previous = frame
            while format_index < len(formats) and frame > formats[format_index][1]:
                format_index += 1
            if format_index == len(formats) or frame < formats[format_index][0]:
                raise ValueError('Tracking frame outside EPTS format intervals')
            indices = formats[format_index][2]
            xy = np.fromstring(player_text.replace(';', ','), sep=',')
            if xy.size != 2 * len(indices):
                raise ValueError('EPTS observation count differs from channel metadata')
            players[frame - origin, indices] = xy.reshape(-1, 2)
            ball[frame - origin] = np.fromstring(ball_text, sep=',')
    # Index zero is the first tracking frame; do not retain a metadata-only tail.
    n = previous - origin + 1
    blocks = [(p, max(0, a - origin), min(n - 1, b - origin)) for p, a, b in raw_blocks
              if b >= origin and a <= previous]
    return players[:n], ball[:n], teams, blocks, home, origin


def _load_game(game_id, root=None):
    folder = _folder(game_id, root)
    paths = sorted(folder.glob('*_RawTrackingData_*.csv') if int(game_id) != 3
                   else [*folder.glob('*_tracking.txt'), *folder.glob('*_metadata.xml')])
    if not paths:
        raise FileNotFoundError('Metrica tracking files not found in {}'.format(folder))
    signature = json.dumps([(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                            for path in paths] + [('version', CACHE_VERSION)])
    key = hashlib.sha256(signature.encode()).hexdigest()
    cache_path = CACHE_DIR / ('game_{}_{}.npz'.format(game_id, key))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            tracking = Tracking(cached['players'], cached['ball'], cached['teams'], PITCH)
            blocks = [tuple(map(int, block)) for block in cached['blocks']]
            return tracking, blocks, str(cached['home']), int(cached['origin'])
    players, ball, teams, blocks, home, origin = (
        _epts_tracking(folder) if int(game_id) == 3 else _csv_tracking(folder))
    tracking = Tracking(with_speeds(players, blocks=blocks), with_speeds(ball, blocks=blocks),
                        teams, PITCH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, players=tracking.players, ball=tracking.ball, teams=teams,
                     blocks=np.asarray(blocks, dtype=np.int64), home=home, origin=origin)
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return tracking, blocks, home, origin


def load_metrica_game(game_id, root=None):
    """Return Tracking, inclusive zero-based period blocks, and the home event name."""
    tracking, blocks, home, _ = _load_game(game_id, root)
    return tracking, blocks, home


def _subtypes(value):
    if isinstance(value, dict):
        return str(value.get('name', ''))
    if isinstance(value, list):
        return ';'.join(_subtypes(item) for item in value)
    return '' if value is None or pd.isna(value) else str(value)


def load_metrica_events(game_id, root=None):
    """Normalize CSV/JSON into CSV-named columns on the tracking's zero-based clock."""
    folder = _folder(game_id, root)
    # The first record provides the common offset without loading tracking twice.
    path = next(folder.glob('*_tracking.txt' if int(game_id) == 3
                            else '*_RawTrackingData_Home_Team.csv'))
    with path.open() as handle:
        if int(game_id) == 3:
            origin = int(handle.readline().split(':', 1)[0])
        else:
            for _ in range(3):
                next(handle)
            origin = int(next(handle).split(',')[1])
    if int(game_id) == 3:
        with next(folder.glob('*_events.json')).open() as handle:
            raw = json.load(handle)['data']
        events = pd.DataFrame([{'Team': event['team']['name'], 'Type': event['type']['name'],
                               'Subtype': _subtypes(event.get('subtypes')),
                               'Period': event['period'], 'Start Frame': event['start']['frame'],
                               'End Frame': (event.get('end') or {}).get('frame')}
                              for event in raw])
    else:
        events = pd.read_csv(next(folder.glob('*_RawEventsData.csv')))
    for column in ('Start Frame', 'End Frame'):
        events[column] = pd.to_numeric(events[column], errors='coerce') - origin
    return events


def possession_truth(events, n_frames, blocks, home):
    """Derive last controlled possession while the ball is in play (CHALLENGE.md).

    Events use CSV column names and ZERO-BASED tracking frames, as returned by
    load_metrica_events. PASS, CARRY, RECOVERY, SET PIECE and SHOT own from their
    start. A CHALLENGE subtype ending in '-WON' owns only if a subsequent owning
    event of that team starts within two seconds in the same block; other duels
    do not transfer control. BALL LOST alone never changes the owner.

    BALL OUT starts D at its end (or start if absent). SHOT with OUT or GOAL in
    its subtype starts D at its end; FAULT RECEIVED starts D at its start.
    D persists until the next SET PIECE starts, even across ordinary owning
    events. Every block starts D until its first SET PIECE; outside blocks is X.
    Endpoints are inclusive. At equal frames, stoppages precede owning events
    and restarts, so a simultaneous SET PIECE restarts play. Stable event order
    resolves simultaneous owning events. Missing/invalid end frames fall back
    to start. Challenge confirmation uses base owning events, not other duels.
    """
    records = events.to_dict('records') if isinstance(events, pd.DataFrame) else list(events)
    labels = np.full(n_frames, 'X', dtype='<U1')
    for period, lower, stop in _bounds(n_frames, blocks):
        block = [event for event in records if int(event['Period']) == period
                 and lower <= event['Start Frame'] < stop]
        block.sort(key=lambda event: event['Start Frame'])
        actions = []
        for index, event in enumerate(block):
            start, kind = int(event['Start Frame']), str(event['Type']).strip().upper()
            subtype = _subtypes(event.get('Subtype')).upper()
            owns = kind in OWNING
            if kind == 'CHALLENGE' and any(s.strip().endswith('-WON') for s in subtype.split(';')):
                owns = any(next_event['Team'] == event['Team']
                           and str(next_event['Type']).strip().upper() in OWNING
                           for next_event in block[index + 1:]
                           if next_event['Start Frame'] <= start + CHALLENGE_FOLLOWUP_S * FPS)
            end = event.get('End Frame')
            end = start if end is None or pd.isna(end) or end < start else int(end)
            dead = (kind == 'BALL OUT' or kind == 'FAULT RECEIVED'
                    or (kind == 'SHOT' and ('OUT' in subtype or 'GOAL' in subtype)))
            if dead:
                frame = start if kind == 'FAULT RECEIVED' else end
                if frame < stop:
                    actions.append((frame, 0, index, 'D', False))
            if owns:
                actions.append((start, 1, index, 'H' if event['Team'] == home else 'A',
                                kind == 'SET PIECE'))
        state, cursor = 'D', lower
        for frame, _, _, owner, restart in sorted(actions):
            labels[cursor:frame] = state
            if owner == 'D' or restart or state != 'D':
                state = owner
            cursor = frame
        labels[cursor:stop] = state
    return labels


def _burst_mask(valid, count, rng, mean_frames, ends, weights=None):
    """Weighted starts without replacement guarantee termination and exact removal."""
    removed = np.zeros(len(valid), bool)
    candidates = np.flatnonzero(valid)
    if count <= 0 or not len(candidates):
        return removed
    # An exponential race samples starts proportional to crowd weight.
    priorities = rng.exponential(size=len(candidates))
    if weights is not None:
        priorities /= weights[candidates]
    remaining = min(count, len(candidates))
    for start in candidates[np.argsort(priorities, kind='stable')]:
        if removed[start]:
            continue
        length = int(rng.geometric(1. / mean_frames))
        stop = min(int(ends[start]), int(start + length))
        frames = np.flatnonzero(valid[start:stop] & ~removed[start:stop]) + start
        frames = frames[:remaining]
        removed[frames] = True
        remaining -= len(frames)
        if not remaining:
            break
    return removed


def degrade(tracking, seed, ball_missing=BALL_MISSING, *, blocks=None,
            player_missing=PLAYER_MISSING, burst_mean_s=BURST_MEAN_S,
            jitter_m=JITTER_M, crowd_radius_m=CROWD_RADIUS_M,
            crowd_players=CROWD_PLAYERS, crowd_weight=CROWD_WEIGHT, fps=FPS):
    """Simulate occlusions without changing inputs; no parameters are fitted here.

    Ball absence reaches round(ball_missing * in-block frames), counting existing
    gaps; existing excess absence is retained. Player drops remove 7% of originally
    observed in-block player-frames per slot (inactive substitutes are excluded).
    Both use geometric bursts with mean 1.5 s, clipped at period boundaries and
    at the final quota. Crowded starts get fourfold weight, an explicit untuned
    assumption. Player burst length shares the ball default because none was
    specified. Independent 0.15 m Gaussian noise is added on both position axes.
    Speeds are recomputed; xy64 is cleared so methods cannot see pristine data.
    Omitted blocks mean the whole input; external_eval always supplies periods.
    """
    if not 0 <= ball_missing <= 1 or not 0 <= player_missing <= 1:
        raise ValueError('Missing fractions must be between zero and one')
    if (fps <= 0 or burst_mean_s * fps < 1 or jitter_m < 0
            or crowd_weight <= 0 or crowd_radius_m < 0 or crowd_players < 1):
        raise ValueError('Invalid degradation parameters')
    rng = np.random.default_rng(seed)
    players = np.array(tracking.players[..., :2], dtype=np.float64, copy=True)
    ball = np.array(tracking.ball[..., :2], dtype=np.float64, copy=True)
    n = len(ball)
    inside, ends = np.zeros(n, bool), np.zeros(n, np.int64)
    for _, start, stop in _bounds(n, blocks):
        inside[start:stop], ends[start:stop] = True, stop
    observed = np.isfinite(ball).all(axis=1)
    distance = np.linalg.norm((players - ball[:, None]) * tracking.pitch, axis=2)
    crowded = (distance <= crowd_radius_m).sum(axis=1) >= crowd_players
    weights = np.where(crowded, crowd_weight, 1.)
    count = round(ball_missing * inside.sum()) - np.count_nonzero(inside & ~observed)
    missing = _burst_mask(inside & observed, count, rng, burst_mean_s * fps, ends, weights)
    ball[missing] = np.nan
    for slot in range(players.shape[1]):
        valid = inside & np.isfinite(players[:, slot]).all(axis=1)
        missing = _burst_mask(valid, round(player_missing * valid.sum()), rng,
                              burst_mean_s * fps, ends)
        players[missing, slot] = np.nan
    players[inside] += rng.normal(0., jitter_m, players[inside].shape) / tracking.pitch
    ball[inside] += rng.normal(0., jitter_m, ball[inside].shape) / tracking.pitch
    return Tracking(with_speeds(players, tracking.pitch, blocks, fps),
                    with_speeds(ball, tracking.pitch, blocks, fps),
                    tracking.team_of_slot.copy(), tracking.pitch)
