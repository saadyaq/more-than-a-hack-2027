"""Synthetic Metrica examples exercise adapters and truth without any match data."""

import json

import numpy as np
import pandas as pd
import pytest

from possession import external as ext
from possession.data import Tracking
from possession.external_eval import run_method, score, truth_stats


def event(kind, start, team='Home', end=None, subtype='', period=1):
    return {'Type': kind, 'Start Frame': start, 'Team': team, 'End Frame': end,
            'Subtype': subtype, 'Period': period}


def truth(events, n=100, blocks=None):
    return ext.possession_truth(pd.DataFrame(events), n, blocks or [(1, 0, n - 1)], 'Home')


def test_ball_out_dead_until_restart_and_x_outside():
    labels = truth([event('PASS', 3), event('SET PIECE', 5),
                    event('BALL OUT', 10, end=12), event('RECOVERY', 14, 'Away'),
                    event('PASS', 17, 'Away'), event('SET PIECE', 20, 'Away')],
                   n=30, blocks=[(1, 2, 25)])
    assert ''.join(labels) == 'XX' + 'D' * 3 + 'H' * 7 + 'D' * 8 + 'A' * 6 + 'X' * 4


def test_lost_ball_keeps_last_owner_until_controlled_recovery():
    labels = truth([event('SET PIECE', 0), event('BALL LOST', 10),
                    event('CHALLENGE', 13, 'Away', subtype='TACKLE-LOST'),
                    event('RECOVERY', 20, 'Away'), event('BALL LOST', 30, 'Away'),
                    event('CARRY', 40)], n=60)
    assert ''.join(labels) == 'H' * 20 + 'A' * 20 + 'H' * 20


@pytest.mark.parametrize('delay,switched', [(50, True), (51, False), (None, False)])
def test_won_challenge_requires_owning_followup_within_two_seconds(delay, switched):
    events = [event('SET PIECE', 0), event('CHALLENGE', 10, 'Away', subtype='AERIAL-WON')]
    if delay is not None:
        events.append(event('PASS', 10 + delay, 'Away'))
    labels = truth(events)
    assert labels[10] == ('A' if switched else 'H')
    assert labels[9] == 'H'


def test_challenge_other_team_or_other_duel_does_not_confirm_control():
    labels = truth([event('SET PIECE', 0), event('CHALLENGE', 10, 'Away', subtype='TACKLE-WON'),
                    event('PASS', 11), event('CHALLENGE', 12, 'Away', subtype='AERIAL-WON')])
    assert np.all(labels == 'H')


@pytest.mark.parametrize('kind,subtype,end,dead_start', [
    ('BALL OUT', '', None, 10), ('BALL OUT', '', 13, 13),
    ('SHOT', 'HEAD-OUT', 15, 15), ('SHOT', 'ON TARGET-GOAL', 15, 15),
    ('FAULT RECEIVED', '', 15, 10),
])
def test_stoppage_rules(kind, subtype, end, dead_start):
    labels = truth([event('SET PIECE', 0), event(kind, 10, end=end, subtype=subtype),
                    event('PASS', 18, 'Away'), event('SET PIECE', 20, 'Away')], n=30)
    assert ''.join(labels) == 'H' * dead_start + 'D' * (20 - dead_start) + 'A' * 10


def test_owning_shot_and_carry_and_each_period_starts_dead():
    events = [event('SET PIECE', 0), event('CARRY', 5, 'Away'), event('SHOT', 8),
              event('CHALLENGE', 10, 'Away', subtype='TACKLE-WON'),
              event('PASS', 16, 'Away', period=2), event('SET PIECE', 18, 'Away', period=2)]
    labels = truth(events, n=25, blocks=[(1, 0, 11), (2, 15, 22)])
    assert ''.join(labels) == 'H' * 5 + 'A' * 3 + 'H' * 4 + 'X' * 3 + 'D' * 3 + 'A' * 5 + 'XX'
    assert np.all(ext.possession_truth(pd.DataFrame(), 5, [(1, 1, 3)], 'Home') == list('XDDDX'))


def observations(n=1000, slots=6):
    players = np.tile([.5, .5, 99.], (n, slots, 1))
    ball = np.tile([.5, .5, 99.], (n, 1))
    return Tracking(players, ball, np.arange(slots) % 2, ext.PITCH,
                    players[..., :2].copy(), ball[:, :2].copy())


def test_speed_metres_gaps_and_adjacent_periods():
    xy = np.zeros((8, 2))
    xy[:, 0] = np.arange(8) / (ext.FPS * ext.PITCH[0])
    speeds = ext.with_speeds(xy, blocks=[(1, 0, 3), (2, 4, 7)])[:, 2]
    np.testing.assert_allclose(speeds[[1, 2, 5, 6]], 1.)
    assert np.isnan(speeds[[0, 3, 4, 7]]).all()
    xy[2] = np.nan
    speeds = ext.with_speeds(xy)[:, 2]
    assert np.isnan(speeds[[0, 1, 2, 3, 7]]).all()
    np.testing.assert_allclose(speeds[4:7], 1.)


def test_degrade_determinism_rates_no_mutation_and_blocks():
    original = observations()
    original.players[:, -1] = np.nan  # An unused substitute must not count as observations.
    blocks = [(1, 100, 499), (2, 600, 999)]
    first = ext.degrade(original, 7, blocks=blocks)
    repeat = ext.degrade(original, 7, blocks=blocks)
    other = ext.degrade(original, 8, blocks=blocks)
    np.testing.assert_array_equal(first.players, repeat.players)
    np.testing.assert_array_equal(first.ball, repeat.ball)
    assert not np.array_equal(first.ball, other.ball, equal_nan=True)
    inside = np.zeros(1000, bool)
    inside[100:500] = inside[600:] = True
    assert np.isnan(first.ball[inside, 0]).sum() == round(.41 * inside.sum())
    assert np.isnan(first.players[inside, :-1, 0]).sum() == round(.07 * inside.sum()) * 5
    np.testing.assert_array_equal(first.ball[~inside, :2], original.ball[~inside, :2])
    np.testing.assert_array_equal(original.ball, np.tile([.5, .5, 99.], (1000, 1)))
    assert first.players_xy64 is None and first.ball_xy64 is None
    np.testing.assert_allclose(first.ball, ext.with_speeds(first.ball[:, :2], blocks=blocks),
                               rtol=1e-4, atol=1e-4, equal_nan=True)


def test_degrade_counts_existing_ball_gaps_and_preserves_excess_missing():
    original = observations()
    original.ball[:300] = np.nan
    degraded = ext.degrade(original, 0)
    assert np.isnan(degraded.ball[:, 0]).sum() == 410
    original.ball[:600] = np.nan
    degraded = ext.degrade(original, 0)
    assert np.isnan(degraded.ball[:, 0]).sum() == 600
    assert np.isnan(degraded.ball[:600]).all()


def test_degrade_prefers_crowds_and_has_bursts():
    original = observations(n=10000)
    original.players[5000:, :, 0] = .1
    degraded = ext.degrade(original, 0, ball_missing=.2, player_missing=0, jitter_m=0)
    missing = np.isnan(degraded.ball[:, 0])
    assert missing[:5000].sum() > missing[5000:].sum()
    # Temporal correlation distinguishes bursts from independent frame dropping.
    assert np.count_nonzero(missing[1:] & missing[:-1]) > .8 * missing.sum()


@pytest.mark.parametrize('rate', [0., 1.])
def test_degrade_extreme_rates_terminate(rate):
    degraded = ext.degrade(observations(20), 0, ball_missing=rate, player_missing=rate)
    assert np.isnan(degraded.ball[:, 0]).sum() == round(rate * 20)
    assert np.isnan(degraded.players[..., 0]).sum() == round(rate * 20) * 6


def test_explicit_blocks_and_empty_tags_never_query_competition(monkeypatch):
    from possession import evidence, decode
    def forbidden(*args):
        raise AssertionError('Competition block lookup')
    monkeypatch.setattr(evidence, 'half_video_blocks', forbidden)
    monkeypatch.setattr(decode, 'half_video_blocks', forbidden)
    tracking = observations(40)
    blocks = [(1, 3, 14), (2, 20, 30)]
    scores = evidence.build_evidence(tracking, pd.DataFrame(), blocks)
    labels = decode.decode(scores, blocks)
    outside = np.ones(40, bool)
    outside[3:15] = outside[20:31] = False
    assert not scores[outside].any()
    assert np.all(labels[outside] == 'X')
    assert labels[3] == labels[20] == 'D'
    for method in ('baseline', 'v1', 'v2', 'v3'):
        assert np.all(run_method(method, tracking, blocks)[outside] == 'X')


def test_overrides_are_effective_and_do_not_mutate_defaults():
    from possession.evidence import build_evidence
    from possession import params as p
    tracking = observations(100)
    tracking.players[:, 1::2, 0] = .9
    tracking.players_xy64 = None
    blocks = [(1, 0, 99)]
    default = build_evidence(tracking, pd.DataFrame(), blocks)
    disabled = build_evidence(tracking, pd.DataFrame(), blocks, params={'W_CONTROL': 0.})
    assert default[:, 0].max() == p.W_CONTROL
    assert disabled[:, 0].max() == 0.
    np.testing.assert_array_equal(default, build_evidence(tracking, pd.DataFrame(), blocks))
    labels = run_method('v1', tracking, blocks, {'W_CONTROL': 0.})
    assert np.all(labels == 'D')
    assert np.any(run_method('v1', tracking, blocks) == 'H')
    # Transition overrides reach the actual decoder too.
    assert np.all(run_method('v1', tracking, blocks, {'P_FROM_DEAD': 1e6}) == 'D')
    assert np.any(run_method('baseline', tracking, blocks, {'DEFAULT_WINDOW': 0}) == 'H')
    with pytest.raises(ValueError, match='Unknown parameters'):
        run_method('v1', tracking, blocks, {'TYPO': 1})
    with pytest.raises(ValueError, match='fake stoppage'):
        run_method('v3', tracking, blocks, {'P_SWITCH': 1000.})
    result = score(np.array(list('DDHHAA')), np.array(list('DDHHAA')))
    assert result['spell_f1_mean'] == result['macro_f1'] == 1.
    assert truth_stats(np.array(list('XDDHHAA'))) == (2, .08, 1 / 3)


def test_csv_adapter_origin_cache_and_event_normalization(tmp_path, monkeypatch):
    monkeypatch.setattr(ext, 'CACHE_DIR', tmp_path / 'cache')
    folder = tmp_path / 'Sample_Game_1'
    folder.mkdir()
    for team in ('Home', 'Away'):
        rows = 'header\nheader\nheader\n10,101,0,0.1,0.2,0.5,0.5\n10,102,0.04,0.2,0.2,0.5,0.5\n'
        (folder / ('test_RawTrackingData_{}_Team.csv'.format(team))).write_text(rows)
    pd.DataFrame([event('SET PIECE', 101, period=10)]).to_csv(folder / 'test_RawEventsData.csv', index=False)
    tracking, blocks, home = ext.load_metrica_game(1, tmp_path)
    assert tracking.players.shape == (2, 2, 3)
    assert blocks == [(10, 0, 1)] and home == 'Home'
    assert list(tracking.team_of_slot) == [0, 1]
    assert ext.load_metrica_events(1, tmp_path)['Start Frame'].tolist() == [0]
    def forbidden(*args):
        raise AssertionError('Cache missed')
    monkeypatch.setattr(ext, '_csv_tracking', forbidden)
    cached, _, _ = ext.load_metrica_game(1, tmp_path)
    np.testing.assert_array_equal(tracking.players, cached.players)


def test_epts_substitution_mapping_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ext, 'CACHE_DIR', tmp_path / 'cache')
    folder = tmp_path / 'Sample_Game_3'
    folder.mkdir()
    prefix = '''<main><Metadata><Score idLocalTeam="h"/>
    <Teams><Team id="h"><Name>Hosts</Name></Team></Teams>
    <Players><Player id="a" teamId="h"/><Player id="b" teamId="v"/>
    <Player id="c" teamId="h"/></Players>
    <GlobalConfig><ProviderGlobalParameters>
    <ProviderParameter><Name>first_half_start</Name><Value>10</Value></ProviderParameter>
    <ProviderParameter><Name>first_half_end</Name><Value>12</Value></ProviderParameter>
    </ProviderGlobalParameters></GlobalConfig><PlayerChannels>'''
    channels = ''.join('<PlayerChannel id="{0}{1}" playerId="{0}" channelId="{1}"/>'.format(player, axis)
                       for player in 'abc' for axis in 'xy')
    specs = []
    for start, end, players in [(10, 11, 'ab'), (12, 12, 'cb')]:
        groups = ''.join('<SplitRegister><PlayerChannelRef playerChannelId="{0}x"/>'
                         '<PlayerChannelRef playerChannelId="{0}y"/></SplitRegister>'.format(player)
                         for player in players)
        specs.append('<DataFormatSpecification startFrame="{}" endFrame="{}"><SplitRegister>{}'
                     '</SplitRegister></DataFormatSpecification>'.format(start, end, groups))
    (folder / 'test_metadata.xml').write_text(prefix + channels + '</PlayerChannels></Metadata>'
                                             + ''.join(specs) + '</main>')
    (folder / 'test_tracking.txt').write_text('10:0.1,0.2;0.7,0.8:0.5,0.5\n'
                                             '11:0.2,0.2;0.6,0.8:0.5,0.5\n'
                                             '12:0.3,0.2;0.5,0.8:0.5,0.5\n')
    raw = {'data': [{'team': {'name': 'Hosts'}, 'type': {'name': 'CARRY'},
                     'subtypes': [{'name': 'ONE'}, {'name': 'TWO'}],
                     'start': {'frame': 10}, 'end': {'frame': 12}, 'period': 1}]}
    (folder / 'test_events.json').write_text(json.dumps(raw))
    tracking, blocks, home = ext.load_metrica_game(3, tmp_path)
    assert blocks == [(1, 0, 2)] and home == 'Hosts'
    assert tracking.players.shape == (3, 3, 3)
    assert np.isnan(tracking.players[:2, 2]).all() and np.isnan(tracking.players[2, 0]).all()
    np.testing.assert_allclose(tracking.players[2, 2, :2], [.3, .2])
    events = ext.load_metrica_events(3, tmp_path)
    assert events['Start Frame'].tolist() == [0] and events['End Frame'].tolist() == [2]
    assert events['Subtype'].tolist() == ['ONE;TWO']
