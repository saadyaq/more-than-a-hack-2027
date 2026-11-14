"""Synthetic DFL files verify streaming, truth and multi-source training without match data."""

import numpy as np
import pytest

from possession import dfl
from possession.data import Tracking
from possession.external import fill_dead_ball, with_speeds


@pytest.fixture
def game(tmp_path, monkeypatch):
    monkeypatch.setattr(dfl, 'CACHE_DIR', tmp_path / 'cache')
    suffix = '_DFL-COM-test_DFL-MAT-J03WOH.xml'
    (tmp_path / ('DFL_02_01_matchinformation' + suffix)).write_text(
        '<Root><Match HomeTeamId="home" GuestTeamId="guest" PitchX="100" PitchY="64"/></Root>')
    lines = ['<Root>']
    for half, offset in [('firstHalf', 10000), ('secondHalf', 100000)]:
        for team, person in [('home', 'h2'), ('guest', 'a2'), ('home', 'h1'),
                             ('guest', 'a1'), ('home', 'h3'), ('referee', 'r'), ('BALL', 'b')]:
            if person == 'h3' and half == 'firstHalf':
                continue
            lines.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">'.format(
                half, team, person))
            for i in range(60):
                if (person == 'h2' and half == 'secondHalf') or (team == 'BALL' and i == 59):
                    continue
                # First half: short flanked A, dead ball, then a long A spell.
                owner = 2 if 15 <= i < 20 or i >= 40 else 1
                status = 0 if 30 <= i < 35 else 1
                lines.append('<Frame N="{}" X="{}" Y="16" S="999" '
                             'BallPossession="{}" BallStatus="{}"/>'.format(
                                 offset + i, -25 + i * .1, owner, status))
            lines.append('</FrameSet>')
    lines.append('</Root>')
    path = tmp_path / ('DFL_04_03_positions_raw_observed' + suffix)
    path.write_text('\n'.join(lines))
    return tmp_path, path


def test_parse_and_cache(game, monkeypatch):
    root, _ = game
    tracking, blocks, truth = dfl.load_dfl_game('J03WOH', root)
    assert tracking.players.shape == (120, 5, 3)
    assert tracking.ball.shape == (120, 3)
    assert tracking.players.dtype == np.float32
    assert tracking.ball.dtype == np.float32
    assert truth.dtype == np.dtype('<U1')
    assert tracking.team_of_slot.dtype == np.int8
    np.testing.assert_array_equal(tracking.team_of_slot, [0, 0, 0, 1, 1])
    assert tracking.pitch == (100., 64.)
    assert blocks == [(1, 0, 59), (2, 60, 119)]
    np.testing.assert_allclose(tracking.players[0, 0, :2], [.25, .25])
    assert np.isnan(tracking.players[:60, 2]).all()  # h3 only plays half two.
    assert np.isnan(tracking.players[60:, 1]).all()  # h2 only plays half one.
    assert np.isfinite(tracking.players[60:, 0, :2]).all()
    assert np.isfinite(tracking.players[60:, 2, :2]).all()
    assert np.isnan(tracking.players[[0, 59, 60, 119], :, 2]).all()
    assert tracking.players[1, 0, 2] == pytest.approx(2.5, abs=1e-4)
    assert (truth[15:20] == 'H').all()
    assert (truth[30:35] == 'D').all()
    assert (truth[40:59] == 'A').all()
    assert truth[59] == truth[119] == 'X'
    assert np.isnan(tracking.ball[59]).all()
    assert len(list((root / 'cache').glob('dfl_J03WOH_*.npz'))) == 1

    def unexpected(*args):
        pytest.fail('Cache round-trip reparsed positions')
    monkeypatch.setattr(dfl, '_parse_game', unexpected)
    cached, cached_blocks, cached_truth = dfl.load_dfl_game('J03WOH', root)
    assert cached_blocks == blocks
    assert cached.pitch == tracking.pitch
    for a, b in [(cached.players, tracking.players), (cached.ball, tracking.ball),
                 (cached.team_of_slot, tracking.team_of_slot), (cached_truth, truth)]:
        np.testing.assert_array_equal(a, b)


def test_cache_invalidates(game):
    root, path = game
    dfl.load_dfl_game('J03WOH', root)
    path.write_text(path.read_text() + '\n')
    dfl.load_dfl_game('J03WOH', root)
    assert len(list((root / 'cache').glob('*.npz'))) == 2


def test_merge_flicker():
    truth = np.array(list('HHHAHHHDAADHHH' + 'A' * 12 + 'HHH'))
    merged = dfl.merge_flicker(truth)
    assert merged[3] == 'H'
    np.testing.assert_array_equal(merged[7:], truth[7:])
    np.testing.assert_array_equal(truth[:7], list('HHHAHHH'))
    # A short run cannot be flanked through halftime.
    truth = np.array(list('HHAAHH'))
    np.testing.assert_array_equal(dfl.merge_flicker(truth, blocks=[(1, 0, 2), (2, 3, 5)]), truth)


def test_fill_dead_ball_pitch_and_blocks():
    xy = np.array([[0., .5], [.01, .5], [np.nan, np.nan], [.03, .5], [.04, .5], [.5, .5]])
    players = np.full((6, 2, 3), np.nan, np.float32)
    blocks = [(1, 0, 4), (2, 5, 5)]
    truth = np.array(list('HHDHHH'))
    small = Tracking(players, with_speeds(xy), np.array([0, 1], np.int8), (100., 64.))
    large = Tracking(players, with_speeds(xy), small.team_of_slot, (200., 64.))
    filled = fill_dead_ball(small, truth, blocks=blocks)
    other = fill_dead_ball(large, truth, blocks=blocks)
    assert filled.ball[2, 2] == pytest.approx(25.)
    assert other.ball[2, 2] == pytest.approx(50.)
    assert np.isnan(filled.ball[[0, 4, 5], 2]).all()


def test_load_dfl_cases_v3(monkeypatch):
    from possession import tune
    tracking = object()
    blocks, truth = [(1, 0, 2)], np.array(list('HAD'))
    filled = object()
    monkeypatch.setattr(dfl, 'load_dfl_game', lambda game: (tracking, blocks, truth))
    def fill(source, labels, blocks=None):
        assert source is tracking and labels is truth and blocks == [(1, 0, 2)]
        return filled
    def degrade(source, seed, **kwargs):
        assert source is filled and kwargs == dict(blocks=blocks, profile='v2')
        return seed
    monkeypatch.setattr(tune, 'fill_dead_ball', fill)
    monkeypatch.setattr(tune, 'degrade', degrade)
    cases = tune.load_dfl_cases(['J03WOH'], [3, 4], 'v3')
    assert [(case[0], case[1], case[2]) for case in cases] == [('dfl:J03WOH', 3, 3), ('dfl:J03WOH', 4, 4)]


@pytest.mark.parametrize('use_dfl,dfl_oof,override_seeds', [
    (False, False, False), (True, False, False), (True, True, False), (True, False, True)])
def test_training_sources_and_folds(monkeypatch, use_dfl, dfl_oof, override_seeds):
    from possession import train_learned as train
    fits, predictions, scored, saved = [], [], [], []
    def cases(keys, seeds):
        return [(key, seed, key, [(1, 0, 2)], np.array(list('HAD')))
                for key in keys for seed in seeds]
    def load_metrica(games, seeds, profile):
        assert seeds == (0, 1) and profile == 'v3'
        return cases(games, seeds)
    def load_dfl(games, seeds, profile):
        assert use_dfl and seeds == ((7,) if override_seeds else (0, 1)) and profile == 'v3'
        return cases(['dfl:' + game for game in games], seeds)
    def rows(key, blocks, truth):
        code = key if isinstance(key, int) else 10 + train.DFL_GAMES.index(key[4:])
        labels = np.array([0, 1, 2] if isinstance(key, int) else [2, 2, 2])
        return np.full((3, 1), code), labels
    def fit(x, y):
        members = set(x[:, 0])
        fits.append(members)
        return members
    def predict(model, key, blocks):
        code = key if isinstance(key, int) else 10 + train.DFL_GAMES.index(key[4:])
        assert code not in model
        predictions.append(key)
        return np.full((3, 3), code)
    def emissions(probs, prior, temperature):
        np.testing.assert_allclose(prior, [1/3, 1/3, 1/3])
        return probs
    def score(truth, labels):
        scored.append(labels[0, 0])
        return dict(spell_f1_mean=.5, macro_f1=.6, n_pred_spells=1., n_gt_spells=1.)
    monkeypatch.setattr(train, 'load_cases', load_metrica)
    monkeypatch.setattr(train, 'load_dfl_cases', load_dfl)
    monkeypatch.setattr(train, 'training_rows', rows)
    monkeypatch.setattr(train, 'fit', fit)
    monkeypatch.setattr(train, 'predict_proba', predict)
    monkeypatch.setattr(train, 'emissions', emissions)
    monkeypatch.setattr(train, 'decode', lambda evidence, **kw: evidence)
    monkeypatch.setattr(train, 'score', score)
    monkeypatch.setattr(train, 'save', lambda model, config, version: saved.append((model, config, version)))
    monkeypatch.setattr(train, 'GRID', {})
    argv = ['--profile', 'v3', '--seeds', '0', '1']
    if use_dfl:
        argv += ['--dfl', '--version', 'v11']
    if dfl_oof:
        argv += ['--dfl-oof']
    if override_seeds:
        argv += ['--dfl-seeds', '7']
    train.main(argv)
    expected = {1, 2, 3} | (set(range(10, 17)) if use_dfl else set())
    assert fits[-1] == expected
    for game in (1, 2, 3):
        assert fits[game - 1] == expected - {game}
    assert len(fits) == 4 + (7 if dfl_oof else 0)
    assert len(predictions) == 6 + (14 if dfl_oof else 0)
    tuning_scores = scored[:-14] if dfl_oof else scored
    assert all(value < 10 for value in tuning_scores)
    config = saved[0][1]
    np.testing.assert_allclose(config['prior'], [1/3, 1/3, 1/3])
    if use_dfl:
        assert saved[0][2] == 'v11'
        assert config['dfl_games'] == list(train.DFL_GAMES)
        assert config['dfl_seeds'] == ([7] if override_seeds else [0, 1])
        assert config['tuning_cases'] == 'metrica out-of-fold'
    else:
        assert saved[0][2] == 'v7' and 'dfl_games' not in config


def action(frame, kind='Play', team=0, player='h1'):
    return dict(frame=frame, kind=kind, team=team, player=player, subtype={})


def test_event_parsing_and_timezones(game):
    root, positions = game
    text = positions.read_text().replace(' S="999"', ' T="2022-08-26T16:00:00+00:00" S="999"')
    positions.write_text(text)
    path = root / 'DFL_03_02_events_raw_DFL-COM-test_DFL-MAT-J03WOH.xml'
    path.write_text('''<PutDataRequest>
      <Event EventTime="2022-08-26T18:00:00.400+02:00"><Play Team="home" Player="h1"><Pass/></Play></Event>
      <Event EventTime="2022-08-26T18:00:00.800+02:00"><ThrowIn><Play Team="guest" Player="a1"/></ThrowIn></Event>
      <Event EventTime="2022-08-26T18:00:01.000+02:00"><TacklingGame WinnerTeam="guest" Winner="a2" WinnerRole="Defender" PossessionChange="true"/></Event>
      <Event EventTime="2022-08-26T18:00:01.200+02:00"><ShotAtGoal Team="home" Player="h2"><SavedShot/></ShotAtGoal></Event>
      <Event EventTime="2022-08-26T18:00:05+02:00"><BallClaiming Team="home"/></Event>
      <Event><Delete/></Event><Event><Foul/></Event>
    </PutDataRequest>''')
    records = dfl.load_dfl_events('J03WOH', root)
    assert [e['frame'] for e in records] == [10, 20, 25, 30]
    assert [e['team'] for e in records] == [0, 1, 1, 0]
    assert [e['player'] for e in records] == ['h1', 'a1', 'a2', 'h2']
    assert records[1]['subtype'] == {'set_piece': 'ThrowIn'}
    assert records[2]['subtype'] == {'WinnerRole': 'Defender', 'PossessionChange': 'true'}
    assert records[3]['subtype'] == {'outcome': 'SavedShot'}
    clean, blocks, _, event_labels, alive, stats = dfl.dfl_truth_details('J03WOH', root)
    assert stats['events_dropped'] == 1
    assert clean.player_ids.tolist() == ['h1', 'h2', 'h3', 'a1', 'a2']
    np.testing.assert_array_equal(event_labels == 'D', ~alive)
    assert blocks == [(1, 0, 59), (2, 60, 119)]
    # Cache retains alignment metadata, alive and raw flags as well as default truth.
    np.testing.assert_array_equal(dfl.load_dfl_game('J03WOH', root, truth='events')[2], event_labels)


def test_snap_events():
    tracking = Tracking(np.full((12, 1, 3), .2), np.zeros((12, 3)), np.array([0]), (100., 64.))
    tracking.player_ids = ['h1']
    tracking.blocks = [(1, 0, 5), (2, 6, 11)]
    tracking.players[4, 0, :2] = .01
    tracking.players[6, 0, :2] = 0
    events = [action(1), action(8, player='unknown')]
    snapped, stats = dfl.snap_events(events, tracking, np.ones(12, bool))
    assert [e['frame'] for e in snapped] == [4, 8]
    assert events[0]['frame'] == 1
    assert stats == {'snap_share': .5, 'median_shift_s': .06}
    tracking.players[4, 0, :2] = .2
    assert dfl.snap_events([action(1)], tracking, np.ones(12, bool))[0][0]['frame'] == 1
    # A finite nearby touch outside the requested time window cannot attract the event.
    tracking.players[4, 0, :2] = 0
    assert dfl.snap_events([action(0)], tracking, np.ones(12, bool), window_s=.04)[0][0]['frame'] == 0


def test_event_truth_restart_persistence_fallback():
    alive = np.ones(32, bool)
    alive[8:11] = False
    blocks = [(1, 1, 24), (2, 27, 30)]
    events = [action(3), action(5, team=1), action(9, team=0),
              action(13, team=1), action(16, 'TacklingGame', team=0),
              action(18, team=0), action(20, 'TacklingGame', team=1)]
    flags = np.full(32, 'H')
    flags[29:] = 'A'
    truth, stats = dfl.event_truth(events, alive, blocks, 32, flags=flags, return_stats=True)
    assert ''.join(truth[:8]) == 'XHHHHAAA'
    assert ''.join(truth[8:16]) == 'DDDAAAAA'
    assert (truth[16:25] == 'H').all()  # confirmed tackle, then persistence; unconfirmed ignored.
    assert ''.join(truth[25:]) == 'XXHHAAX'
    assert stats['fallback_runs'] == 1
    assert np.all(truth[~alive] == 'D')


def test_tackle_followup_limit_and_blocks():
    events = [action(0), action(5, 'TacklingGame', team=1), action(56, team=1),
              action(59, 'TacklingGame', team=0), action(61)]
    truth = dfl.event_truth(events, np.ones(70, bool), [(1, 0, 59), (2, 60, 69)], 70)
    assert (truth[:56] == 'H').all()  # 51 frames exceeds the 2-second follow-up.
    assert (truth[56:60] == 'A').all()  # confirmation cannot cross the block boundary.
    assert (truth[60:] == 'H').all()


def test_dead_set_piece_restart():
    alive = np.array([False, False, True, True, False, True, True])
    events = [action(1, 'KickOff', 1), action(4, 'Play', 0)]
    flags = np.full(7, 'H')
    truth, stats = dfl.event_truth(events, alive, [(1, 0, 6)], 7, flags=flags, return_stats=True)
    assert ''.join(truth) == 'DDAADHH'
    assert stats['fallback_runs'] == 1


def test_all_game_tuning_with_distinct_score_truth(monkeypatch, capsys):
    from possession import train_learned as train
    saved, truths_scored, prediction_sources = [], [], []
    def cases(games, seeds, profile, truth='flags'):
        assert truth == 'flags'
        return [(game, seed, game, [(1, 0, 2)], np.array(list('HAD')))
                for game in games for seed in seeds]
    def dfl_cases(games, seeds, **kwargs):
        return cases(['dfl:' + game for game in games], seeds, **kwargs)
    monkeypatch.setattr(train, 'load_cases', cases)
    monkeypatch.setattr(train, 'load_dfl_cases', dfl_cases)
    monkeypatch.setattr(train, 'load_dfl_game', lambda game, truth: (None, None, np.array(list('AAD'))))
    monkeypatch.setattr(train, 'training_rows', lambda tracking, blocks, truth:
                        (np.ones((1, 1)), np.array([0 if tracking in train.GAMES else 2])))
    monkeypatch.setattr(train, 'fit', lambda *args: None)
    def predict(model, tracking, blocks):
        prediction_sources.append(tracking)
        return tracking
    monkeypatch.setattr(train, 'predict_proba', predict)
    def emissions(probs, prior, temperature):
        np.testing.assert_allclose(prior, [6/13, 0., 7/13])
        return probs, temperature
    monkeypatch.setattr(train, 'emissions', emissions)
    monkeypatch.setattr(train, 'decode', lambda evidence, **kwargs: evidence)
    def score(truth, prediction):
        game, temperature = prediction
        is_metrica = game in train.GAMES
        assert ''.join(truth) == ('HAD' if is_metrica else 'AAD')
        truths_scored.append(is_metrica)
        value = (.95 if is_metrica else 0.) if temperature == .1 else (0. if is_metrica else .6)
        return dict(spell_f1_mean=value, macro_f1=value, n_pred_spells=1., n_gt_spells=1.)
    monkeypatch.setattr(train, 'score', score)
    monkeypatch.setattr(train, 'GRID', {'TEMPERATURE': [.1, .2]})
    monkeypatch.setattr(train, 'save', lambda model, config, version: saved.append(config))
    train.main(['--dfl', '--dfl-oof', '--dfl-score-truth', 'events', '--tune-on', 'all',
                '--seeds', '0', '1', '--dfl-seeds', '0'])
    assert saved[0]['temperature'] == .2
    assert saved[0]['dfl_truth'] == 'flags'
    assert saved[0]['dfl_score_truth'] == 'events'
    assert saved[0]['tuning_cases'] == 'all-10 out-of-fold'
    assert saved[0]['snap_window_s'] == dfl.SNAP_WINDOW_S
    assert len(prediction_sources) == 13
    assert 'all-10 game-weighted mean spellF1 0.4200' in capsys.readouterr().out


@pytest.mark.parametrize('options', [['--tune-on', 'all'], ['--dfl-score-truth', 'events']])
def test_training_options_require_oof(options):
    from possession.train_learned import main
    with pytest.raises(SystemExit) as error:
        main(options)
    assert error.value.code == 2
