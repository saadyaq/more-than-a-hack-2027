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
