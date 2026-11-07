"""Evaluate fixed methods on Metrica event truth, without Smart Tagging or tuning.

Example: .venv/bin/python -m possession.external_eval --methods baseline v1 v2 v3
         --seeds 0 1 2
Metrica Sports supplies these public samples; see external.py for usage terms.
The degraded mean weights each game/seed equally, never competition ground truth.
"""

import argparse
import json

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from possession import params as p
from possession.baseline import majority_smooth, nearest_team
from possession.decode import decode
from possession.evidence import EVIDENCE_PARAMS, build_evidence
from possession.external import (FPS, degrade, load_metrica_events, load_metrica_game,
                                 possession_truth)

METHODS = ('baseline', 'v1', 'v2', 'v3')
TRANSITION_PARAMS = frozenset({'P_SWITCH', 'P_TO_DEAD', 'P_FROM_DEAD'})


def run_method(method, tracking, blocks, params=None):
    """Pure grid-search entry point: overrides use possession.params constant names.

    Baseline accepts DEFAULT_WINDOW. v1/v2/v3 accept EVIDENCE_PARAMS plus the
    three transition penalties. Unknown keys fail rather than silently doing
    nothing. No global constants are mutated. Explicit periods mask baseline
    output and stop its smoothing window crossing a halftime boundary.
    """
    if method not in METHODS:
        raise ValueError('Unknown method: {}'.format(method))
    params = {} if params is None else dict(params)
    supported = {'DEFAULT_WINDOW'} if method == 'baseline' else EVIDENCE_PARAMS | TRANSITION_PARAMS
    if set(params) - supported:
        raise ValueError('Unknown parameters for {}: {}'.format(method, sorted(set(params) - supported)))
    if method == 'baseline':
        window = params.get('DEFAULT_WINDOW', p.DEFAULT_WINDOW)
        if not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError('DEFAULT_WINDOW must be a nonnegative integer')
        codes = nearest_team(tracking)
        labels = np.full(len(codes), 'X', dtype='<U1')
        for _, start, end in blocks:
            start, stop = max(0, start), min(len(labels), end + 1)
            if start < stop:
                labels[start:stop] = majority_smooth(codes[start:stop], window)
        return labels
    penalties = dict(p.METHOD_TRANSITION_PENALTIES[method])
    penalties.update({key: value for key, value in params.items() if key in TRANSITION_PARAMS})
    if method not in p.HISTORICAL_METHODS:
        p.check_no_cheap_fake_stoppage(penalties)
    evidence = build_evidence(tracking, pd.DataFrame(), blocks=blocks, fps=FPS,
                              params={key: value for key, value in params.items()
                                      if key in EVIDENCE_PARAMS})
    return decode(evidence, blocks=blocks, penalties=penalties)


def score(truth, labels):
    """Skip bootstrap draws: the harness reports point estimates for grid searches."""
    metrics = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
    return {key: metrics[key] for key in (
        'spell_f1_mean', 'macro_f1', 'n_pred_spells', 'n_gt_spells',
        'mean_spell_duration_s', 'gt_mean_spell_duration_s')}


def truth_stats(truth):
    codes = pm.label_codes(truth)
    stats = pm.possession_stats(codes, pm.valid_runs(codes))
    return stats['n_spells'], stats['mean_duration_s'], float(np.mean(truth[truth != 'X'] == 'D'))


def _print_row(game, condition, method, metrics):
    print('{:<5} {:<12} {:<9} {:8.4f} {:8.4f} {:8.1f}/{:<8.1f} {:7.2f}/{:<7.2f}'.format(
        game, condition, method, metrics['spell_f1_mean'], metrics['macro_f1'],
        metrics['n_pred_spells'], metrics['n_gt_spells'],
        metrics['mean_spell_duration_s'], metrics['gt_mean_spell_duration_s']), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--games', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--root', help='external Metrica root; default EXT_DATA_DIR or ~/ext_data/metrica')
    parser.add_argument('--params', default='{}', help='JSON object keyed by method, then constant name')
    args = parser.parse_args(argv)
    overrides = json.loads(args.params)
    if not isinstance(overrides, dict) or set(overrides) - set(METHODS):
        parser.error('--params must be an object keyed by method name')
    rows = []
    print('game  condition    method     spellF1  macroF1   possessions P/T   mean seconds P/T', flush=True)
    for game in args.games:
        tracking, blocks, home = load_metrica_game(game, args.root)
        events = load_metrica_events(game, args.root)
        truth = possession_truth(events, len(tracking.ball), blocks, home)
        count, duration, dead = truth_stats(truth)
        print('Truth game {}: {} possessions, {:.2f} s mean, {:.2%} dead'.format(
            game, count, duration, dead), flush=True)
        for seed in [None] + args.seeds:
            condition = 'clean' if seed is None else 'degraded/{}'.format(seed)
            observations = tracking if seed is None else degrade(tracking, seed, blocks=blocks)
            for method in args.methods:
                labels = run_method(method, observations, blocks, overrides.get(method))
                metrics = score(truth, labels)
                _print_row(game, condition, method, metrics)
                if seed is not None:
                    rows.append(dict(game=game, seed=seed, method=method, **metrics))
    print('Equal-weight mean over degraded games x seeds:', flush=True)
    for method in args.methods:
        selected = [row for row in rows if row['method'] == method]
        mean = {key: float(np.mean([row[key] for row in selected])) for key in metrics}
        _print_row('ALL', 'degraded', method, mean)


if __name__ == '__main__':
    main()
