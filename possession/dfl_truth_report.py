"""Print aggregate external possession definitions without exposing observations.

Run with .venv/bin/python -m possession.dfl_truth_report (or this file directly).
Rates use recorded block duration, including stoppages, per 45 minutes.
"""

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from possession.dfl import DFL_GAMES, dfl_truth_details
from possession.external import FPS, load_metrica_events, load_metrica_game, possession_truth
from possession.external_eval import truth_stats


def possessions_per_45(truth, blocks):
    """Count H/A spells with the evaluator and normalize by recorded match time."""
    count, _, _ = truth_stats(truth)
    frames = sum(end - start + 1 for _, start, end in blocks)
    return count * (45 * 60 * FPS) / frames


def shares(truth):
    return '/'.join('{}={:.2%}'.format(label, np.mean(truth == label)) for label in 'HADX')


def main():
    for game in DFL_GAMES:
        _, blocks, flags, events, alive, stats = dfl_truth_details(game)
        agreement = float(np.mean(flags[alive] == events[alive])) if np.any(alive) else 0.
        print('DFL {} n={} flags[{}] events[{}] poss/45 flags={:.2f} events={:.2f} '
              'alive agreement={:.2%} snap={:.2%} median|shift|={:.2f}s '
              'dropped={} fallback_runs={}'.format(
                  game, len(flags), shares(flags), shares(events),
                  possessions_per_45(flags, blocks), possessions_per_45(events, blocks),
                  agreement, stats['snap_share'], stats['median_shift_s'],
                  stats['events_dropped'], stats['fallback_runs']), flush=True)
    for game in (1, 2, 3):
        tracking, blocks, home = load_metrica_game(game)
        truth = possession_truth(load_metrica_events(game), len(tracking.ball), blocks, home)
        print('Metrica {} poss/45 events={:.2f}'.format(
            game, possessions_per_45(truth, blocks)), flush=True)


if __name__ == '__main__':
    main()
