"""Summarise possession and automatic-tag agreement without exposing frame-level labels."""

import argparse

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks, source_frames
from possession import params as p
from possession.data import load_smart_tagging
from possession.evidence import phase_intervals


def offensive_coverage(events, n_frames):
    """Union same-team phases so repeated attacking descriptions do not multiply votes.

    Conflicting team tags each count once: agreement is over tagged team/frame pairs.
    """
    changes = np.zeros((n_frames + 1, 2), dtype=np.int64)
    for team, start, stop, _ in phase_intervals(events, n_frames, offensive_only=True):
        changes[start, team] += 1
        changes[stop, team] -= 1
    return np.cumsum(changes[:-1], axis=0) > 0


def summarise(labels, events, blocks=None, *, fps=p.FPS):
    """Home share uses alive H+A frames; dead share uses all frames of the graded half."""
    if blocks is None:
        blocks = half_video_blocks(GAME)
    labels = np.asarray(labels)
    coverage = offensive_coverage(events, len(labels))
    reports = []
    for half, start, end in blocks:
        start, stop = max(0, start), min(len(labels), end + 1)
        if start >= stop:
            continue
        part = labels[start:stop]
        boundaries = np.r_[0, np.flatnonzero(part[1:] != part[:-1]) + 1, len(part)]
        alive_runs = np.isin(part[boundaries[:-1]], ["H", "A"])
        durations = np.diff(boundaries)[alive_runs] / fps
        home, away = np.count_nonzero(part == "H"), np.count_nonzero(part == "A")
        direct = (((part[:-1] == "H") & (part[1:] == "A"))
                  | ((part[:-1] == "A") & (part[1:] == "H")))
        tags = coverage[start:stop]
        agree = np.count_nonzero(tags & (part[:, None] == np.array(["H", "A"])))
        tagged = int(np.count_nonzero(tags))
        reports.append(dict(
            half=half, possessions=len(durations),
            mean_s=float(durations.mean()) if len(durations) else 0.,
            median_s=float(np.median(durations)) if len(durations) else 0.,
            dead_share=float(np.mean(part == "D")),
            home_share=float(home / (home + away)) if home + away else float("nan"),
            direct_switches=int(np.count_nonzero(direct)), tagged_frames=tagged,
            tag_agreement=float(agree / tagged) if tagged else float("nan")))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission")
    args = parser.parse_args(argv)
    try:
        table = pm.read_label_csv(args.submission)
        valid = pm.validate_submission(table, source_frames()) is None
    except Exception:
        raise SystemExit("Could not read submission (details suppressed)") from None
    if not valid:
        raise SystemExit("Invalid submission (details suppressed)")
    labels = np.char.upper(np.char.strip(table["label"].to_numpy(dtype=str)))
    reports = summarise(labels, load_smart_tagging())
    print("Home share = H/(H+A); tag agreement counts each tagged team/frame once.")
    for report in reports:
        print("half {half}: possessions {possessions}; mean {mean_s:.2f} s; median {median_s:.2f} s; "
              "D {dead_share:.2%}; home {home_share:.2%}; direct H/A switches {direct_switches}; "
              "attack-tag agreement {tag_agreement:.2%} ({tagged_frames} tagged team/frames)"
              .format(**report))


if __name__ == "__main__":
    main()
