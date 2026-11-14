"""Train emissions on external games; tune the decoder only on Metrica out-of-fold cases.

    .venv/bin/python -m possession.train_learned --dfl --profile v3 --version v11

DFL games augment each training fold and the final model. Optional DFL held-out scores
are reported separately and never enter decoder tuning or the Metrica emission prior.
Defaults retain the original Metrica-only v7 training configuration.
"""

import numpy as np

from possession import params as p
from possession.decode import decode
from possession.external_eval import score
from possession.features import FEATURE_NAMES
from possession.learned import emissions, fit, predict_proba, save, training_rows
from possession.tune import GAMES, MIN_GAIN, SEEDS, load_cases, load_dfl_cases
from possession.dfl import DFL_GAMES, DFL_MIN_SPELL_FRAMES

# Params
GRID = {  # W_CONTROL-free scale: emissions are log-likelihood ratios times the temperature.
    "TEMPERATURE": [0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
    "P_SWITCH": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "P_TO_DEAD": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "P_FROM_DEAD_EXTRA": [0.0, 1.0, 4.0],
}
START = {"TEMPERATURE": 0.1, "P_SWITCH": 8.0, "P_TO_DEAD": 8.0, "P_FROM_DEAD_EXTRA": 0.0}
MAX_PASSES = 4


def penalties_of(point):
    switch, to_dead = point["P_SWITCH"], point["P_TO_DEAD"]
    penalties = {"P_SWITCH": switch, "P_TO_DEAD": to_dead,
                 "P_FROM_DEAD": max(0., switch - to_dead) + point["P_FROM_DEAD_EXTRA"]}
    p.check_no_cheap_fake_stoppage(penalties)
    return penalties


def log(line):
    print(line, flush=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v7", help="output version name (v7 = original files)")
    parser.add_argument("--profile", default="v1", choices=["v1", "v2", "v3"],
                        help="degrader profile: v2 matches player loss and speeds; v3 also fills dead-ball gaps")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="degradation seeds per external game")
    parser.add_argument("--dfl", action="store_true", help="also train on seven DFL open games")
    parser.add_argument("--dfl-seeds", type=int, nargs="+", default=None,
                        help="DFL degradation seeds (default: --seeds)")
    parser.add_argument("--dfl-oof", action="store_true", help="also report DFL held-out games")
    args = parser.parse_args(argv)
    if args.dfl_oof and not args.dfl:
        parser.error("--dfl-oof requires --dfl")
    dfl_seeds = args.seeds if args.dfl_seeds is None else args.dfl_seeds
    cases = load_cases(GAMES, tuple(args.seeds), profile=args.profile)
    if args.dfl:
        cases += load_dfl_cases(DFL_GAMES, tuple(dfl_seeds), profile=args.profile)
    metrica_indices = [i for i, case in enumerate(cases) if case[0] in GAMES]
    dfl_indices = [i for i, case in enumerate(cases) if case[0] not in GAMES]
    rows = {}
    for index, (game, _, tracking, blocks, truth) in enumerate(cases):
        rows[index] = training_rows(tracking, blocks, truth)
    log("training rows per case: {}".format(len(rows[0][1])))

    for source, indices in (("Metrica", metrica_indices), ("DFL", dfl_indices)):
        if indices:
            log("training rows {}: {}".format(source, sum(len(rows[i][1]) for i in indices)))

    def stack(indices):
        return (np.vstack([rows[i][0] for i in indices]),
                np.concatenate([rows[i][1] for i in indices]))

    oof = {}
    held_out_games = list(GAMES)
    if args.dfl_oof:
        held_out_games += ["dfl:" + match_id for match_id in DFL_GAMES]
    for held_out in held_out_games:
        train = [i for i, case in enumerate(cases) if case[0] != held_out]
        model = fit(*stack(train))
        for i, case in enumerate(cases):
            if case[0] == held_out:
                oof[i] = predict_proba(model, case[2], case[3])
        log("fold {} trained on {} rows".format(held_out, sum(len(rows[i][1]) for i in train)))

    # DFL labels use a different possession definition: only Metrica supplies the prior.
    counts = sum((np.bincount(rows[i][1], minlength=3) for i in metrica_indices),
                 np.zeros(3, dtype=np.int64))
    prior = counts / counts.sum()
    log("Metrica emission prior: {}".format(prior.tolist()))

    def metrics(point, indices=None):
        indices = metrica_indices if indices is None else indices
        penalties = penalties_of(point)
        results = [score(case[4], decode(emissions(oof[i], prior, point["TEMPERATURE"]),
                                         blocks=case[3], penalties=penalties))
                   for i in indices for case in [cases[i]]]
        return {key: float(np.mean([r[key] for r in results])) for key in results[0]}, results

    point = dict(START)
    best = metrics(point)[0]["spell_f1_mean"]
    log("start out-of-fold spellF1 {:.4f}".format(best))
    for sweep in range(1, MAX_PASSES + 1):
        changed = False
        for name, values in GRID.items():
            for value in values:
                if value == point[name]:
                    continue
                candidate = dict(point, **{name: value})
                result = metrics(candidate)[0]["spell_f1_mean"]
                if result > best + MIN_GAIN:
                    point, best, changed = candidate, result, True
            log("  pass {} {:<18} -> {:<6} spellF1 {:.4f}".format(sweep, name, point[name], best))
        if not changed:
            break

    mean, per_case = metrics(point)
    log("out-of-fold: spellF1 {:.4f}  macroF1 {:.4f}  possessions {:.1f}/{:.1f}".format(
        mean["spell_f1_mean"], mean["macro_f1"], mean["n_pred_spells"], mean["n_gt_spells"]))
    for game in GAMES:
        values = [r["spell_f1_mean"] for r, i in zip(per_case, metrica_indices)
                  if cases[i][0] == game]
        macros = [r["macro_f1"] for r, i in zip(per_case, metrica_indices)
                  if cases[i][0] == game]
        log("  held-out game {}: spellF1 {:.4f}  macroF1 {:.4f}".format(
            game, float(np.mean(values)), float(np.mean(macros))))

    if args.dfl_oof:
        dfl_mean, dfl_results = metrics(point, dfl_indices)
        for match_id in DFL_GAMES:
            results = [r for r, i in zip(dfl_results, dfl_indices)
                       if cases[i][0] == "dfl:" + match_id]
            log("  held-out DFL {}: spellF1 {:.4f}  macroF1 {:.4f}".format(
                match_id, float(np.mean([r["spell_f1_mean"] for r in results])),
                float(np.mean([r["macro_f1"] for r in results]))))
        log("DFL mean: spellF1 {:.4f}  macroF1 {:.4f}".format(
            dfl_mean["spell_f1_mean"], dfl_mean["macro_f1"]))

    model = fit(*stack(range(len(cases))))
    extra = {}
    if args.dfl:
        extra = {"trained_on": "Metrica public sample games 1-3 + 7 DFL open games, degraded (external only)",
                 "dfl_games": list(DFL_GAMES), "dfl_seeds": list(dfl_seeds),
                 "dfl_min_spell_frames": DFL_MIN_SPELL_FRAMES,
                 "tuning_cases": "metrica out-of-fold"}
    save(model, {"features": list(FEATURE_NAMES), "prior": prior.tolist(),
                 "temperature": point["TEMPERATURE"], "penalties": penalties_of(point),
                 "out_of_fold_spell_f1": mean["spell_f1_mean"],
                 "out_of_fold_macro_f1": mean["macro_f1"],
                 "trained_on": "Metrica public sample games 1-3, degraded (external only)",
                 "degrader_profile": args.profile, "seeds": list(args.seeds),
                 **extra}, version=args.version)
    log("saved model and config")


if __name__ == "__main__":
    main()
