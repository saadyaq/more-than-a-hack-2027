"""Train the learned emission model and tune its decoder, on external Metrica games only.

    .venv/bin/python -m possession.train_learned       # -> learned_model.joblib, learned_v7.json

1. Out-of-fold probabilities: for each external game, a model trained on the other two games
   predicts it (all three degradation seeds). Nothing is ever scored on a game its model saw.
2. The temperature and the three transition penalties are tuned by coordinate descent on mean
   spellF1 over those out-of-fold predictions, with the fake-stoppage constraint enforced.
3. The final model is fitted on all three games and saved with the tuned decoder settings.

The out-of-fold spellF1 printed at the end is the honest estimate to compare with v6's 0.55.
"""

import numpy as np

from possession import params as p
from possession.decode import decode
from possession.external_eval import score
from possession.features import FEATURE_NAMES
from possession.learned import emissions, fit, predict_proba, save, training_rows
from possession.tune import GAMES, MIN_GAIN, SEEDS, load_cases

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


def main():
    cases = load_cases(GAMES, SEEDS)
    rows = {}
    for index, (game, _, tracking, blocks, truth) in enumerate(cases):
        rows[index] = training_rows(tracking, blocks, truth)
    log("training rows per case: {}".format(len(rows[0][1])))

    def stack(indices):
        return (np.vstack([rows[i][0] for i in indices]),
                np.concatenate([rows[i][1] for i in indices]))

    oof = {}
    for held_out in GAMES:
        train = [i for i, case in enumerate(cases) if case[0] != held_out]
        model = fit(*stack(train))
        for i, case in enumerate(cases):
            if case[0] == held_out:
                oof[i] = predict_proba(model, case[2], case[3])
        log("fold {} trained on {} rows".format(held_out, len(stack(train)[1])))

    _, all_y = stack(range(len(cases)))
    prior = np.bincount(all_y, minlength=3) / len(all_y)

    def metrics(point):
        penalties = penalties_of(point)
        results = [score(case[4], decode(emissions(oof[i], prior, point["TEMPERATURE"]),
                                         blocks=case[3], penalties=penalties))
                   for i, case in enumerate(cases)]
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
        values = [r["spell_f1_mean"] for r, case in zip(per_case, cases) if case[0] == game]
        log("  held-out game {}: spellF1 {:.4f}".format(game, float(np.mean(values))))

    model = fit(*stack(range(len(cases))))
    save(model, {"features": list(FEATURE_NAMES), "prior": prior.tolist(),
                 "temperature": point["TEMPERATURE"], "penalties": penalties_of(point),
                 "out_of_fold_spell_f1": mean["spell_f1_mean"],
                 "out_of_fold_macro_f1": mean["macro_f1"],
                 "trained_on": "Metrica public sample games 1-3, degraded seeds 0-2 (external only)"})
    log("saved model and config")


if __name__ == "__main__":
    main()
