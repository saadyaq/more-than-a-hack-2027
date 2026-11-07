# Ball possession from automatic tracking — More than a Hack 2027, Phase 1

Submission of team **Saad Yaqine** to the [More than a Hack 2027](ORGANISERS_README.md) Initial
Challenge: for every frame of the tactical video of FC Barcelona – Athletic Club, label who has
the ball (`H` / `A`) or whether it is dead (`D`).

**Method in one line:** a gradient-boosting classifier turns 31 per-frame tracking features into
P(home, away, dead); Smart Tagging cues are added and a three-state Viterbi decoder produces whole
possessions. The model and every tuned parameter were fitted on external Metrica games only. The technical document is [`docs/saad_yaqine.md`](docs/saad_yaqine.md).

## Reproduce the submission

Requirements: Python 3.9+ (tested with 3.15), and the competition data bundle.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt          # pinned numpy, pandas, scikit-learn, pytest
export MTAH_DATA_DIR=/path/to/unpacked/bundle          # or unpack it into ./data
scripts/reproduce.sh saad_yaqine
```

This writes `evaluation/results/saad_yaqine.csv`, runs the official `validate_submission.py`
and prints the file's SHA-1. The submitted file has

```
9e4a3c00855b7c8fc6965eac744511a56d89da9e  evaluation/results/saad_yaqine.csv
```

The run takes about 10 s the first time (it parses and caches the tracking under
`$MTAH_DATA_DIR/cache/`) and about 4 s afterwards. It is deterministic. Everything it needs beyond
the data bundle is committed: `possession/learned_model.joblib` and `possession/learned_v7.json`
(classifier and decoder settings), `possession/shape_model.json`, and `possession/params.py`.

## Re-run the training (optional)

The tuned parameters and the shape model were produced from external data only, and can be
rebuilt without touching the competition data's labels:

```bash
scripts/fetch_external_data.sh                      # Metrica public sample games -> ~/ext_data/metrica
.venv/bin/python -m possession.train_shape          # -> possession/shape_model.json   (~10 s)
.venv/bin/python -m possession.train_learned        # -> learned_model.joblib, learned_v7.json (~15 min)
.venv/bin/python -m possession.tune --no-folds \
    --start possession/tuned_v4.json --output possession/tuned_v6.json      # (~40 min)
.venv/bin/python -m possession.external_eval --methods baseline v3          # external scores
```

`possession.tune` without `--no-folds` adds the leave-one-game-out check (about 2 h).

## Tests

```bash
.venv/bin/python -m pytest -q tests/        # synthetic tests; data tests run when MTAH_DATA_DIR is set
```

## Layout

| path | what |
|---|---|
| `possession/run.py` | CLI: `python -m possession.run --method v7 --name <file stem>` |
| `possession/data.py` | cached loader for the provided tracking and Smart Tagging |
| `possession/ball.py` | ball gap interpolation, ball speed, out-of-pitch test |
| `possession/evidence.py` | per-frame log-evidence for H / A / D |
| `possession/features.py` | the 31 per-frame features of the learned model |
| `possession/learned.py` | learned emission model (gradient boosting, team-symmetric) |
| `possession/shape.py` | team-shape features and model |
| `possession/decode.py` | three-state Viterbi decoder, per half |
| `possession/params.py` | every parameter, with the reason for its value |
| `possession/external.py` | Metrica sample loaders, event-derived possession truth, tracking degrader |
| `possession/train_learned.py`, `tune.py`, `train_shape.py`, `external_eval.py` | training and external evaluation |
| `possession/diagnostics.py` | aggregate statistics of a submission (no ground truth) |
| `docs/decisions.md` | dated log of every design decision and its measured effect |
| `docs/saad_yaqine.md` | technical document |
| `games.py`, `evaluation/`, `starter/`, `*.md` in capitals | the organisers' repository, unchanged |

The match data is not in this repository and must not be: it is licensed for the competition only.

## Credits

External training data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
(credit: Metrica Sports). Competition framework, scorer and baseline: the organisers'
repository, MIT licence (see `LICENSE`).
