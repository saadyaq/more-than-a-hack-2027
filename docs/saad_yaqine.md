# Ball possession from automatic tracking — technical document

**Team:** Saad Yaqine · **Challenge:** More than a Hack 2027, Initial Challenge (Phase 1)
**Submission:** `saad_yaqine.csv`, method `v10` · **Repository:**
https://github.com/saadyaq/more-than-a-hack-2027 (branch `dev`; the README gives the exact command
that regenerates the file).

## 1. Approach

Possession is a sequence of states, so we model it as one:

1. **Per-frame classifier.** A gradient-boosting classifier reads 31 features from the tracking
   (ball, players, team shape, stillness, ball-blackout context) and returns P(home, away, dead)
   for every frame.
2. **Smart Tagging cues.** Phase team tags, set-piece restarts and kick-off times are added as
   log-evidence with fixed weights.
3. **Decoder.** A three-state hidden-Markov decoder (Viterbi) finds the most likely H/A/D
   sequence for each half.

The decoder turns noisy frame-level beliefs into whole possessions. A change of team or a
stoppage costs a penalty, so a short flicker of contrary evidence is absorbed, while sustained
evidence switches the state. This matches the challenge's definition of controlled possession:
a touch, a deflection or an aerial duel without control does not change possession, and a loose
ball stays with the last team that had it. It also fits the ranking metric, spellF1, which scores
whole possessions.

The rules forbid fitting anything on the provided game. **The classifier and every tuned value
are therefore fitted on external games only** (section 3).

### 1.1 Features (`possession/features.py`)

| group | features |
|---|---|
| ball (when seen) | distance to the nearest home and away player and their difference; ball speed; position along home's attacking direction; distance to the nearest line; out of pitch; home and away players within 3 m |
| ball blackout context | time since the ball was last seen and until it is next seen; the home-minus-away distance, the attacking position and the out-of-pitch flag at the last and at the next sighting; the distance difference on a linearly interpolated ball (gaps ≤ 4 s) |
| team shape | territory, depth difference, width difference, collective forward movement (±1 s) |
| stillness | median player speed, and its rolling mean over ±1, ±3 and ±6 s; number of tracked players |
| control balance | (home − away) share of frames with a slow ball within 2 m of that team, over ±1, ±3 and ±8 s |

The labelling is offline, so features may look ahead as well as back. That matters because the
competition tracker loses the ball in long blackouts: about 150 gaps per match, median 10.7 s, and
99.7 % of missing frames are in gaps longer than 5 s. During a blackout, the team nearest the
ball when it vanished and when it reappears, and the team shapes in between, are the evidence
there is.

### 1.2 Team symmetry

The classifier is trained on every game twice: once as recorded, and once with the teams swapped
and the H/A labels exchanged. At prediction time the two views are averaged, so P(home | frame) is
exactly P(away | swapped frame). The model therefore cannot carry a home or away bias learned on
other games. A unit test enforces this.

### 1.3 Emissions and decoder

The emission score of state s is `T · (log P(s | x) − log prior(s))`. T is a tuned temperature, and
the prior is the class frequency in training. Smart Tagging terms are then added:

- attacking phases give +0.3 to the tagged team, defending phases +0.2 to the other team, capped
  at 0.6 per frame;
- set pieces give D over the 3 s before the restart, then +0.5 to the restarting team for 2 s;
- the graded half before each tagged Kick Off (Start) gets +10 to D.

These weights were set against clear ball control, worth 2 per frame in our hand-written model.
In the learned model they are rescaled by `T · ln 2.7 / 2`, so that they keep the same strength relative to clear
control. They could not be tuned, because the external games have no Smart Tagging.

Viterbi runs over (H, A, D), independently on each graded half, starting in D. The transition
penalties are P_SWITCH (H↔A), P_TO_DEAD and P_FROM_DEAD. One structural constraint is enforced in
code: `P_TO_DEAD + P_FROM_DEAD >= P_SWITCH`. Without it, the decoder hides turnovers behind phantom
one-frame stoppages, which our version v2 did on every turnover.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | **used**: every feature |
| tactical Smart Tagging | **used**: phase team tags, set-piece restarts, kick-off times |
| tactical homography | dropped: all our features are in pitch coordinates, which the ATD already gives |
| tactical video | not used |
| broadcast video, ATD, homography | dropped: not synchronised to the tactical timeline, and a third of its frames have no homography |
| public 5-minute ground truth | pipeline checks only (alignment, format, teams not inverted); never used to choose or fit anything |

One aggregate statistic of the provided tracking was used: the distribution of ball-gap lengths
(21 quantiles, `possession/ball_gap_quantiles.json`). It makes the external training data resemble
the competition tracker, and contains no labels and no frame-level information.

## 3. Training data

**External data only.** No label, annotation or human-produced data about the provided game was
used, and the video was never used.

- **Source.** Metrica Sports' public sample data, games 1-3
  (https://github.com/metrica-sports/sample-data): synchronised tracking and event data. The
  README invites community use and asks for acknowledgement; there is no formal licence file.
  Credit: Metrica Sports.
- **Labels.** Derived automatically from Metrica's event data on those games, by
  `possession_truth` in `possession/external.py`:
  - an owning event (pass, carry, recovery, set piece, shot) gives the ball to its team from its
    start frame;
  - a won challenge counts only if the same team has an owning event within 2 s;
  - a lost ball changes nothing until the other team's next owning event;
  - the ball is dead from a ball out, a shot going out or in, or a foul, until the next set
    piece, and each half starts dead until kick-off.

  This gives 258-325 possessions per game, of 11-12 s on average.
- **Dead-ball ball positions.** The Metrica sample tracking drops the ball at almost every
  stoppage: 32-41 % of graded frames have no ball, and 82-90 % of those are dead ball. The
  competition tracker behaves differently. It keeps seeing the ball out of play and loses it mostly
  in open play: the ball is out of the pitch at its last sighting before only 11 % of long gaps.
  Left as is, the external data teaches the model that a missing ball means a dead ball. During
  stoppages we therefore reconstruct the ball by linear interpolation between its exit point and
  its restart point (`fill_dead_ball`); open-play gaps are left untouched. After degradation,
  9 % of long gaps start with the ball out of the pitch (competition: 11 %), and the share of dead
  ball inside gaps falls from 71-87 % to 40-45 %, its overall base rate.
- **Degradation.** The sample tracking is cleaner than the competition's, so it is degraded to
  match the automatic tracker:
  - the ball is removed on 41 % of frames, in blackouts whose lengths follow the competition's
    ball-gap quantiles, started preferentially in crowded situations;
  - players are removed in runs of about a second, as many per frame as the competition
    tracker misses (its per-frame missing-count shares, an aggregate statistic), choosing those
    farthest from play, as a camera following the action would lose them;
  - speeds are kept from the clean track, as a smoothing tracker reports them, and blanked on 8 %
    of player observations, as in the competition ATD;
  - 0.15 m of time-correlated position noise is added (AR(1), about 1 s, as a filtered tracker
    drifts).

  Six random seeds per game (three before v10).
- **Fitting.** `possession/train_learned.py` builds about 520k training rows: every fifth graded
  frame of each game and seed (six seeds for v10, about 1 M rows), in both team orientations. The classifier is scikit-learn's
  HistGradientBoostingClassifier (300 iterations, learning rate 0.05, 31 leaves, minimum 200
  samples per leaf, L2 1.0, fixed random state).
- **Out-of-fold tuning.** For each external game, a classifier trained on the other two games
  predicts it. The temperature and the three penalties are tuned by coordinate descent on mean
  spellF1 over these out-of-fold predictions, so no score comes from a game its model saw. The
  final classifier is then refitted on all three games.

### 3.1 Results on external data (degraded; spellF1 is the competition metric)

| method | spellF1 | macroF1 | how it was measured |
|---|---|---|---|
| organisers' baseline (nearest player, ±2 s vote) | 0.31 | 0.69 | no tuning |
| v3: hand-written evidence, reasoned parameters | 0.33 | 0.68 | no tuning |
| v6: hand-written evidence + team shape, parameters tuned | 0.55 | 0.78 | in-sample |
| tuned hand-written evidence, leave-one-game-out | 0.49 | — | held-out game |
| v7: learned classifier, first degrader | 0.61 | 0.80 | out-of-fold, held-out game |
| v8: learned classifier, realistic player loss and speeds | 0.62 | 0.80 | out-of-fold, held-out game |
| v9: as v8, dead-ball ball positions reconstructed | 0.52 | 0.77 | out-of-fold, held-out game, harder data |
| **v10: as v9, six degradation seeds per game** | **0.53** | **0.78** | **out-of-fold, held-out game, harder data** |

Every v7, v8 and v9 number is measured on a game the classifier never saw. v9's score is lower
because its data is harder, not because the model is worse: v7 and v8 were scored on data where the
ball was mostly missing during stoppages, which made missing ball an easy dead-ball cue. On the
same realistic data as v9, a v8-style model (trained on v8's data, tested out of fold on v9's)
scores spellF1 0.39 and macroF1 0.64, against 0.52 and 0.77 for v9. v10 trains on six degradation
seeds per game instead of three, and is the submission. Per held-out game: 0.529, 0.539, 0.519
(v9: 0.515, 0.525, 0.520). The decoder settings were tuned on the pooled out-of-fold
predictions, which is slightly optimistic for four parameters.

### 3.2 Final settings (`possession/learned_v10.json`)

T = 0.2 · P_SWITCH = 2 · P_TO_DEAD = 16 · P_FROM_DEAD = 1 · prior (H, A, D) =
(0.28, 0.28, 0.44).

## 4. Possession rules implemented

- **Who has the ball.** The classifier learns, from external games, what controlled possession
  looks like in the tracking. The decoder then demands that the evidence for a change of team
  outweigh P_SWITCH, so a single touch or deflection does not switch possession.
- **Loose and contested balls.** Where the evidence is balanced, the decoder stays in its current
  state, so the last team keeps possession.
- **Ball lost by the tracker.** The blackout context (team nearest at the last and next sighting)
  and team shape decide.
- **Dead ball.** The classifier's D probability comes from stillness, the ball out of the pitch
  and blackout context. On top of it come the seconds before a tagged set piece and everything
  before each half's kick-off. When the ball is dead the label is D; we do not assign dead-ball
  ownership.
- **Restarts.** The restarting team named by Smart Tagging gets a brief prior. Throw-ins are
  tagged without a team, so there the evidence after the restart decides.
- **Outside the graded halves.** `X`.

## 5. Limitations and what we would do next

- **Domain gap.** The classifier is trained on degraded external games, whose labels come from
  event data rather than from corrected tracking. We compared the distribution of every feature
  between the competition match and the degraded external games. The first degrader tracked too
  many players and inflated still-player speeds; profile v2 fixes both (tracked players p10/50/90
  16/20/21 against 15/20/21, still-player speed p10 0.60 against 0.60 m/s). The ball still
  reappears outside the pitch more often in the competition (restarts taken from beyond the line). External scores are an estimate, not a prediction of our score.
- **Untuned Smart Tagging weights.** The external games have no Smart Tagging, so its weights
  are rescaled by reasoning, not fitted. Simulating tags to fit them was rejected: the real tags'
  accuracy is unknown, so any simulated noise level would be arbitrary. On the competition match,
  v10 agrees with the tagged attacking team on 52-58 % of tagged frames (v8: 60-62 %), while its
  predicted dead-ball share (33-43 %) is closer to football's usual ~40 % than v8's (29-35 %).
- **Three training games.** More external games with tracking and events would reduce variance,
  and per-game scores still range from 0.52 to 0.54 (v8: 0.56 to 0.66).
- **Next steps:** synthesise Smart Tagging-like cues on the external games so their weights can be fitted; add
  external games.

## 6. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
export MTAH_DATA_DIR=/path/to/unpacked/bundle
scripts/reproduce.sh saad_yaqine
# -> evaluation/results/saad_yaqine.csv, SHA-1 cbffc044b141415d91e3781e377fb199cd612c64
```

The run is deterministic and takes about 4 s once the tracking is cached. The training can be
re-run from `scripts/fetch_external_data.sh`, then `python -m possession.train_shape` and
`python -m possession.train_learned --version v10 --profile v3 --seeds 0 1 2 3 4 5`. Every design decision is logged in `docs/decisions.md`,
with its measured effect.

## 7. Tools

The code was written with AI coding assistants (Claude Code, OpenAI Codex). In line with the
terms, no provided data file, video or extract was sent to them. The match data was kept outside
the repository, and the assistants only saw aggregate run outputs (counts, timings, quantiles,
scores on external data) and the public documentation.
