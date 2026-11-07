# Ball possession from automatic tracking — technical document

**Team:** Saad Yaqine · **Challenge:** More than a Hack 2027, Initial Challenge (Phase 1)
**Submission:** `saad_yaqine.csv`, method `v6` · **Repository:**
https://github.com/saadyaq/more-than-a-hack-2027 (branch `dev`; the README gives the exact command
that regenerates the file).

## 1. Approach

Possession is a sequence of states, so we treat it as one: an evidence model scores each frame
for the three states home (H), away (A) and dead ball (D), and a hidden-Markov decoder (Viterbi)
finds the most likely state sequence for each half. The decoder is what turns noisy per-frame
cues into whole possessions. A change of team or a stoppage costs a penalty, so a short flicker
of contrary evidence is absorbed, while sustained evidence switches the state. That matches the
challenge's definition: a touch, a deflection or a lost aerial duel is not a change of possession,
and a contested ball stays with the last team that had it.

We chose this over a learned frame classifier for two reasons. The ranking metric, spellF1,
rewards whole possessions, which an explicit transition model produces directly. And the rules
forbid fitting on the provided game, so the model needs few, interpretable parameters that can be
fitted on other games.

### Evidence terms (log-scores per frame, `possession/evidence.py`)

| term | state | rule |
|---|---|---|
| ball control | H or A | ball within `CONTROL_RADIUS_M` of a player and slower than `CONTROL_SPEED_MPS`: `W_CONTROL` to that player's team, weakened when an opponent is almost as close (`CONTEST_MARGIN_M`). Interpolated ball positions get `INTERPOLATED_CONTROL_WEIGHT` |
| team shape | H vs A | log-odds from a team-shape model (below), times `W_SHAPE`; times `SHAPE_BALL_FACTOR` when the ball is seen |
| Smart Tagging phases | H or A | attacking phases (Build Up, Progression, Creating Chances, Ball in Final Third / Box, Players in the Box, Attacking Transition, Long Ball): +0.3 to the tagged team. Defending phases: +0.2 to the other team. Capped at 0.6 per frame |
| ball out of pitch | D | observed ball beyond a line by more than 0.5 m: `W_OUT` |
| collective stillness | D | rolling mean (±`STILL_WINDOW_S`) of the median player speed below `STILL_SPEED_MPS`: `W_STILL`, plus `W_STILL_NO_BALL` if the ball is unseen throughout |
| set-piece restart | D, then H/A | Smart Tagging set pieces: D for the 3 s before the restart, then +0.5 to the restarting team for 2 s when the tag names it |
| before kick-off | D | from the start of each graded half to its tagged Kick Off (Start): +10 |

### Team-shape model (`possession/shape.py`)

The competition tracker loses the ball in long blackouts: about 150 gaps per match, median 10.7 s,
and 99.7 % of missing frames lie in gaps longer than 5 s. For those stretches, the geometry of the
22 players is the only signal. Four features, each smoothed over ±1 s:

- **territory:** where the whole block of players sits along home's attacking direction;
- **depth:** home minus away spread along the pitch;
- **width:** home minus away spread across the pitch;
- **move:** mean forward velocity of all players along home's attacking direction.

Every feature changes sign when home and away are swapped, and the logistic model has **no
intercept**. It is therefore symmetric by construction and cannot carry a home advantage learned
on other games. On a held-out external game it predicts the team in possession on 82-83 % of
alive frames (chance 50 %). Width is the dominant feature (weight +1.88): the team with the ball
stretches across the pitch.

### Decoder (`possession/decode.py`)

Viterbi over (H, A, D), run independently on each graded half, starting in D. The transition
penalties are `P_SWITCH` (H↔A), `P_TO_DEAD` and `P_FROM_DEAD`, with one structural constraint
enforced in code: `P_TO_DEAD + P_FROM_DEAD >= P_SWITCH`. Without it, the decoder hides turnovers
behind phantom one-frame stoppages, which an earlier version (v2) did on every turnover.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | **used**: positions and speeds for every evidence term |
| tactical Smart Tagging | **used**: phase team tags, set-piece restarts, kick-off times |
| tactical homography | dropped: our features are all in pitch coordinates, which the ATD already gives |
| tactical video | not used |
| broadcast video, ATD, homography | dropped: not synchronised to the tactical timeline, and a third of its frames have no homography |
| public 5-minute ground truth | pipeline checks only (alignment, format, teams not inverted). Never used to choose or fit anything |

One aggregate statistic of the provided tracking was used: the distribution of ball-gap lengths
(21 quantiles, `possession/ball_gap_quantiles.json`), to make the external training data
resemble the competition tracker. It contains no labels and no frame-level information.

## 3. Training data

**Only external data.** No label, annotation or human-produced data about the provided game was
used, and the video was never used.

- **Source:** Metrica Sports' public sample data, games 1-3
  (https://github.com/metrica-sports/sample-data), synchronised tracking and event data.
  Its README invites community use and asks for acknowledgement; it has no formal licence file.
  Credit: Metrica Sports.
- **Labels:** derived automatically from Metrica's event data on those games
  (`possession_truth` in `possession/external.py`). An owning event (pass, carry, recovery, set
  piece, shot) gives the ball to its team from its start frame. A won challenge counts only if
  that team has an owning event within 2 s. A lost ball changes nothing until the other team's
  next owning event. The ball is dead from a ball-out event, a shot that goes out or in, or a
  foul, until the next set piece; each half starts dead until kick-off. This yields 258-325
  possessions per game with a mean duration of 11-12 s.
- **Degradation:** the sample tracking is cleaner than the competition's, so it is degraded to
  match the automatic tracker. The ball is removed on 41 % of frames, in blackouts whose lengths
  follow the competition's ball-gap quantiles, started preferentially in crowded situations.
  Player observations are dropped on 7 % of frames, and 0.15 m of time-correlated position noise
  is added (AR(1), about 1 s, as a filtered tracker drifts). Three random seeds per game.
- **Fitting:** the shape model is a ridge-penalised logistic regression on every fifth alive
  frame (144k frames). The evidence and decoder parameters were tuned by coordinate descent on
  mean spellF1 over the 3 games × 3 seeds (`possession/tune.py`). Smart Tagging and restart
  weights were **not** tuned, because the external games have no Smart Tagging. They keep the
  reasoned values in the table above.

### Results on external data (degraded, spellF1 = the competition metric)

| method | spellF1 | macroF1 |
|---|---|---|
| organisers' baseline (nearest player, ±2 s vote) | 0.31 | 0.69 |
| v3 (our model, reasoned parameters) | 0.33 | 0.68 |
| v6 (tuned, with team shape) | **0.55** | **0.78** |

Leave-one-game-out (tune on two games, test on the third; without shape): game 1 0.378 → 0.501,
game 2 0.319 → 0.540, game 3 0.291 → 0.440. The tuning generalises to an unseen game, by
+0.12 to +0.22.

### Final parameters (v6, `possession/tuned_v6.json`)

`CONTROL_RADIUS_M` 1.5 · `CONTROL_SPEED_MPS` 30 · `CONTEST_MARGIN_M` 0.25 ·
`INTERPOLATED_CONTROL_WEIGHT` 1.0 · `MAX_GAP_FRAMES` 100 · `W_OUT` 12 · `STILL_WINDOW_S` 1 ·
`STILL_SPEED_MPS` 1.6 · `W_STILL` 0.25 · `W_STILL_NO_BALL` 1.0 · `W_SHAPE` 0.1 ·
`SHAPE_BALL_FACTOR` 1.0 · `P_SWITCH` 20 · `P_TO_DEAD` 25 · `P_FROM_DEAD` 0 (`W_CONTROL` = 2
fixes the scale).

## 4. Possession rules implemented

- **Who has the ball:** a team gains possession only through sustained evidence, mainly the ball
  close to and slow at one of its players. The decoder requires enough of it to pay `P_SWITCH`,
  roughly half a second of clear control, or longer when the evidence is contested.
- **Loose and contested balls:** contested control is down-weighted, and absent evidence leaves
  the decoder in its current state, so the last team keeps possession.
- **Ball lost by the tracker:** team shape takes over, with no other team cue.
- **Dead ball:** ball out of the pitch, collective stillness, the seconds before a tagged set
  piece, and everything before each half's kick-off. When the ball is dead the label is D; we
  do not assign dead-ball ownership.
- **Restarts:** the restarting team named by Smart Tagging gets a brief prior. Throw-ins are
  tagged without a team, so there the first control decides.
- **Outside the graded halves:** `X`.

## 5. Limitations and what we would do next

- **Simulator gap.** All tuning happens on degraded external games. Their tracking, their
  event-derived labels and our degradation model all differ from the competition's corrected
  tracking ground truth, so external scores are an estimate, not a prediction of our score.
- **Untuned Smart Tagging weights.** On the competition match, v6 agrees with the Smart Tagging
  attacking team on 57-60 % of tagged frames, against 62-68 % for v3. The tags are noisy, but this
  drop is the one diagnostic that moved the wrong way.
- **Grid edges.** Several tuned values sit on the edge of their search range, so the optimum may
  lie beyond it.
- **Dead-ball detection** relies on stillness and an observed ball out of play. Quick restarts
  and stoppages while the ball is unseen are its weak cases.
- **Next steps:** pseudo-tags for the external games to tune the Smart Tagging weights; a
  learned emission model (e.g. gradient boosting on the evidence features) trained on external
  games; and more external games with tracking and events.

## 6. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
export MTAH_DATA_DIR=/path/to/unpacked/bundle
scripts/reproduce.sh saad_yaqine
# -> evaluation/results/saad_yaqine.csv, SHA-1 36e144635f0f6e244d66203f938aba5bed2c04d4
```

Deterministic, about 1 s once the tracking is cached. Training can be re-run from
`scripts/fetch_external_data.sh` (see README). Every design decision, with its measured effect,
is logged in `docs/decisions.md`.

## 7. Tools

The code was written with AI coding assistants (Claude Code, OpenAI Codex). In line with the
terms, no provided data file, video or extract was sent to them. The match data was kept outside
the repository, and the assistants only ever saw aggregate run outputs (counts, timings, scores
on external data) and the public documentation.
