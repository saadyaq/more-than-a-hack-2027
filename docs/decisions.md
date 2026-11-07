# Decisions log

## 2026-11-07 — Task 1: pipeline skeleton (Codex)

- **What:** `possession/` package — cached ATD loader (`data.py`), vectorised baseline
  (`baseline.py`), writer (`io.py`), method registry CLI (`run.py`); 30 tests.
- **Why:** every later method plugs into the same loader and CLI; reproducing the official
  baseline byte-for-byte proves the loader and frame alignment are right.
- **Result:** `ours_baseline50.csv` is SHA-identical to `baseline_smooth50.csv`. Load 6 s cold,
  0.05 s warm (cache in `~/mtah_data/cache`, 251 MB). Schema: every player and ball group has
  exactly 3 fields (x, y, speed) — the raw ATD carries **no** ball-owner or ball-alive flag, so
  dead ball and possession must be inferred entirely.

## 2026-11-07 — Task 2: method v1 (evidence + 3-state Viterbi)

- **What:** ball cleanup (short-gap interpolation, out-of-pitch, speed), per-frame log-evidence
  for H/A/D (close slow ball = control, Smart Tagging phases as a weak team prior, out of pitch /
  collective stillness / set-piece lead-in / pre-kickoff as dead evidence), Viterbi per half with
  switch and alive/dead transition penalties. All parameters in `possession/params.py`, set from
  football reasoning, not from the public ground truth.
- **Full-match diagnostics (no ground truth involved):** 445 possessions, median 4-5 s,
  D share 35-40 %, 112-117 direct H<->A switches per half; agreement with Smart Tagging attacking
  team 65 % (baseline50: 49 %).
- **Public window, bug check only:** valid, teams not inverted, latency +0.9 frames.

## 2026-11-07 — Task 3: v2, penalties in units of evidence

- **What:** `P_SWITCH = W_CONTROL * FPS * MIN_CONTROL_S` (1 s of clear control, = 50) and
  `P_TO_DEAD = W_STILL * FPS * MIN_STOPPAGE_S` (1 s of stillness, = 25), from CHALLENGE.md: a touch,
  deflection or aerial duel without control is not a change of possession. v1 kept frozen.
- **Result (full match, no ground truth):** 225 possessions, median 10-14 s.
- **Defect found:** 0 direct H<->A switches. Alive -> D -> alive costs 25 + 4 = 29 < 50, so every
  turnover is routed through a fake stoppage. Fixed in v3.

## 2026-11-07 — Task 4: v3, no fake stoppage cheaper than a turnover

- **What:** `P_FROM_DEAD = P_SWITCH - P_TO_DEAD` (= 25), so alive -> D -> alive can never undercut
  a direct H <-> A switch; enforced by `check_no_cheap_fake_stoppage` at import. v1/v2 frozen,
  hashes unchanged. (Codex timed out on this task; implemented by the orchestrator.)
- **Result (full match, no ground truth):** 163 possessions, median 17-18 s, 21-31 direct
  switches per half (v2: 0), D share unchanged at 35-41 %, tag agreement 62-68 %.
- **Public window, bug check only:** valid, latency +0.2 frames. Not used to choose anything:
  17 possessions cannot separate methods (see EVALUATION.md).
- **Open question for v4:** direct switches look low for a full match — the 1 s control rule may
  now be too strict where the ball is invisible (41 % of frames), since missing ball = no evidence.

## 2026-11-07 — Organisers' rules clarification (overrides the repo docs)

- **New constraint:** every tuned parameter must be tuned on external data; never on the provided
  game, its video or any ground truth. Output is never checked against the video.
- **Audit of v1-v3:** no parameter was fitted. Values come from football reasoning; v2 and v3 are
  a rescaling derived from CHALLENGE.md's definition of control and a structural consistency
  constraint. Full-match diagnostics were used to detect two bugs (touch-level turnovers, phantom
  stoppages). Public-window scores were reported as bug checks and never drove a choice. The video
  was never used.
- **From now on:** parameter choices go through an external-data harness. Candidate: Metrica
  `sample-data` (3 games with synchronised tracking and events; Game 3 in the same EPTS format as
  the ATD). Its README has no formal licence ("be responsible", "acknowledge the source"), so its
  use must be confirmed with the organisers first.
