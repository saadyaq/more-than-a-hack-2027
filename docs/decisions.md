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
