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
