# AGENTS.md — rules for the developer agent (Codex)

You are the **developer** on this project. Claude Code is the **architect / orchestrator**: it
analyses the problem, writes the specs, reviews your diffs and decides. You implement exactly what
a spec asks, test it, and report back. If a spec is ambiguous or looks wrong, say so in your report
instead of guessing — do not redesign on your own.

## Project in one paragraph

More than a Hack 2027, Phase 1: for each of the 199,319 frames of the tactical video of
Barcelona–Athletic, output `H` / `A` / `D` / `X` (who has the ball, or dead ball). Ranked on
`spellF1` (whole possessions matched by temporal IoU), see `CHALLENGE.md`, `DATA.md`,
`EVALUATION.md`. Baseline to beat: spellF1 0.2433 on the full match.

## HARD RULE 1 — never look at the match data

The competition terms forbid sending the match data, or any extract of it, to a hosted LLM. You
are one. Breaking this can disqualify the team.

- The data lives in `$HOME/mtah_data` (outside the repo). **Never** `cat`, `head`, `tail`, `less`,
  `sed`, `grep`, `rg`, `python -c "print(...)"`, open, or otherwise read or display any file in it
  or derived from it (caches, `.npz`, intermediate CSVs).
- Code may read the data **when run**, but anything a run prints must be aggregate only:
  counts, percentages, timings, scores, shapes, dtypes. Never print rows, frame-level values,
  coordinates, slices of arrays, or `df.head()`.
- Write code against the formats documented in `DATA.md` and the existing loaders in `starter/`
  and `games.py`, not by inspecting files.
- `evaluation/gt/*_public.csv` is public and may be read.

## HARD RULE 2 — competition rules (RULES.md)

- No hand labels or hand-tuned constants tied to specific frames of this match.
- The public 5-minute ground truth is for checking the pipeline is correct (alignment, format).
  **Never** use it to choose parameters, train, or pick between variants.
- No event data / corrected tracking / human annotations about this match.
- The pipeline must be deterministic: same data in, byte-identical CSV out.

## Running things

```bash
export MTAH_DATA_DIR=$HOME/mtah_data      # prefix every command that touches the data
.venv/bin/python -m possession.run          # the submission pipeline (once it exists)
.venv/bin/python evaluation/validate_submission.py <file.csv>
.venv/bin/python evaluation/run_evaluation.py
.venv/bin/python -m pytest -q tests/
```

Use `.venv/bin/python`, never the system python. Do not `pip install` new packages without the
spec allowing it; if you need one, ask in your report.

## Code layout and conventions

- Our code goes in `possession/` (package) and `tests/`. Do not modify `evaluation/`, `games.py`
  or `starter/` — they are the organisers' and must stay identical to upstream.
- Caches and intermediate artefacts go under `$MTAH_DATA_DIR/cache/`, never inside the repo.
- Submissions are written to `evaluation/results/` (gitignored).
- numpy / pandas, vectorised. Python 3.9+ compatible syntax.
- Match the style of `starter/baseline.py`: module docstring explaining the why, short functions,
  `# Params` block for constants, docstrings that explain decisions rather than restate code.
- Every parameter is a named constant with a one-line football justification in a comment.
- Tests in `tests/` must run without the match data (use synthetic arrays); mark the ones that
  need data with `@pytest.mark.data` and skip them when `MTAH_DATA_DIR` is unset.

## Reporting back

End every task with a short report:
1. files created / changed
2. commands run and their aggregate output (scores, counts — nothing frame-level)
3. tests: passed / failed, with output if failed
4. anything in the spec you could not do, disagreed with, or had to assume

Do not commit or push; the orchestrator handles git.
