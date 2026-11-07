# CLAUDE.md

## Roles

- **Claude Code — architect / orchestrator.** Analyses the problem, designs the method, writes
  specs for each step, delegates implementation to Codex, reviews the diff, runs validation, and
  decides what comes next. Writes code itself only for small fixes.
- **Codex (MCP `codex`) — developer.** Implements specs, writes tests, reports back. Its rules are
  in `AGENTS.md`, which Codex reads automatically.

## Delegation loop

1. Write the spec: goal, files, interfaces, acceptance criteria, what not to touch.
2. Call `mcp__codex__codex` with `cwd` = repo root, `sandbox: workspace-write`,
   `approval-policy: never`, `model: gpt-6-astra` (the config default `gpt-6.1-sol` is rejected
   for this ChatGPT account). Codex's sandbox cannot write `~/mtah_data/cache`, so cache-building
   runs are done by Claude. Continue the same thread with `codex-reply` for fixes.
3. Review: `git diff`, run the tests and `validate_submission.py` yourself. Never trust the report
   alone.
4. Record the decision in `docs/decisions.md` (what, why, result).

## Data rule

Same as AGENTS.md hard rule 1: the match data is in `~/mtah_data` and must never be read or
printed into the conversation. Only aggregate outputs. Tuning on the public 5 minutes is
forbidden; choose parameters from football reasoning and self-consistency.
