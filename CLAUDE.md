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
   `approval-policy: never`. Model for coding tasks: try `gpt-6.1-sol` first (it was once
   rejected for this ChatGPT account; retest), else `gpt-6.1-terra`, else `gpt-6-luna` for small
   tasks — `gpt-6-astra` is overkill for implementation and burns the usage limit. Reasoning
   effort `medium`. Prefer `codex exec ... -o <report>` run in the background over the MCP tool:
   the MCP call aborts after ~40 min without progress. Codex's sandbox cannot write
   `~/mtah_data/cache`, so cache-building runs are done by Claude. Continue the same thread with `codex-reply` for fixes.
3. Review: `git diff`, run the tests and `validate_submission.py` yourself. Never trust the report
   alone.
4. Record the decision in `docs/decisions.md` (what, why, result).

## Data rule

Same as AGENTS.md hard rules 1 and 2: the match data is in `~/mtah_data` and must never be read
or printed into the conversation; only aggregate outputs. Per the organisers' clarification email
(Nov 2026), every tuned parameter is tuned on **external data** only — never on the provided game,
its video or any ground truth — and our output is never checked against the video.
