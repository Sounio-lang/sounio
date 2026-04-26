# Claude Code Agent Handoff

Start here for parallel Sounio work:

1. Read `CLAUDE_HANDOFF.md`.
2. Read `AGENTS.md`.
3. Read `.agent-orchestration/HANDOFF.md`.
4. Use `.claude/MEMORY_LANES.md` to pick only the memory lane needed.

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch: `integration/sounio-dev-ready-base`.
- Do not trust stale Beagle branch metadata that says `main`.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it; if modified, do not overwrite or revert it without explicit user
  direction.
- `self-hosted/check/check.sio` has a serialized window declared in
  `.claude/check_sio_integration_window.v1.json`.

Claude's preferred role in the parallel flow:

- read-only repository surveys
- memory lookup and synthesis
- secondary review of diffs
- documentation lookup when external docs are needed
- architectural/context mapping before Codex or another implementation agent edits

When Claude needs to hand off implementation, write the exact target files,
current branch, relevant memory lane files, and required validation commands.
