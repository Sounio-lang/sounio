# Codex Agent Handoff

Start every non-trivial Sounio task with:

1. `git branch --show-current`
2. `git status --short`
3. `CLAUDE_HANDOFF.md`
4. `AGENTS.md`
5. `.agent-orchestration/HANDOFF.md`

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch: `integration/sounio-dev-ready-base`.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it; if modified, treat it as user/parallel-agent work unless proven
  otherwise.
- Check `.claude/check_sio_integration_window.v1.json` before touching
  `self-hosted/check/check.sio`.
- Use `artifacts/omega/agent_handoff.log.md` for locks on shared/high-risk files.

Codex's preferred role in the parallel flow:

- implementation
- file creation
- surgical refactors
- test harness wiring
- validation-path work
- small reversible commits

Before editing shared files, append a lock entry to
`artifacts/omega/agent_handoff.log.md`. Release it with checks and status.
