# Beagle Context Handoff

This folder contains useful workspace metadata, but it is not branch authority.

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch verified by Git during orchestration setup:
  `integration/sounio-dev-ready-base`.
- Some Beagle context files still report branch `main`; agents must verify with
  `git branch --show-current` before acting.
- Beagle status may be `local-only` or `degraded`; use repo files and Git as
  source of truth.

Shared startup packet:

1. `CLAUDE_HANDOFF.md`
2. `AGENTS.md`
3. `.agent-orchestration/HANDOFF.md`
4. `.beagle/context/workspace-subagents.json`

Current subagent metadata says the live role is `sounio-core` with role tag
`compiler-runtime`. Treat this as workspace orientation, not edit permission.
