# Cursor Agent Handoff

Read before interactive edits:

1. `CLAUDE_HANDOFF.md`
2. `AGENTS.md`
3. `.agent-orchestration/HANDOFF.md`
4. `.cursor/rules/syntax.mdc`
5. `.cursor/rules/effects.mdc`
6. `.cursor/rules/types.mdc`
7. `.cursor/rules/patterns.mdc`

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch: `integration/sounio-dev-ready-base`.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it; if modified, do not overwrite, autoformat, or revert it without
  explicit user direction.
- Use `artifacts/omega/agent_handoff.log.md` for shared locks.

Cursor-specific caution:

Sounio is not Rust. Follow the `.cursor/rules/*` files before editing `.sio`:

- no semicolons
- `var`, not `let mut`
- `&!`, not `&mut`
- no Rust macros
- no closure literals
- explicit effects
