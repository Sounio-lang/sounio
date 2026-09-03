# Kimi Agent Handoff

This repo-local `.kimi` folder is for Sounio task handoffs only. Do not put
credentials here; global Kimi credentials live outside the repo.

Read before Kimi starts a Sounio task:

1. `CLAUDE_HANDOFF.md`
2. `AGENTS.md`
3. `.agent-orchestration/HANDOFF.md`
4. Any task-specific briefing under `artifacts/kimi-briefing/`

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch: `integration/sounio-dev-ready-base`.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it; if modified, do not overwrite it unless the user explicitly
  assigns that file to Kimi.
- Check `artifacts/omega/agent_handoff.log.md` before editing shared surfaces.

Kimi's preferred role in the parallel flow:

- large-context implementation against a precise briefing
- broad single-file compiler changes after Codex/Claude define the target
- second-pass implementation when tests and constraints are explicit

Useful existing briefings:

- `artifacts/kimi-briefing/BUG3_IMPL_DISPATCH_FIX.md`
- `artifacts/kimi-briefing/PHASE1_MODULE_RESOLVER.md`
- `docs/MADAROS_STATUS.md` — fleet coordination note (current: Madaros is green on origin/main; valid proof gate `make madaros-full-gate`)

When returning work, include changed files, checks run, unresolved blockers, and
whether any assumptions came from old VM paths or current `/workspace/sounio`.
