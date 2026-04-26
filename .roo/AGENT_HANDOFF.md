# Roo Agent Handoff

Read before selecting a Roo mode:

1. `CLAUDE_HANDOFF.md`
2. `AGENTS.md`
3. `.agent-orchestration/HANDOFF.md`
4. `.roomodes`

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch: `integration/sounio-dev-ready-base`.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it; if modified, avoid touching it unless the task explicitly targets
  it and ownership is clear.
- Shared locks live in `artifacts/omega/agent_handoff.log.md`.

Mode routing:

- `sounio-compiler-dev`: compiler source and docs.
- `sounio-epistemic-units`: `stdlib/epistemic`, `stdlib/units`, epistemic docs.
- `sounio-gpu-hpc`: GPU backends, GPU stdlib, interop, GPU examples.
- `sounio-test-validation`: tests, benchmarks, scripts, artifacts.
- `sounio-docs-writer`: docs/spec/website/examples.
- `sounio-webmaster`: website/brand surfaces.
- `sounio-stdlib-expert`: stdlib source, stdlib docs, stdlib tests.

Use the mode's file restrictions, but still honor the shared handoff protocol.
