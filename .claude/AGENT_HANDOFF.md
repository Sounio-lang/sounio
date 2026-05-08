# Claude Code Agent Handoff

Start here for parallel Sounio work:

1. Read `CLAUDE_HANDOFF.md`.
2. Read `AGENTS.md`.
3. Read `.claude/PARALLEL_BLOCKER_CONTRACT.md`.
4. Read `.agent-orchestration/HANDOFF.md`.
5. Use `.claude/MEMORY_LANES.md` to pick only the memory lane needed.

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Current upstream merge target: `origin/main`.
- Historical safe branch: `integration/sounio-dev-ready-base`. Treat it as
  stale/divergent unless the human explicitly revives it; do not broad-merge
  it into current work.
- Do not trust stale Beagle branch metadata without checking `git status -sb`
  and `git rev-parse --short HEAD origin/main`.
- Three-agent coordination for this window:
  - Claude A is the integration shepherd. Claude A owns final merge authority,
    branch race checks, blocker classification, and pushes to `main`.
  - Claude B is the implementation worker. Claude B owns one compiler lane at a
    time; current candidate lane is the locked worktree commit
    `93ddd0e5 compiler: fold imported else-if chains at import time`.
  - Codex is support implementer/reviewer. Codex may inspect, validate, prepare
    handoffs, or implement explicitly assigned scoped tasks, but should not push
    `main` unless Claude A or the human delegates that merge step.
- Coordination protocol:
  - Before editing, announce `CLAIM <files>`.
  - After committing, aborting, or handing off, announce `RELEASE <files>`.
  - If two agents need the same file, Claude A decides the order.
  - Treat `self-hosted/compiler/module_frontend.sio`,
    `self-hosted/compiler/native_compile_driver.sio`, compiler resolver
    scripts, `bin/souc`, `.claude/settings.local.json`, and CI workflow files
    as serialized surfaces even under soft coordination.
- Current known worktree posture:
  - `/workspace/sounio` is the active main worktree. Leave any existing local
    dirt alone unless explicitly assigned.
  - `/workspace/sounio-native-v2-fnref-calls`,
    `/workspace/sounio-native-v2-hof-lock`, and
    `/workspace/sounio-native-v2-imported-hof-abi` are stale/read-only evidence
    worktrees. `git cherry -v origin/main <branch>` showed only `-` entries
    for these branches on 2026-05-08, so their patches appear represented on
    `origin/main`; Claude A should approve any worktree/branch deletion.
  - `/workspace/sounio/.claude/worktrees/agent-a04d29d914b22568f` is the locked
    Claude worker lane for imported else-if folding. `git cherry -v origin/main
    worktree-agent-a04d29d914b22568f` showed one live `+` commit:
    `93ddd0e5 compiler: fold imported else-if chains at import time`.
- Current dirty-state warning:
  - `.claude/settings.local.json` is locally modified; do not touch it.
  - `scripts/ci/kretikos_kaxi_lowering_gate.sh` is locally modified in the main
    worktree as of 2026-05-08; classify ownership before touching KAXI gate
    files.
  - `scripts/ci/kretikos_f64_runtime_gate.sh` is locally modified in the main
    worktree as of 2026-05-08; classify ownership before touching Kretikos
    runtime gate files.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it; if modified, do not overwrite or revert it without explicit user
  direction.
- `self-hosted/check/check.sio` has a serialized window declared in
  `.claude/check_sio_integration_window.v1.json`.
- Any blocker left for another agent must use the severity, failure class,
  evidence level, ownership, and `Next-Command` shape from
  `.claude/PARALLEL_BLOCKER_CONTRACT.md`.
- Apple native-v2 lane: Codex added `scripts/apple/apple_native_v2_ssh_gate.sh`
  as the SSH orchestration entrypoint. It runs the maintained Apple/Mach-O
  `selfhost_host_gate.sh` first. Current AArch64 native-v2 runtime attestation
  is intentionally `not_run` because full `native::codegen.sio`
  import/typecheck is dirty in this checkout; review this lane read-only unless
  coordinating ownership through `artifacts/omega/agent_handoff.log.md`.

Claude's preferred role in the parallel flow:

- read-only repository surveys
- memory lookup and synthesis
- secondary review of diffs
- documentation lookup when external docs are needed
- architectural/context mapping before Codex or another implementation agent edits

When Claude needs to hand off implementation, write the exact target files,
current branch, relevant memory lane files, and required validation commands.
