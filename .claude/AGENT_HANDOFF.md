# Claude Code Agent Handoff

## Session summary (2026-05-10, Claude #1)

- Phase J wired into umbrella as `cbe6716e` (compile-time confidence gate,
  dissertation contribution #2). CPU-only gate, ~1s. rc=0 confirmed.
- Phase Y wired into umbrella as `2b6b5b1a` (GUM variance propagation,
  dissertation contribution #1). Self-skips without libcuda. rc=0 confirmed.
- Shell fallback added to `emit_summary_json` (`4def6b5e`) for aggregator
  SRET Heisenbug — umbrella now ships pass/fail signal regardless.
- **Native-compiler 5-gate FAIL — root cause found and fixed.** After
  `dfe0894a` (S-A defensive backstop), parse_stmt_ir's call to
  `refuse_let_array_lit` added 7 IR ops, pushing the per-function
  `V2_UFN_OPS[2048]` buffer over its silent-drop threshold during
  self-compile. Dropped IR ops produced incomplete x86-64 → SIGSEGV in
  stage1's `compile_user_fn` for the first user fn. Bisected: pre-`dfe0894a`
  source self-compiles (md5=ccfbf10a…), post-`dfe0894a` source
  self-compiles to a binary that crashes on its own self-compile
  (md5=f821f5…, rc=139). Fix: bumped V2_UFN buffers 2048→4096 (12 arrays
  + 26 bounds checks). Self-host fixed-point restored: stage1==stage2==stage3
  (md5=c54005c7…, 405,968 bytes).

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
