# Claude Code Agent Handoff

> **⚠️ WORKSPACE STABILITY (2026-05-29).** The pod was recycled twice by the k8s
> liveness probe under CPU saturation — four agents were running directly on the
> shared `/workspace/sounio` checkout, each firing a full `souc main.sio` bundle
> build (15-min load ~153 on 64 cores). Two non-negotiable rules:
>
> 1. **One worktree per agent.** Never run a second agent on `/workspace/sounio`
>    directly — that checkout is the integration-shepherd lane only. Each worker
>    gets its own worktree (lane table below). Ceiling: **≤2 agents doing
>    compiler work at once** on this pod.
> 2. **Serialize heavy builds** through `scripts/dev/souc-build-lock.sh`
>    (`souc main.sio` / `lean_single.sio` / `make build`). Bare full builds are
>    what saturate CPU and trip the probe. Cheap `souc check` is exempt.

## 6-Agent Lane Activation — 2026-05-10T13:35Z

**Authority**: human-approved at 2026-05-10 (this commit).
**Companion docs**: `.agent-orchestration/coordination/6_lane_assignment.md`
(full matrix), `.claude/PARALLEL_BLOCKER_CONTRACT.md` (blocker shape).
**Canonical live log**: `artifacts/omega/agent_handoff.log.md`.

**Active lanes** (see companion doc for file-set + build target details):

| # | Lane                          | Owner       | Branch                              | Worktree                                  |
|---|-------------------------------|-------------|-------------------------------------|-------------------------------------------|
| 1 | golden-recapture              | Claude #1   | `coord/lane-1-golden-recapture`     | `/workspace/sounio-lane-1-goldens`        |
| 2 | dissertation-evidence         | Codex #1    | `coord/lane-2-dissertation-evidence`| `/workspace/sounio-lane-2-dissertation`   |
| 3 | paper-168-cohomological       | Claude #2   | `coord/lane-3-paper-168`            | `/workspace/sounio-lane-3-paper168`       |
| 4 | nv2-compiler-hardening        | Codex #2    | `coord/lane-4-nv2-hardening`        | `/workspace/sounio-lane-4-nv2`            |
| 5 | python-extermination phase 5  | Codex #3    | `coord/lane-5-phase5-recognizer`    | `/workspace/sounio-lane-5-phase5`         |
| 6 | integration-shepherd (merge)  | Claude A    | `main`                              | `/workspace/sounio` (canonical)           |

**Out-of-scope lanes** (continue independent, NOT part of the 6):
- `garden/above-stars` (existing, separate)
- `cursor/quaternionic-ssm-88c0` (existing remote, Cursor agent)
- `worktree-agent-a04d29d914b22568f` (locked Claude worker, imported else-if folding)

**Merge order when multiple lanes are PR-ready**: 1 → 4 → 5 → 2 → 3.
Lane 4 must serialize against Lane 1 on `bin/souc-linux-x86_64`.
Lanes 2/3/5 are file-disjoint and may land in any order.

**Live CLAIMs / RELEASEs** (most recent first):

```text
LANE-1 RELEASE 2026-05-10T13:48Z claude-1 tests/golden/kaxi_ptx/**
  blocker: BLK-20260510-lane1-golden-drift closed by PR #95
  evidence: kaxi_ptx_golden_gate.sh = 318 PASS / 0 FAIL / 0 MISSING
  binary-token: bin/souc-linux-x86_64 NOT consumed; Lane 4 may claim next if needed

LANE-1 CLAIM 2026-05-10T13:35Z claude-1 tests/golden/kaxi_ptx/** bin/souc-linux-x86_64{,.sha256,.sig}
  blocker: BLK-20260510-lane1-golden-drift  severity:B1  class:gate-regression
  evidence: kaxi_ptx_golden_gate.sh = 209/52 FAIL/57 MISSING vs 318 nominal
  next-command: bash scripts/ci/kaxi_ptx_capture.sh && bash scripts/ci/kaxi_ptx_golden_gate.sh
```

**Closed Blocker (lane 1)**:

```text
Blocker-ID: BLK-20260510-lane1-golden-drift
Status: closed by PR #95
Severity: B1
Class: gate-regression
Owner: Claude #1
Lane: 1 (golden-recapture)
Worktree: /workspace/sounio-lane-1-goldens
Branch: coord/lane-1-golden-recapture
Evidence: tests/golden/kaxi_ptx/default/exit_only.ptx golden last touched
  3f3af0cd (Phase L, 2026-05-08) declares `.reg .b32 %r<8>`, but current
  `bin/souc` against current `self-hosted/gpu/kaxi_to_ptx.sio` produces
  `.reg .b32 %r<260>` and `.reg .f32 %f<260>`. 38 commits to
  kaxi_to_ptx.sio between Phase L and HEAD did not regenerate goldens.
Reproduction: bash scripts/ci/kaxi_ptx_golden_gate.sh in any worktree
  off origin/main HEAD (8a1a6fa2). Result: PASS=209 FAIL=52 MISSING=57.
Resolution: PR #95 regenerated tests/golden/kaxi_ptx/**. Post-merge
  kaxi_ptx_golden_gate.sh reports PASS=318, FAIL=0, MISSING=0.
Acceptance: satisfied.
Evidence-Level: E3 (gate-bound)
```

**Per-lane init**: each agent runs the checklist in
`.agent-orchestration/coordination/6_lane_assignment.md#per-lane-initialization-checklist`
on session start. Branch flips to verify via
`git branch --show-current` before any edit.

---

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

---

## Collision Resolution — 2026-05-20 (post-cluster-maintenance lane recovery)

Context: cluster maintenance + new instructions caused several Claude lanes to
drop and resume in a degraded state ("zuadas"). Two branch collisions were
identified and resolved. This section is the current truth; the 6-lane matrix
above (dated 05-10) is STALE — do not treat it as live.

### Collision 1 — GPU SPIR-V/Vulkan: `kretikos/top-tier-runtime` vs `sounio-pure/python-extermination-phase7`

Both branch from merge-base `982963e09`. Both create
`self-hosted/gpu/kretikos_emit_spirv.sio`,
`scripts/gpu/kretikos_spirv_vulkan_storage_vec_add.c`, and
`scripts/ci/kretikos_spirv_vulkan_storage_semantic_baseline_gate.sh`
with DIVERGENT content (`kretikos_emit_spirv.sio` differs 356/323 lines).

Evidence (reflog, not file-count):
- `kretikos/top-tier-runtime`: clean lineage, no reset, comprehensive superset
  (+27 commits, 93 files: SPIR-V emitter + Vulkan dispatch + cross-backend
  CUDA/Metal + MultiPL-E + audit). Last commit 05-19 20:07. No upstream, not
  checked out in any worktree.
- `sounio-pure/python-extermination-phase7`: reflog shows
  `reset: moving to origin/main` then cherry-picks, originally
  `Created from HEAD` at `50e0ebfe5` (a kretikos commit). This is the
  POST-CRASH PARTIAL REBUILD — it re-did a SPIR-V subset divergently 6h later
  (05-20 01:58). Pushed to origin; checked out at /workspace/sounio.

RESOLUTION:
- **`kretikos/top-tier-runtime` is the canonical GPU lane** (stable superset).
- phase7's 3 GPU commits (`a576a4ac0`, `63b9bda7e`, `dde3a8ac4`) are a
  post-crash divergent reimplementation — DO NOT merge them to main; they are
  superseded by kretikos. NOT deleted/force-pushed (branch is pushed to origin;
  rewriting would disrupt origin trackers per feedback_workspace_branch_flips).
- phase7's one UNIQUE non-GPU commit `e869515c4` (heredoc-kill in
  sinkhorn16 gate, disjoint from the GPU conflict) is PRESERVED on new branch
  **`chore/sinkhorn16-heredoc-kill`** (off main, 1 clean commit) — land as a
  tiny standalone PR.
- PENDING VERIFICATION: the deciding gate
  `scripts/ci/kretikos_spirv_vulkan_storage_vec_add_gate.sh` could NOT be run
  here — `spirv-dis` is missing in this environment (cluster maintenance).
  Re-run on a spirv-tools-equipped node to confirm kretikos's gate passes
  before retiring phase7's GPU work permanently. Canonical call is reflog-based,
  not execution-based.

### Collision 2 — Park-Miller/CUDA duplicate: `claude-2/wip-2026-05-19` vs `sounio-pure/r2-1-park-miller`

BYTE-IDENTICAL: same 11 SHAs, empty diff, shared tip `10d69fc4`
(Merge PR #165 from `sounio-pure/r2-1-park-miller`).

RESOLUTION:
- **PR #165 is the single canonical landing path** for this 11-commit
  Park-Miller RNG + epistemic-CUDA-runtime work. Do NOT double-merge.
- `sounio-pure/r2-1-park-miller` is canonical (PR source).
- `claude-2/wip-2026-05-19` is a redundant alias at the identical SHA, checked
  out LIVE at /workspace/sounio-claude-2 (tracks origin/main). NOT deleted
  (checked out by a live agent; git would refuse, and removal would disrupt
  that agent). It will retire naturally when that agent advances; until then it
  carries no unique commits.

---

## Active agent lanes (2026-06-04) — modular-checker / parser lane claim

Three concurrent lanes; this entry claims the **modular-checker / parser** lane to
avoid file + build-lock collisions.

- **This lane (modular checker → parser):** closure/HOF unlock LANDED, **PR #235**
  (`check/closure-hof-triple-e008`, sound +7, modular checker only). **Next lever:
  async `spawn`/`await` parser cluster** (11 tests: async_basic/channels/join/sleep/
  spawn/...) — INVESTIGATED, DEFERRED: async is a parser+checker FEATURE (needs
  check.sio semantics for spawn/await builtins = i128 territory; parser-only = +0,
  the const/kernel/extern double-block trap). **Pivoted to PARSE-TO-PASS items**:
  parser productions that flip FAIL→PASS with parser-only changes (tuple-let
  `let (a,b)=`, struct/while-for patterns, turbofish, match patterns) — checker
  already handles the resulting AST. `self-hosted/parser/*` only, zero check.sio
  overlap with i128.
- **i128 / arbitrary-width-int lane** (`feat/i128-modular`): owns `check/types.sio`,
  `check/compat.sio`, `check/check.sio` (int-width additions) + native codegen
  (`native/codegen_x86_linux.sio`, `native/encode.sio`, `ir/ir.sio`). I avoid these.
- **SRET / source→ELF lane** (`sret-fix-probe`): owns `self-hosted/ir/*` (lower.sio,
  IR lowering, for-loops). I avoid these.

Build discipline: all heavy `souc main.sio` builds via `scripts/dev/souc-build-lock.sh`
(never two at once). Census gotcha: `chmod +x` the mc ELF before running.
