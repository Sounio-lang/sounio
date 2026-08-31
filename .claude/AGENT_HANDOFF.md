# Claude Code Agent Handoff

> **⚠️ MADAROS STATUS (current):** Madaros is green on `origin/main`.
> Do not judge checker issues from stale worktrees or prebuilt raw artifacts.
> See [`docs/MADAROS_STATUS.md`](../docs/MADAROS_STATUS.md) for sync/rebuild
> instructions and the valid proof gate (`make madaros-full-gate`).

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

> **⚠️ SOUNIO_STDLIB_PATH in a worktree (2026-08-15).** CLAUDE.md's own dev
> instructions export `SOUNIO_STDLIB_PATH` globally in the interactive pod. If
> your worktree session inherits that export, every `souc`/`madaros`/gate-script
> invocation in your worktree silently resolves the stdlib from the SHARED
> checkout (`/workspace/sounio/stdlib`), not your own worktree's copy — a
> worktree edit to `stdlib/` then measures as absent no matter what you change.
> Bit this session twice independently (a false "pub not honored in mod.sio"
> conclusion, retracted in `f0e7869765`, and the same confound hit a parallel
> lane separately). This is NOT a real CI bug — GitHub Actions runners are
> ephemeral checkouts and only set the var locally per-step
> (`madaros-prebuilt-refresh.yml`) — it is purely a worktree/dev-pod trap.
> Before trusting any `souc check`/gate result from a worktree, run
> `echo $SOUNIO_STDLIB_PATH` and confirm it points at *this* worktree's
> `stdlib/`, or `unset SOUNIO_STDLIB_PATH` and pass it explicitly per-command.

> **⚠️ SOUC_BIN does the SAME thing, for the compiler binary itself
> (2026-08-15).** The pod also exports `SOUC_BIN=/workspace/sounio/bin/souc`
> globally. `scripts/lib/resolve_souc.sh`'s `_sounio_resolve_bin()` honors a
> pre-set `SOUC_BIN` before it ever tries `$ROOT_DIR/bin/souc` relative to the
> script's own location — so `scripts/run_sio_test_suite.sh` (and anything
> else that sources `resolve_souc.sh`) silently runs against the SHARED
> checkout's compiler from a worktree, regardless of `MADAROS_RAW_BIN`. This
> one is more dangerous than the stdlib-path trap because it fails silently
> with plausible-looking numbers rather than an obvious "file not found": a
> regression check can report "identical results before/after" while never
> having touched your build at all (caught this session — see
> `CAMINHO_CRITICO_CORTADO_2026-08-14.md`'s 2026-08-15 update, which corrects
> an already-landed commit's unsupported claim). `unset SOUC_BIN` before any
> regression check in a worktree, alongside `SOUNIO_STDLIB_PATH`.

---

## LANE CLAIM + ownership-conflict — 2026-08-06

### Parallel Lane Contract

```text
Lane:            Madaros self-compilation (gen2 == gen3) — the fixed-point line
Owner:           claude-1
Base:            b260dba66e (origin/main, 2026-08-05) — 24 commits BEHIND origin/main
Worktree:        /tmp/claude-1000/-workspace-sounio/fecdd497-.../scratchpad/wt-fpw
Branch:          feat/madaros-fixed-point-line @ a807138314 (45 commits ahead)
Write-Set:       self-hosted/ir/ir.sio, self-hosted/ir/lower.sio,
                 self-hosted/check/specializer.sio,
                 self-hosted/compiler/module_frontend.sio,
                 self-hosted/compiler/main.sio, self-hosted/compiler/main_tests.sio (new),
                 scripts/ci/madaros_*_gate.sh, scripts/ci/global_aggregate_store_gate.sh,
                 scripts/lib/souc_invoke.sh, tests/known_failures/lean_single_global_*.sio
                 (89 files total vs origin/main)
Read-Set:        self-hosted/check/check.sio, self-hosted/native/*
Required-Gates:  scripts/ci/madaros_fixed_point_gate.sh (rung `check`, green),
                 scripts/ci/global_aggregate_store_gate.sh (4/4, green)
Merge-Target:    main
Known-Blockers:  BLK-20260806-madaros-fixedpoint-mainsio-overlap (below)
```

### Blocker record

```text
Blocker-ID: BLK-20260806-madaros-fixedpoint-mainsio-overlap
Status: reproduced
Severity: B2
Class: ownership-conflict
Owner: claude-1
Lane: Madaros self-compilation (gen2 == gen3)
Worktree: /tmp/claude-1000/-workspace-sounio/fecdd497-.../scratchpad/wt-fpw
Branch: feat/madaros-fixed-point-line @ a807138314
Files-Owned: see Write-Set above
Files-Read-Only: self-hosted/ir/opt_cleanup.sio, self-hosted/ir/tailcall.sio
Do-Not-Touch: self-hosted/compiler/main.sio until this record is resolved
Repro: git worktree add --detach <tmp> probe/ir-soa-phase0
       && cd <tmp> && git merge --no-commit --no-ff feat/madaros-fixed-point-line
Observed: 10 conflicted files, ~110 conflict hunks. 81 of them are in
          self-hosted/compiler/main.sio alone. Measured 2026-08-06 against
          probe/ir-soa-phase0 @ 6b0698ce22.
Expected: disjoint write sets, per Ownership Rule 1 (one active writer per file)
Acceptance-Gate: a clean `git merge` of both lanes, then
                 scripts/ci/madaros_fixed_point_gate.sh at rung `check`
Evidence-Level: E1
Evidence: trial merge above; per-file hunk counts recorded in this entry
Fallback-Path: land this lane WITHOUT the main.sio test-suite extraction
Legacy-Kept: yes — no test assertion removed (1163 test fns before and after)
LLM-Offload: not-required
Next-Action: human decision on merge order (see below)
```

### What actually overlaps, measured

`probe/ir-soa-phase0` (tip `6b0698ce22`, last commit 2026-08-06 15:25Z, NOT merged
to main, merge-base 2026-07-25) and this lane both write
`ir/ir.sio`, `ir/lower.sio` and `compiler/main.sio`. That sounds total. It is not:

| file | conflict hunks |
|---|---|
| `compiler/main.sio` | **81** |
| `ir/opt_cleanup.sio` | 13 |
| `compiler/module_frontend.sio` | 7 |
| `ir/ir.sio` | **2** |
| `ir/lower.sio` | **2** |
| 5 others | 1 each |

Of 89 files this lane changes, 10 conflict. The two lanes turn out to work on
DIFFERENT PARTS of the same files: `probe/ir-soa-phase0` is in the optimizer
(`compact_nops`, float markers, the `-O` miscompile, #1667); this lane is in
lowering, capacities and diagnostics. `ir/lower.sio` — 3600 changed lines on
their side, ~700 on ours — collides in two hunks.

**The wall is one change, not the lane:** the extraction of main.sio's 1163-test
self-test suite into `compiler/main_tests.sio` (main.sio 28525 -> 9408 lines).
It is also the most re-doable change here — it is mechanical, its inputs are
recorded (commit `1241ac359e`), and re-running it after the other lane lands
costs a script, not a rediscovery.

### Recommendation (human decision required)

Land this lane WITHOUT `compiler/main.sio`, and redo the extraction afterwards.
That drops the conflict from ~110 hunks to ~29 across 9 files, all of them
ordinary review-sized. Everything this lane found stays landable:

- module-level globals are `ItemFn`, and cross-module DCE was deleting them all
  (`541536f777`) — a silent-wrong-code fix, independent of main.sio
- `IR_MAX_INSTRS` 3072 -> 4096, required by `lex_source_to_globals` at 3269
- `StructFieldEntry` name interning (~18.6 KB -> ~2.1 KB per layout entry)
- the IR slot census, the error-site ledger, the unresolved-identifier ledger
- two checked-in seed-defect reproductions + `global_aggregate_store_gate.sh`

The cost of the alternative — merging main.sio too — is 81 hunks in a file both
lanes restructured, where a wrong resolution is a silent miscompile rather than
a build error.


### Active-lane map over `self-hosted/ir` + `compiler` — surveyed 2026-08-06

Surveyed because this lane worked ~15 hours in `ir/lower.sio` without one, and
`#1649` — which its own commits reference — had two PRs open against it since
2026-08-05. The survey is the deliverable; the numbers below are measured trial
merges, not impressions.

**The three "IR lanes" are one lane, stacked.** Each contains the previous:

```
refactor/ir-instruction-arena (#1650)  ⊂  refactor/ir-arena-step2 (#1651)  ⊂  probe/ir-soa-phase0
      7dde0d403d, 08-05                       6c32908852, 08-05                94bcff71a3, 08-06
```

`probe/ir-soa-phase0` is the tip, not a rival. (`6c32908852` is also where this
lane took `tests/known_failures/lean_single_global_aggregate_store.sio` from —
the file only, never the commit, which is why no history was duplicated.)

**Conflict against `feat/madaros-fixed-point-core` (#1672), measured:**

| lane | files | hunks | where |
|---|---|---|---|
| #1650 `refactor/ir-instruction-arena` | **0** | **0** | merges clean |
| #1651 `refactor/ir-arena-step2` | 8 | 14 | `module_frontend.sio` 6, `ir/lower.sio` 2, six files at 1 |
| `probe/ir-soa-phase0` | 10 | 30 | as above plus `ir/opt_cleanup.sio` 13 |

`ir/ir.sio` does NOT conflict with #1651: this lane's `StructFieldEntry`
interning and their arena work sit in different regions of the file.

**Twelve other open PRs touch this write-set**, all last active 07-17..07-29 and
none in the past week: #1527, #1500, #1508, #1531, #1501, #1493, #1421, #1339,
#1069, #1605. #1527 is the self-parse/visibility/Box lane this lane's own plan
said to re-slice rather than merge.

### Proposed landing order

```
1. #1650  refactor/ir-instruction-arena   — base of the stack, 0 conflicts with anything
2. #1672  feat/madaros-fixed-point-core   — 0 conflicts with #1650
3. #1651  refactor/ir-arena-step2         — absorbs 14 hunks
4.        probe/ir-soa-phase0             — tip, absorbs the rest
```

Rationale, and the one place it is arguable:

- #1650 first because it is free — it conflicts with nothing measured here.
- #1672 second because its load-bearing fix is **silent wrong code**:
  cross-module DCE deleted every module-level global, and the emitted binary
  printed nothing rather than failing. That class should not wait behind a
  refactor.
- #1651 third. Note it **removes** `instrs: [IrInstr; 4096]`, the field #1672
  grows from 3072. The two agree on the number, so this is supersession and not
  contradiction. **If #1651 is landed before #1672 instead**, drop the
  `IR_MAX_INSTRS` hunks from #1672 (21 type positions across six files, the
  constant, the struct field, eight initialisers) — nothing else in #1672
  depends on them.
- The finding behind that raise survives the representation and argues FOR the
  arena: `lex_source_to_globals` needs 3269 instructions, and it never appeared
  in the corpus census that justified the 3072 cap, because that census does not
  contain this compiler's own 120-module closure. A fixed cap chosen from the
  corpus will keep being wrong for self-compilation.

Owner of #1650/#1651/`probe/ir-soa-phase0` has not been contacted. This map is
on the #1672 branch; relaying it is a human action.


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
