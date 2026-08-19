# 6-Agent Lane Assignment (active — 2026-05-10)

Status: **active**. PR #94 activated the overlay; PR #95 closed the Lane 1
golden-recapture blocker.
Author: Claude #1 (this session, dissertation-examples branch).
Companion contract: `.claude/PARALLEL_BLOCKER_CONTRACT.md`.
Companion live state: `artifacts/omega/agent_handoff.log.md`.
Claude-private mirror: `.claude/AGENT_HANDOFF.md`.

## Goal

Coordinate 3 Claude + 3 Codex sessions running in parallel against a single
`origin/main`, so that:

1. Each agent edits a **disjoint file-set** (file-level ownership, not just
   branch separation — branches alone do not prevent serialized-surface
   collisions when two PRs touch `bin/souc` in the same window).
2. Each lane has a **single, verifiable build target** that must pass
   before that lane proposes a PR.
3. **Merge order is deterministic** when multiple PRs are green
   simultaneously, so re-bases don't churn.
4. Branch flips (per `feedback_workspace_branch_flips.md`) and dirty-WT
   leaks (per the 202-line WIP incident) are made impossible-by-construction
   via dedicated worktrees.

## Roles (mapped to existing `reference_three_agent_protocol.md`)

The existing 3-agent protocol (Claude A=push owner, Claude B=implementer,
Codex=support) extends naturally to 6:

- **Claude A** — Integration shepherd (Lane 6). One-of-a-kind role: final
  merge authority, branch-race tiebreak, blocker classification, runs the
  umbrella gate on `main` after every merge. Does NOT do feature work.
- **Claude #1, Claude #2** — Implementation lanes 1 and 3 (compiler
  goldens + math/paper). One lane each, no cross-lane edits.
- **Codex #1, Codex #2, Codex #3** — Implementation lanes 2, 4, 5
  (dissertation evidence + N-v2 hardening + python-extermination Phase 5).
  One lane each.

## Lane assignments

Each lane has: owner role · branch · worktree · disjoint file-set · build
target · merge prerequisites. Lanes are numbered to define merge order when
multiple are green at once: lower number lands first.

### Lane 1 — `golden-recapture` (Claude #1, this session)

**Status**: complete. PR #95 closed
`BLK-20260510-lane1-golden-drift` after regenerating K-AXI PTX goldens:
`kaxi_ptx_golden_gate.sh` reports 318 PASS, 0 FAIL, 0 MISSING.

**Why first**: This lane opened with a `gate-regression` (B1). At
activation, `kaxi_ptx_golden_gate.sh` was RED at 209/318 PASS because
38 commits to `kaxi_to_ptx.sio` since Phase L (`3f3af0cd`) had drifted
from goldens.

- **Branch**: `coord/lane-1-golden-recapture`
- **Worktree**: `/workspace/sounio-lane-1-goldens`
- **File-set (own + edit)**:
  - `tests/golden/kaxi_ptx/**`
  - `bin/souc-linux-x86_64`, `bin/souc-linux-x86_64.sha256`,
    `bin/souc-linux-x86_64.sig`
- **File-set (read-only, no edit without RELEASE token)**:
  - `self-hosted/gpu/kaxi_to_ptx.sio` (the source the goldens are
    captured from)
- **Procedure**:
  1. CLAIM in `artifacts/omega/agent_handoff.log.md`.
  2. Verify current `bin/souc` can compile current `kaxi_to_ptx.sio`
     without error: `bin/souc check self-hosted/gpu/kaxi_to_ptx.sio`.
  3. If souc rebuild needed: `bash scripts/ci/lean_single_fixed_point_gate.sh`
     to confirm self-host fixed-point still green; rebuild only if
     fixed-point is broken or souc rejects current source.
  4. Run `bash scripts/ci/kaxi_ptx_capture.sh` to regenerate goldens.
  5. Run `bash scripts/ci/kaxi_ptx_golden_gate.sh`. The May 2026
     closeout target was 318/318 PASS, 0 FAIL, 0 MISSING. That
     number is stale. Live measurement 2026-08-18 (#1915): **0/318**,
     rc=1, 80 s. Do not treat 318/318 as current health.
  6. Diff the regenerated goldens vs HEAD: if huge diff, ROLLBACK and
     classify as `compiler-semantics` Blocker (souc producing wrong PTX,
     not goldens being stale).
  7. Commit goldens; PR title `[lane-1] regenerate kaxi_ptx goldens vs
     post-Phase-Y emitter source`.
- **Build target (May 2026 closeout, stale):** `kaxi_ptx_golden_gate.sh`
  rc=0, `PASS: 318`, `FAIL: 0`, `MISSING: 0`. Live 2026-08-18 (#1915):
  rc=1, 0/318 PASS. Do not use 318/318 as a current pass criterion.
- **Merge status**: merged via PR #95.

### Lane 2 — `dissertation-evidence` (Codex #1)

**Why**: Just had 6 PRs in 30 min (PRs #88–93). High cadence, well-scoped
fixtures. Owns the dissertation evidence story.

- **Branch**: `coord/lane-2-dissertation-evidence`
- **Worktree**: `/workspace/sounio-lane-2-dissertation`
- **File-set (own + edit)**:
  - `scripts/ci/dissertation_pbpk_suite_gate.sh`
  - `stdlib/darwin_pbpk/validation/**`
  - `stdlib/darwin_pbpk/release/**`
  - `tests/run-pass/rapamycin_*.sio`
  - `tests/run-pass/haloperidol_*.sio`
  - `tests/run-pass/d2_*.sio`
  - `tests/run-pass/des_*.sio`
- **Build target**: `bash scripts/ci/dissertation_pbpk_suite_gate.sh`
  returns rc=0, all configured tests PASS.
- **Merge prerequisites**: Build target green + umbrella green.

### Lane 3 — `paper-168-cohomological` (Claude #2)

**Why**: PR #92 just landed §7 + cocycle k=4 enumeration. The
`Revised Open Question 1` opened by §7 explicitly asks for k=5,6 data.
A second data point is the next computational step.

- **Branch**: `coord/lane-3-paper-168` (rebase off existing
  `paper-168-cohomological` once #92 is in main)
- **Worktree**: `/workspace/sounio-lane-3-paper168`
- **File-set (own + edit)**:
  - `examples/cocycle_*.sio`
  - `examples/phi_fano_*.sio`
  - `examples/*168*.sio` *except* the Codex-assigned ones
    (`examples/jordan_168_hunt.sio`, `examples/octonion_168_associators.sio`
    are stable; do not edit without claim)
  - `docs/papers/main/168-theorem.typ`
  - `docs/papers/main/168-refs.yml`
  - `docs/papers/main/168-binary-norm-proof.typ`,
    `docs/papers/main/168-binary-norm-refs.yml`
- **Build target**: `bin/souc check examples/cocycle_subspace_168.sio`
  rc=0, plus `bin/souc compile examples/cocycle_subspace_k5.sio -o /tmp/k5
  && /tmp/k5` produces a `PASS` final line.
- **Merge prerequisites**: Build target green + umbrella green.
- **Out of scope**: Compiler edits, native-v2, K-AXI emitter.

### Lane 4 — `nv2-compiler-hardening` (Codex #2)

**Why**: Multiple `codex/native-v2-*` branches are open in flight; this
lane consolidates them and owns the N-v2 frontend close-out (run-pass
285→fewer fail; M1.2 surface area).

- **Branch**: `coord/lane-4-nv2-hardening`
- **Worktree**: existing `/workspace/sounio-native-v2-hof-lock` or new
  `/workspace/sounio-lane-4-nv2`
- **File-set (own + edit)**:
  - `self-hosted/native-v2/**`
  - `self-hosted/compiler/native_compile_driver.sio` (**SERIALIZED**;
    must hold an `artifacts/omega/agent_handoff.log.md` CLAIM before editing)
- **File-set (coordinated, RELEASE token required)**:
  - `bin/souc-linux-x86_64` — coordinate with Lane 1. Rule: Lane 1 holds
    the binary token until its goldens land; Lane 4 takes the token next
    if it needs to ship a new compiler binary.
  - `self-hosted/compiler/lean_single.sio` — needs claim window per
    `feedback_lean_single_features.md`.
- **Read-only**:
  - `tests/run-pass/**` (fixtures; only Lane 2 edits dissertation
    fixtures, only Lane 3 edits paper-168 fixtures, neither is in
    `tests/run-pass/`)
- **Build target**: `bash scripts/ci/native_v2_serious_track_gate.sh`
  rc=0 + `bash scripts/ci/lean_single_fixed_point_gate.sh` rc=0 + N-v2
  run-pass fail count strictly less-than baseline-285.
- **Merge prerequisites**: Build target green + umbrella green + Lane 1
  has merged or explicitly released the bin/souc token.

### Lane 5 — `phase-5-kaxi-recognizer` (Codex #3)

**Why**: Largest remaining python heredoc in `bin/kretikos` (~300 LoC
regex at line 2137). Phase 5 of the python-extermination plan that
landed in 5.A–5.G + Phase 1–4 already.

- **Branch**: `coord/lane-5-phase5-recognizer`
- **Worktree**: `/workspace/sounio-lane-5-phase5`
- **File-set (own + edit)**:
  - `bin/kretikos` — but ONLY between markers
    `### kaxi-recognize-lower-source begin/end` (Lane 5 introduces these
    markers and does not edit outside them)
  - `self-hosted/gpu/kretikos_kaxi_lower_source_recognize.sio` (NEW)
  - `tests/golden/kretikos_kaxi_lower_source/**` (NEW)
- **File-set (read-only)**:
  - `stdlib/regex/lib.sio` (open a Blocker if a deficiency surfaces;
    do not edit)
  - `self-hosted/gpu/kretikos_kaxi_validate_evidence.sio`,
    `self-hosted/gpu/kretikos_json_emit.sio`,
    `self-hosted/gpu/kretikos_kaxi_asm_summary.sio` (driver patterns;
    reference only)
- **Build target**: `bash scripts/ci/kretikos_kaxi_lowering_gate.sh`
  rc=0 + new `tests/golden/kretikos_kaxi_lower_source/` gate green
  (golden-first capture before driver work).
- **Merge prerequisites**: Build target green + umbrella green.

### Lane 6 — `integration-shepherd` (Claude #3 = Claude A)

**Why**: Single point of merge authority prevents two simultaneous PRs
that pass their own builds but conflict on `main`'s umbrella gate (the
post-merge state is the only state users actually run against).

- **Branch**: `main` (no feature work; merge commits and coordination-log
  edits only)
- **Worktree**: `/workspace/sounio` (canonical, kept clean)
- **File-set (own + edit)**:
  - `artifacts/omega/agent_handoff.log.md` (live coordination state)
  - `.claude/AGENT_HANDOFF.md` (Claude-private mirror)
  - merge commits to `main` only
- **Forbidden**: feature edits, fixture edits, prose edits.
- **Procedure (per merge)**:
  1. Pull latest `main`.
  2. Verify `git status` clean.
  3. Run umbrella gate: `bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`.
  4. If green: merge the next PR in lane-number order.
  5. Re-run umbrella gate post-merge. If RED: revert merge, mark
     PR `merge-blocked` with Blocker-ID, hand back to lane owner.
  6. Update `artifacts/omega/agent_handoff.log.md` with merge timestamp
     + next-up lane.
- **Build target**: post-merge umbrella gate rc=0.

## Coordination mechanics

### Live state file

Single source of truth for who-is-doing-what is
`artifacts/omega/agent_handoff.log.md`. Each lane owner appends:

```text
LANE-N CLAIM 2026-MM-DDTHH:MM:SSZ <agent-id> <files...>
LANE-N RELEASE 2026-MM-DDTHH:MM:SSZ <agent-id> commit=<sha>|abort=<reason>
```

### Branch hygiene rule

Per `feedback_workspace_branch_flips.md`: after every `git checkout` or
`git switch`, run `git branch --show-current` and visually verify the
expected branch before any edit. The session-header banner is **not**
authoritative; `git status -sb` is.

### Dirty-WT rule

Per the 202-line WIP-leak incident: NEVER `git add -A` on a shared
worktree. Always stage by explicit path. Lane workers operate in their
own worktree (`/workspace/sounio-lane-N-...`), which makes accidental
cross-lane staging impossible-by-construction.

### Serialized surfaces

These files require a CLAIM in `artifacts/omega/agent_handoff.log.md`
before any edit, regardless of lane assignment:

- `bin/souc-linux-x86_64` (and `.sha256`, `.sig`)
- `self-hosted/compiler/lean_single.sio`
- `self-hosted/compiler/module_frontend.sio`
- `self-hosted/compiler/native_compile_driver.sio`
- `.claude/settings.json`, `.claude/settings.local.json`
- CI workflow files under `.github/workflows/`
- `self-hosted/check/check.sio` (windowed per
  `.claude/check_sio_integration_window.v1.json`)

### Blocker reporting

Any failure that prevents lane progress is a Blocker per
`.claude/PARALLEL_BLOCKER_CONTRACT.md`. Use the full record shape
(Blocker-ID, Severity, Class, Owner, Lane, Worktree, Branch, ...).

### Merge order when N>1 lanes are PR-ready

Strict priority order:

1. Lane 1 (golden-recapture) — gate-regression, B1
2. Lane 4 (nv2-hardening) — only if it touches `bin/souc`; otherwise
   parallel-mergeable with Lanes 2/3/5
3. Lane 5 (phase-5-recognizer)
4. Lane 2 (dissertation-evidence)
5. Lane 3 (paper-168)

Lanes 2/3/5 are file-disjoint and can land in any order (Lane 6 picks
PR creation timestamp as tiebreaker). Lane 4 must serialize against
Lane 1 on `bin/souc`.

### Per-lane initialization checklist

Each lane owner runs this once on session start:

```bash
# 1. Verify lane assignment
LANE=N  # set to your lane number
WORKTREE=/workspace/sounio-lane-${LANE}-<slug>
git worktree add "$WORKTREE" "coord/lane-${LANE}-<slug>" 2>/dev/null \
  || git worktree add -B "coord/lane-${LANE}-<slug>" "$WORKTREE" origin/main
cd "$WORKTREE"

# 2. Confirm branch identity (branch-flip discipline)
test "$(git branch --show-current)" = "coord/lane-${LANE}-<slug>" \
  || { echo "BRANCH FLIP DETECTED — abort"; exit 1; }

# 3. CLAIM in artifacts/omega/agent_handoff.log.md
# (manually append a lock entry before editing)

# 4. Run baseline build target before any edit
bash scripts/ci/<lane-build-target>.sh
# If RED before edits: classify and report as gate-regression Blocker.
```

### Per-lane PR template

```markdown
[lane-N] <change summary>

## What
<1-3 lines>

## Build target green
- Command: <exact bash command>
- Result: rc=0, <expected output line>

## File-set scope (declare disjoint)
- <file 1>
- <file 2>
...

## Serialized-surface claims (if any)
- bin/souc-linux-x86_64 — coordinated with Lane 1, token released at <commit>

## Blocker-IDs closed by this PR
- BLK-YYYYMMDD-...
```

## Activation Decisions

1. **Lane 1 (golden-recapture)** — owned by Claude #1 and completed in
   PR #95.
2. **Lane 4 branch strategy** — use `coord/lane-4-nv2-hardening` as the
   coordination branch; any older `codex/native-v2-*` branches remain
   historical evidence unless explicitly revived.
3. **Garden lane** — `garden/above-stars` remains out of scope for this
   6-lane overlay.
4. **Cursor lane** — `origin/cursor/quaternionic-ssm-88c0` remains out of
   scope for this 6-lane overlay.
