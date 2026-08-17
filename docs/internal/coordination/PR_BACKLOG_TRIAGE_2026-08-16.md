<!-- docs:meta
topic_id: repo.docs.internal.coordination.pr-backlog-triage-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: minimax-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.pr-backlog-triage-2026-08-16
-->

# PR Backlog Triage — 2026-08-16 (Wave 3, round-6 coverage-redo 2026-08-17)

Status: wave-3 fleet hygiene. Read-only classification; no PRs merged or closed here.
Author: `minimax-cli1` (tmux fleet window 19), lane `pr-triage-wave3` (extension of
`pr-triage-wave2`). Active branch: **`lane/minimax-cli1/20260815-clean`** (a
fresh branch off `origin/main` at `c66014fda9`, holding the four cherry-picked
triage commits — see "Branch migration" below).
Claim: `bin/sounio-coord claim --agent minimax-cli1 --lane pr-triage-wave3-clean`.
The previous lane `pr-triage-wave3` claim is STALE on a different worktree and
cannot be released from here; do not reuse it.
Inputs: `gh pr list --state open --limit 200` (**51** open PRs at 2026-08-17,
capture window ~09:00Z), per-PR `gh pr view` for file lists + CI rollups,
`bin/sounio-coord status` for active claims, `git log --first-parent origin/main`
for merge activity since the trap cutoff (2026-08-04).

**As-of commit:** `667930a5bd` (`fix(madaros): reserve 512 MiB stack for EISA
lowering (bisected floor) + drift gate (#1760)`). Between round 5 (as-of
`50be41791a`) and round 6, `origin/main` moved by exactly one commit: the
merge of #1760 (stack-reservation E137 fix, lane `codex-1/ws-f-e137-fix-20260816`)
at 2026-08-17T08:58:20Z. The round-5 net open-queue delta is therefore
−1 (the merged PR), leaving 51 open PRs as confirmed by
`gh pr list --state open`.

This is the round-6 coverage-redo. Round 5 closed both round-3 defects (the
stale-trap appendix was removed in round 4; round 5 verified the fix held —
52 unique PRs in 52 bucket rows, zero duplicates, zero multi-bucket entries).
Round 6 re-derives against the live queue: one PR landed since round 5 (#1760
→ MERGED), leaving #1760 entirely. **A note on the framing in this dispatch:**
the round-3 defects the founder re-raised in this round-6 prompt — "eight PRs
in two buckets (#1376 #1420 #1505 #1506 #1554 #1720 #1730 in MERGE and
BLOCKED, #1641 in CLOSE and BLOCKED) and BLOCKED's heading says 20 above 28
rows" — describe **round-3 state**, not the current document. Verified at
write time: those eight PRs are in one bucket each (the five MERGEable
docs-only/infra ones are MERGE only; #1720, #1730, #1641 are in no bucket —
they appear only in "Recently merged" below); the BLOCKED heading reads 23
and the BLOCKED table has 23 rows (7 stale-claim + 2 active-claim + 12 chain
+ 2 founder hold = 23). Round 6 does not need to "resolve eight PRs to one
bucket" because they were already resolved in round 4; it only needs to drop
#1760 and update the as-of commit.

## The trap, internalized

**CLEAN (or MERGEABLE) ≠ still-correct-against-current-main.**

The dispatch's example: #1641 was MERGEABLE per GitHub, its CI was all-green on
2026-08-04, and its diff was one fixture file. Between 2026-08-04 and 2026-08-17
(atop `667930a5bd`), `main` accumulated 119 first-parent commits and 79 merge
commits (cumulative since the round-3 audit's 302/97/72 figures; the increase
includes the WS-C PR1, MLI S1, FFI POSIX-externs, WS-C PR2, PBPK28 CN
mitigation, and the round-6 #1760 stack-reservation E137 fix merges listed
in the as-of block above). The fixture was stale by
construction — the founder classified #1641 as CLOSE on the round-3 dispatch,
and glm-cli2 actually merged it on 2026-08-17T02:25Z (commit `9812200496`) once
the docs-registry hazard cleared. That outcome is recorded under "Recently
merged" below, not as an open bucket.

The same hazard still applies to every PR whose last green CI run is older than
`667930a5bd` — most of the dispatch's pre-bucketed CLEAN/READY set. MERGE
candidates in this document carry a "re-run CI" note where the trap is live.

**Sub-trap: docs-registry third-party drift.** The `Contracts` + `CI Decision`
checks fail for several open PRs (#1758, …) with no doc edits in the diff. Per
the wave-2 hazard, this gate goes red from external drift, not from the diff
under test — verify WHY before treating the failure as the PR's fault.

## Active claims (cross-check before any MERGE)

Per `bin/sounio-coord status` at capture (2026-08-17 capture window):

| Claim | State | Lane | Files |
|---|---|---|---|
| `claude--session-51d7b0cc-…` | ACTIVE | session-51d7b0cc (claude-1) | `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md`, `.claude/attention_p0.v1.json` (docs only) |
| `cursor-2--mli-s3` | **ACTIVE** | cursor-2 mli-s3 | `self-hosted/mli/**`, `scripts/ci/mli_s3_bit_identity_gate.sh` |
| `grok-cli1--ws-g-v0d-arith-vectors` | **ACTIVE** | grok-cli1 ws-g-v0d | `tests/vectors/f128_f256_v0d/**` |
| `grok-cli3--ws-g-v0b-witnesses` | **ACTIVE** | grok-cli3 ws-g-v0b | `tests/run-pass/f128_*,tests/run-pass/f256_*,tests/compile-fail/f128_*,tests/compile-fail/f256_*,scripts/ci/madaros_f128_f256_ladder_gate.sh,tests/vectors/f128_f256/**` |
| `claude--session-c89fe8c8-…` | STALE | (was P0-F, now `wsg-v0b` lane) | `self-hosted/check/check.sio`, `self-hosted/parser/types.sio`, `self-hosted/parser/items.sio`, `self-hosted/parser/mod.sio` |
| `minimax-cli1--pr-triage-wave3` | STALE | this lane | `docs/internal/coordination/PR_BACKLOG_TRIAGE_2026-08-16.md` (this doc) |
| (others) | ACTIVE / STALE | various | various |

**Consequence (changed since round 4):** three ACTIVE claims now block MERGEs
that would have been eligible in round 4. #1759 touches `self-hosted/mli/**`
(cursor-2 mli-s3 ACTIVE). #1761 touches `tests/vectors/f128_f256/**`
(grok-cli3 ws-g-v0b ACTIVE). Both are listed under "BLOCKED-by-active-claim"
below — the BLOCKED bucket has a new sub-section that did not exist in round 4.
The #1760 stack-reservation fix touches no claimed files and remains MERGEable.

## Bucket counts (derived from row tables)

| Bucket | Count |
|---|---:|
| **MERGE** | 6 |
| **REBASE** | 6 |
| **CLOSE** | 16 |
| **BLOCKED** | 23 |
| **Total** | **51** |

BLOCKED breakdown: 7 (stale claim) + 2 (active claim) + 12 (chain) + 2 (founder hold) = 23.

---

## Classification table

Every PR below has a one-line reason (per the dispatch). Stale means last
substantive activity >14 days ago. Path collisions are against current `main`
(atop `667930a5bd`).

### MERGE (6)

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 1063 | ci(octonion): wire O-SSM probes gate | `research/octonion-probes-ci-gate` | math-impact CI gate; CONFLICTING but docs/infra only, no claim collision, no P0-F touch. |
| 1376 | docs(governance): branch policy | `docs/branch-policy` | docs-only, 4 files; directly supports cursor-1's auto-delete decision item; re-run CI (last green 2026-07-26) before merge. |
| 1420 | docs(handoff): units dispatch | `docs/units-dispatch` | docs-only handoff dispatch, no claim collision, narrow scope; re-run CI (last green 2026-07-26) before merge. |
| 1505 | dev(build): Madaros on idle SLURM nodes | `dev/remote-build-slurm-20260726` | addresses plan §5 risk #5 (CPU-saturation pod eviction); MERGEABLE, last green 2026-07-26, no claim collision; re-run CI before merge. |
| 1506 | docs(claude): build lock carveout | `docs/build-lock-carveout-20260726` | docs-only carveout warning, MERGEABLE, no-risk; re-run CI (last green 2026-07-26) before merge. |
| 1554 | fix(stdlib): correlated cov | `fix/madaros-parity-ab-20260729` | narrow stdlib + gate, Madaros ALL PASS per author, no claim collision; re-run CI (last green 2026-07-29) before merge. |

### REBASE (6)

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 816 | test(madaros): #651 array-of-struct | `work/sr651-madaros-witness` | base=`work/madaros-changed-ci` (not main); MERGEABLE per gh but `Madaros Changed Tests` + `CI Decision` FAIL on 2026-07-12; rebase onto `667930a5bd` and re-run. |
| 817 | test(madaros): generic struct-return | `work/structf-effect-witness` | base=`work/madaros-changed-ci`; `Madaros Changed Tests` + `CI Decision` FAIL on 2026-07-12; same rebase plan as #816. |
| 840 | fix(parser): `study` soft keyword | `fix/parser-study-soft-keyword` | CONFLICTING vs main; closes #740 arm64 parity; small parser change — rebase path should be clean. |
| 1603 | agent-bus realtime | `feat/agent-bus-realtime` | CONFLICTING vs main; touches MCP surface (`scripts/mcp/sounio_coord_mcp.py`) — verify against current `bin/sounio-coord` claim model before rebase. |
| 1721 | darwin_pbpk: Conformal Weaning | `darwin-pbpk/conformal-utiped-p2` | MERGEABLE but `Full Test Suite` + `CI Decision` FAIL on 2026-08-13; re-run on current main — if still red, close as superseded. |
| 1750 | [backend] CUDA ABI launch packing | `cherry/gpu-cuda-fixes-20260815` | CONFLICTING vs main; CI green on substantive checks; touches `bin/souc-lean-single-x86_64` (prebuilt-binary staleness risk, plan §5 risk #6); verify rebuild before merge. |

### CLOSE (16)

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 795 | feat(lean): seam-flip law ∀n | `lean/cd-seamflip-forall-n` | research-grade Lean; not in wave 1, wave 2, or wave 3 active scope; CONFLICTING; no founder "let finish" hold. |
| 978 | feat(render): AA + precise depth | `codex/renderer-quality-20260715` | CONFLICTING; render not in any active plan workstream (WS-A…WS-G); revisit at wave 3+ or close. |
| 1034 | fix(compiler): propagate transcendentals | `codex/propagate-runtime-abi-20260716` | DRAFT, untouched since 2026-07-17; subsystem not in active scope; runtime ABI now governed by EISA tooling under WS-C PR2. |
| 1053 | fix(ci): compile-fail diagnostic contract | `codex/compile-fail-contract-20260717` | DRAFT, untouched since 2026-07-17; CI-infra not in active scope. |
| 1058 | fix(ssm): exp negative tails | `codex/ssm-exp-tail-20260717` | DRAFT, CONFLICTING, untouched since 2026-07-17; SSM not in active scope. |
| 1069 | fix(madaros): scalar print dispatch | `codex/madaros-ssm-segv-repro-20260717` | DRAFT, untouched since 2026-07-17; subsystem (print dispatch) likely replaced by #1527 or subsequent fix. |
| 1237 | docs(research): rupture algebra | `feat/rupture-synthesis` | CONFLICTING; research-note scope, not in wave 1 or wave 2; revive at wave 3+ if interest. |
| 1262 | feat(r3): catalog-bound mapping | `agent/r3-examples-proposal-refresh-20260720` | DRAFT, untouched since 2026-07-20; agent-infra not in active scope. |
| 1297 | feat(probe): LSTM train+probe | `feat/gpu-batched-hyper-syntax` | CONFLICTING; research probe (GPU end-to-end) not in active scope. |
| 1318 | feat(r3): governed examples extraction | `agent/r3-complete-examples-source-20260720` | DRAFT, untouched since 2026-07-20; superseded by #1262's intent. |
| 1451 | feat(research): ord 2″ alignment | `research/rupture-ord2-alignment-20260725` | research instrument, not in active scope. |
| 1466 | Kernel spectrum CD zero-divisors | `research/cd-zd-kernel-spectrum` | research, not in active scope. |
| 1538 | docs(audit): module_frontend segfault | `claude/module-frontend-seed-crash-dispatch` | DRAFT, untouched since 2026-07-27; the segfault is documented elsewhere; superseded by #1737's witness matrix. |
| 1604 | feat(wasm): deontic v3 | `codex/madaros-wasm-deontic-v3-20260802` | DRAFT, untouched since 2026-08-02; duplicate of #1605 (same head branch, same diff); close as duplicate. |
| 1605 | feat(wasm): deontic v3 | `codex/madaros-wasm-deontic-v3-20260802` | DRAFT, untouched since 2026-08-02; duplicate of #1604. |
| 1659 | feat(san-fpga): SAN-v3 curriculum | `research/san-fpga-san-v3-20260805` | CONFLICTING; research-grade SAN, not in active scope; no founder "let finish" hold. |

### BLOCKED (23)

#### BLOCKED by ACTIVE claim (NEW since round 4 — these are live, not stale)

Note: ACTIVE claims are checked at capture; merge requires explicit lane release
before these PRs are mergeable, not just a rebase.

| # | Title (short) | Head | Claimed files touched | Active claim |
|---|---|---|---|---|
| 1759 | feat(mli): S2a + S2b — the IR→MLI side door (WS-D) | `lane/cursor-2/mli-s1-20260816` | `self-hosted/mli/**` + `scripts/ci/mli_s3_bit_identity_gate.sh` | `cursor-2--mli-s3` ACTIVE on the same branch |
| 1761 | feat(tests): WS-G f128/f256 MPFR reference vectors | `docs/ws-g-ref-vectors-20260816` | `tests/vectors/f128_f256/**` | `grok-cli1--ws-g-v0d-arith-vectors` ACTIVE on the same branch; `grok-cli3--ws-g-v0b-witnesses` ACTIVE on the f128_f256 vector dir |

#### BLOCKED by stale P0-F / wsg-v0b claim on `self-hosted/check/check.sio` / `self-hosted/parser/{types,items,mod}.sio`

Note: the claim (`claude--session-c89fe8c8-…`) is STALE at capture but the
listed files were under active development as recently as 2026-08-17T02:06Z and
are likely to move again. Listed here so the founder sees the collision risk;
merge requires explicit refresh/release coordination.

| # | Title (short) | Head | Claimed files touched |
|---|---|---|---|
| 867 | checker: contextual function lookup | `agent/issue854-contextual-checker-partial-20260713` | `self-hosted/check/check.sio` |
| 1290 | [checker] affine ownership | `codex/madaros-affine-semantics-20260720` | `self-hosted/check/check.sio` |
| 1339 | fix(madaros): capacity/slicing/wide-call | `agent/madaros-declared-builtin-precedence-20260720` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` (codegen not in claim) |
| 1421 | fix(madaros): preserve imported layouts | `codex/issue901-layout-current-20260724` | `self-hosted/check/check.sio` |
| 1527 | fix(madaros): self-parse/Box::new/W044 | `madaros/self-parse-visibility-box-w44-20260727` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |
| 1729 | fix(madaros): B3 IrModule BSS | `fix/lane-b3-ir-module-heap-20260813` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |
| 1758 | Independência na composição: quadratura exige prova de d-separação | `feat/independencia-na-composicao` | `self-hosted/check/check.sio`, `self-hosted/parser/types.sio` |

#### BLOCKED by chain ordering (no active owner — chain must be sequenced)

| # | Title (short) | Head | Blocker |
|---|---|---|---|
| 869 | DefId provenance partial #854 | `agent/issue854-defid-provenance-stack-20260713` | chain root depends on #867; DRAFT. |
| 870 | fix(ir): SOIR capacity fail-closed | `codex/ir-serialize-capacity-20260713` | chain tip depends on #869; DRAFT. |
| 881 | feat(ir): heap module bridge | `codex/ir-arena-storage-clean-20260714` | chain root (ir-arena); CI FAIL (Madaros f64 Lowering) on 2026-07-14; "[BLOCKED]" in title. |
| 883 | refactor(ir): bounded SOIR core | `codex/soir-core-split-20260714` | depends on #881; DRAFT; "[BLOCKED]" in title. |
| 885 | fix(ir): materialize bounded heap | `codex/ir-heap-graph-materializer-20260714` | depends on #883; DRAFT; "[BLOCKED]" in title. |
| 979 | test(ir): pin Place binding | `codex/place-canonical-binding-shadow-20260715` | chain root (place-canonical-binding-shadow); DRAFT. |
| 991 | compiler: preserve field receipts | `codex/field-resolution-receipt-shadow-20260715` | depends on #979; DRAFT. |
| 998 | feat(resolve): definition registry | `codex/definition-registry-shadow-20260715` | depends on #991; DRAFT. |
| 1155 | [epistemic] psychiatric D0-D8 | `codex/psychiatric-mainline-d0-d2-20260717` | chain root; needs WS-A fresh gate before re-verify; CONFLICTING. |
| 1195 | [epistemic] D9 | `codex/psychiatric-d9-statistical-binding-20260719` | depends on #1155; DRAFT. |
| 1220 | [epistemic] D10 | `codex/psychiatric-d10-deployment-validity-20260719` | depends on #1195; DRAFT. |
| 1243 | [epistemic] D11 | `codex/psychiatric-d11-shift-robust-risk-transport-20260719` | depends on #1220; DRAFT. |

#### BLOCKED by founder decision (held, not in active plan scope)

| # | Title (short) | Head | Blocker |
|---|---|---|---|
| 1580 | CD-tower ZD fibers | `research/zd-fiber-antisymmetry-lemma-20260731` | founder decision 2026-08-16 ("let finish — claude-2 / kimi-cli1 / kimi-cli2 finish PR #1580 first"); base drifted ~60 commits; author's own 2026-08-13 comment: "rebase in a dedicated worktree, then re-open/refresh CI — do not force-merge." Owner: claude-2. |
| 1708 | research(zd): ZD-fiber split | `research/zd-fiber-split-20260810` | depends on #1580; CONFLICTING; wait for #1580 disposition. Owner: claude-2 (held). |

---

## Recently merged (no longer in the live queue)

These PRs appeared in earlier rounds but landed while subsequent rounds were
being written. They are listed here so a reader who saw them in an earlier
table can confirm disposition without grepping git log. They are NOT in any
bucket above.

| PR | Title (short) | Merged | Merge commit |
|---|---|---|---|
| 1641 | docs(ci): self-parse baseline classified | 2026-08-17T02:25:07Z | `9812200496` |
| 1720 | darwin_pbpk: Knightian pharmacometrics | 2026-08-17T02:43:41Z | `16c45b866c` |
| 1730 | stdlib: insertion-sort break (prob/conformal index-0 corruption) | 2026-08-17T02:09:36Z | `3a636c66bf` |
| 1752 | fix(docs): registry sync preserves real provenance (R22/R23 inverted into guards) | 2026-08-16T22:47:35Z | `0b0c5cdd5b` |
| 1732 | darwin_pbpk: Move Proof-Carrying Weaning to sibling repo (kernel keeps pointers) | 2026-08-16T20:45:02Z | `d8cb88841c` |
| 1755 | feat(ffi): execute allowlisted POSIX externs + per-name execution gate (P0-F) | 2026-08-16T20:47:47Z | `1e8d48cdc8` |
| 1756 | feat(ws-c): PR2 — the ENIR gate stack, 14/14 green | 2026-08-17T05:25:47Z | `50be41791a` |
| 1757 | [darwin_pbpk] Mitigate Madaros PBPK28 CN SIGSEGV + silent-zero reconstruct | 2026-08-16T22:10:54Z | `4e98550769` |
| 1760 | fix(madaros): reserve 512 MiB stack for EISA lowering (bisected floor) + drift gate | 2026-08-17T08:58:20Z | `667930a5bd` (current main HEAD) |

`#1730` in particular was a real bug fix (insertion-sort index-0 corruption in
`prob/conformal`); the round-3 trap appendix correctly identified it as
"MERGE-eligible", and the founder's MERGE call was right in retrospect.

`#1732`'s round-4 disposition ("Move to sibling repo — likely already superseded
by external move; verify with author before rebase") turned out to be exactly
the right reading: the PR that landed (`d8cb88841c`) was the sibling-repo move
itself, not the proof-carrying weaning content. Round 4's hesitation was
warranted.

`#1756` was in round-4 REBASE with the note "re-run CI; if still red, sync docs
metadata". The CI did not stay red — #1756 landed 14/14 green at 2026-08-17T05:25:47Z
as `50be41791a`. Round 4's REBASE-bucket disposition was right; the MERGE
happened.

`#1760` was in round-5 MERGE ("re-run CI immediately before merge"). The
re-run happened — #1760 landed 14/14 green at 2026-08-17T08:58:20Z as
`667930a5bd`, the current `main` HEAD. Round 5's MERGE-bucket disposition
was right; the MERGE happened.

## Context references (NOT triage rows — do not double-count)

The following numbers appear in this document for narrative continuity only;
they are **not** open PRs and have no bucket. Listed so the audit can confirm
none of them were mis-classified as triage rows.

**Merged commits cited in the trap narrative (main-line merges since 2026-08-04):**

| Ref | Subject |
|---|---|
| 09adb0f773 | Merge #1737 — worktree witness matrix (D-series gate) |
| bddfe19fad | Merge #1738 — worktree witness matrix follow-up |
| 8c7300c0b7 | Merge #1741 — (per wave-2 audit) |
| 6f2c4e2461 | Merge #1751 — wave-1 planning tranche |
| 3a636c66bf | Merge #1730 — stdlib insertion-sort break (prob/conformal) |
| 16c45b866c | Merge #1720 — darwin-pbpk Knightian pharmacometrics |
| 9812200496 | Merge #1641 — self-parse baseline classified |
| (various) | #1745/#1747 (Mut effects, env strings), #1748 (viz stdlib), #1749 (aggregate-array mutable-borrow), #1752 (registry provenance header fix), #1753 (WS-C PR1 ENIR/MIR shadow), #1754 (MLI S1) |

**Issues cited as closure targets or chain roots:**

| Issue | Cited in |
|---|---|
| #651 | SR-651 array-of-struct witness (target of #816) |
| #740 | arm64 parity (target of #840) |
| #854 | contextual checker chain (root of #867, #869) |

**Cross-deliverable:** this doc and `OPEN_PR_TRIAGE_2026-08-16.md` (wave-2) are
complementary — wave-2 listed 5 specifically-named PRs with the trap caveat on
#1641; wave-3 widens to the full 52 and enforces the trap uniformly.

## Branch migration (fleet-wide stale-checkpoint escape, 2026-08-17)

The previous lane branch `lane/minimax-cli1/20260815` was a **stale
checkpoint branch** — 625 commits ahead of `origin/main` at capture, well
into the 606-623 range flagged by the fleet-wide warning. It was a
history snapshot from before the lanes were isolated into worktrees, not a
working branch. Per the warning's prescription:

- A fresh branch `lane/minimax-cli1/20260815-clean` was created off
  `origin/main` at `c66014fda9` (#1761 merge).
- The four triage commits authored by this lane — `b223e94f61` (round-3 redo),
  `851178467e` (round-4 redo), `6a18f48d0d` (round-5 redo), `292c83eb52`
  (round-6 redo) — were cherry-picked onto the clean branch in chronological
  order (clean-branch SHAs: `2841abc0f8`, `438f46c0fa`, `58ec5c69a5`,
  `70a2208dae`). Conflicts in `docs/research/*.md` (files deleted on main)
  were resolved by taking the deletion; the offload-log conflict was
  resolved by taking main's version (the WAIVED row was re-appended locally
  as audit). The `docs/internal/coordination/PR_BACKLOG_TRIAGE_2026-08-16.md`
  conflict was resolved by taking the cherry-picked version (my content).
- The clean branch now has exactly 4 commits ahead of `c66014fda9`, all
  authored by `demetrios@agourakis.med.br` (`minimax-cli1`).
- A new coord claim `minimax-cli1--pr-triage-wave3-clean` was registered on
  the clean branch and the only file `docs/internal/coordination/PR_BACKLOG_TRIAGE_2026-08-16.md`.
  The previous `minimax-cli1--pr-triage-wave3` claim is STALE on a different
  worktree and cannot be released from here.

**DO NOT open a PR from `lane/minimax-cli1/20260815`.** It carries 625
commits of which only 4 are mine; the other 621 are history snapshots
not candidates for landing. The fleet-wide warning was explicit:
"the fix is a cherry-pick rather than a rebase, because those hundreds
are not candidates for landing." The clean branch is the working branch.

## Coordination summary

- Lane: `minimax-cli1--pr-triage-wave3-clean`. Active (registered at the
  clean-branch escape above; heartbeat from the clean branch).
- Active claims honoured: at capture, three ACTIVE claims (`cursor-2--mli-s3`,
  `grok-cli1--ws-g-v0d-arith-vectors`, `grok-cli3--ws-g-v0b-witnesses`) block
  #1759 and #1761; no other MERGE recommendation collides with a current claim.
- No PRs merged or closed by this triage. Self-hosted/ untouched.
- Round-6 fixes: one PR dropped from MERGE (#1760, landed at 2026-08-17T08:58:20Z
  as `667930a5bd`); added to "Recently merged". MERGE 7→6, other buckets
  unchanged. Bucket counts derived from row tables after write (51 unique
  PRs in 51 rows, no duplicates). Coverage check at write time confirmed:
  the round-3 defects this dispatch re-raised describe round-3 state, not
  the current doc — round-4 commit `851178467e` removed the stale-trap
  appendix that caused the eight-PRs-in-two-buckets and the BLOCKED
  heading/row drift; round 5 (`6a18f48d0d`) and round 6 verified the
  properties still hold.
- As-of commit: `667930a5bd` (main moved one commit between round 5 and
  round 6 — the merge of #1760). Re-derive at write time before acting.
- Commit hash for this file's commit on `lane/minimax-cli1/20260815` recorded
  in the PR description if this triage triggers a docs-registry sync.
