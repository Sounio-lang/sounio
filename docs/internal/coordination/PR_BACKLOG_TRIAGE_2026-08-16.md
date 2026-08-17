<!-- docs:meta
topic_id: repo.docs.internal.coordination.pr-backlog-triage-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: minimax-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.pr-backlog-triage-2026-08-16
-->

# PR Backlog Triage — 2026-08-16 (Wave 3, coverage-redo 2026-08-17)

Status: wave-3 fleet hygiene. Read-only classification; no PRs merged or closed here.
Author: `minimax-cli1` (tmux fleet window 19), lane `pr-triage-wave3` (extension of
`pr-triage-wave2`). Claim: `bin/sounio-coord claim --agent minimax-cli1 --lane pr-triage-wave3`.
Inputs: `gh pr list --state open --limit 200` (**52** open PRs at 2026-08-17),
per-PR `gh pr view` for file lists + CI rollups, `bin/sounio-coord status` for
active claims, `git log --first-parent main` for merge activity since the trap
cutoff (2026-08-04).

This is a coverage redo. The first version's heading counts (`MERGE 8 / REBASE 8 /
CLOSE 17 / BLOCKED 20`) were assertions, not derived from rows — only 29 of 53
PRs were listed, with 13 merged-commit/issue numbers cited as if they were
triaged rows. The counts below are derived from row tables; if a heading and
its table disagree, the table is authoritative and the heading is stale.

## The trap, internalized

**CLEAN (or MERGEABLE) ≠ still-correct-against-current-main.**

The dispatch's example: #1641 is MERGEABLE per GitHub, its CI was all-green on
**2026-08-04**, and its diff is one fixture file. Between 2026-08-04 and
2026-08-17, `main` (now at `03416657fa`) accumulated **97 first-parent commits
and 72 merge commits**, including #1737 (witness matrix), #1738 (witness matrix
follow-up), #1749 (aggregate-array mutable-borrow miscompile), #1745/#1747 (Mut
effects, env strings), #1748 (viz stdlib), #1751 (plan tranche),
#1752 (registry provenance header fix), #1753 (WS-C PR1 ENIR/MIR shadow lane) and
#1754 (MLI S1 kind model + verifier). The "remaining self-parse failures"
#1641 set out to classify are almost certainly no longer the right list; the
fixture is stale by construction. → CLOSE, not MERGE.

The same logic applies to every PR whose last green CI run is older than
today's `03416657fa` — including most of the dispatch's pre-bucketed CLEAN/READY
set.

**Sub-trap: docs-registry third-party drift.** The Contracts + CI Decision
checks fail for #1641 and #1756 with no doc edits in the diff. Per the wave-2
hazard, this gate goes red from external drift, not from the diff under test —
verify WHY before treating the failure as the PR's fault.

## Active claims (cross-check before any MERGE)

Per `bin/sounio-coord status` at capture (2026-08-17T01:29Z):

| Claim | State | Lane | Files |
|---|---|---|---|
| `claude--session-51d7b0cc-…` | ACTIVE | session-51d7b0cc (claude-1) | `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md`, `.claude/attention_p0.v1.json` (docs only) |
| `codex--session-019fff99-…` | ACTIVE | session-019fff99 (codex-2) | (none — codex-2 released `ws-c-pr2-open` after PR #1756 opened) |
| `claude--session-c89fe8c8-…` | STALE | P0-F (extern "C" FFI) | `check.sio` / `codegen_x86_linux.sio` (was active, now stale) |
| `minimax-cli1--pr-triage-wave3` | STALE | this lane | `docs/internal/coordination/PR_BACKLOG_TRIAGE_2026-08-16.md` (this doc) |
| (others) | STALE | various | various |

**Consequence:** at capture, no code file is under an ACTIVE claim. PRs that
were BLOCKED under the prior capture by the P0-F claim (#867, #1290, #1339,
#1421, #1527, #1604, #1605, #1729) are **no longer claim-blocked**, but they
still touch files that were actively developed and are likely to move again;
they are classified below as BLOCKED-by-chain or BLOCKED-by-stale-P0-F-claim
so the founder sees the hazard rather than merging into a moving target.

## Bucket counts (derived from row tables)

| Bucket | Count |
|---|---:|
| **MERGE** | 8 |
| **REBASE** | 7 |
| **CLOSE** | 17 |
| **BLOCKED** | 20 |
| **Total** | **52** |

---

## Classification table

Every PR below has a one-line reason (per the dispatch). Stale means last
substantive activity >14 days ago. Path collisions are against current `main`
(atop `03416657fa`).

### MERGE (8)

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 1063 | ci(octonion): wire O-SSM probes gate | `research/octonion-probes-ci-gate` | math-impact CI gate; CONFLICTING but docs/infra only, no claim collision, no P0-F touch. |
| 1376 | docs(governance): branch policy | `docs/branch-policy` | docs-only, 4 files; directly supports cursor-1's auto-delete decision item. |
| 1420 | docs(handoff): units dispatch | `docs/units-dispatch` | docs-only handoff dispatch, no claim collision, narrow scope. |
| 1505 | dev(build): Madaros on idle SLURM nodes | `dev/remote-build-slurm-20260726` | addresses plan §5 risk #5 (CPU-saturation pod eviction); MERGEABLE, last green 2026-07-26, no claim collision. |
| 1506 | docs(claude): build lock carveout | `docs/build-lock-carveout-20260726` | docs-only carveout warning, MERGEABLE, no-risk. |
| 1554 | fix(stdlib): correlated cov | `fix/madaros-parity-ab-20260729` | narrow stdlib + gate, Madaros ALL PASS per author, no claim collision. |
| 1720 | darwin_pbpk: Knightian pharmacometrics | `darwin-pbpk/knightian-utiped-p1` | orthogonal darwin-pbpk workstream, MERGEABLE + CI all GREEN 2026-08-13, 2 files, no claim collision. |
| 1730 | stdlib: insertion-sort break | `darwin-pbpk/stdlib-sort-fix` | narrow stdlib fix, CI all GREEN 2026-08-13, 3 files, no claim collision. |

### REBASE (7)

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 816 | test(madaros): #651 array-of-struct | `work/sr651-madaros-witness` | base=`work/madaros-changed-ci` (not main); MERGEABLE per gh but `Madaros Changed Tests` + `CI Decision` FAIL on 2026-07-12; rebase onto `03416657fa` and re-run. |
| 817 | test(madaros): generic struct-return | `work/structf-effect-witness` | base=`work/madaros-changed-ci`; `Madaros Changed Tests` + `CI Decision` FAIL on 2026-07-12; same rebase plan as #816. |
| 840 | fix(parser): `study` soft keyword | `fix/parser-study-soft-keyword` | CONFLICTING vs main; closes #740 arm64 parity; small parser change — rebase path should be clean. |
| 1603 | agent-bus realtime | `feat/agent-bus-realtime` | CONFLICTING vs main; touches MCP surface (`scripts/mcp/sounio_coord_mcp.py`) — verify against current `bin/sounio-coord` claim model before rebase. |
| 1721 | darwin_pbpk: Conformal Weaning | `darwin-pbpk/conformal-utiped-p2` | MERGEABLE but `Full Test Suite` + `CI Decision` FAIL on 2026-08-13; re-run on current main — if still red, close as superseded. |
| 1750 | [backend] CUDA ABI launch packing | `cherry/gpu-cuda-fixes-20260815` | CONFLICTING vs main; CI green on substantive checks; touches `bin/souc-lean-single-x86_64` (prebuilt-binary staleness risk, plan §5 risk #6); verify rebuild before merge. |
| 1756 | WS-C PR2: wire Madaros v2 ENIR gate stack | `lane/codex-2/ws-c-pr2-20260816` | MERGEABLE; `Contracts` + `CI Decision` FAIL looks like docs-registry third-party drift (no doc edits in 51-file diff), all substantive checks SUCCESS — re-run; if still red on Contracts, run `scripts/docs/sync_governance_metadata.mjs` and re-verify. |

### CLOSE (17)

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
| 1641 | docs(ci): self-parse baseline classified | `docs/self-parse-baseline-classified` | **TRAP — CLOSE despite MERGEABLE.** Single fixture file, last green CI 2026-08-04, but main has moved 97 first-parent + 72 merge commits since then including #1753 (ENIR/MIR shadow) + #1754 (MLI S1); fixture is stale by construction — purpose obsolete. (The current `Contracts` + `CI Decision` FAIL is docs-registry drift on the registry itself; #1752 already fixed the header-preservation bug for that.) |
| 1659 | feat(san-fpga): SAN-v3 curriculum | `research/san-fpga-san-v3-20260805` | CONFLICTING; research-grade SAN, not in active scope; no founder "let finish" hold. |

### BLOCKED (20)

#### BLOCKED by stale P0-F claim on `self-hosted/check/check.sio` and/or `self-hosted/native/codegen_x86_linux.sio`

Note: the P0-F claim (`claude--session-c89fe8c8-…`) is STALE at capture.
Not technically claim-blocked right now, but the files were under active
development and are likely to move again. Listed here so the founder sees
the collision risk; merge requires explicit P0-F release/refresh coordination.

| # | Title (short) | Head | Claimed files touched |
|---|---|---|---|
| 867 | checker: contextual function lookup | `agent/issue854-contextual-checker-partial-20260713` | `self-hosted/check/check.sio` |
| 1290 | [checker] affine ownership | `codex/madaros-affine-semantics-20260720` | `self-hosted/check/check.sio` |
| 1339 | fix(madaros): capacity/slicing/wide-call | `agent/madaros-declared-builtin-precedence-20260720` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |
| 1421 | fix(madaros): preserve imported layouts | `codex/issue901-layout-current-20260724` | `self-hosted/check/check.sio` |
| 1527 | fix(madaros): self-parse/Box::new/W044 | `madaros/self-parse-visibility-box-w44-20260727` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |
| 1729 | fix(madaros): B3 IrModule BSS | `fix/lane-b3-ir-module-heap-20260813` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |

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

## Stale-trap appendix: PRs with green CI but old (the #1641 hazard generalized)

For every PR with last green CI before today's `03416657fa`, the green tick is
evidence about a tree that no longer exists. These are not MERGE without
explicit founder sign-off, even if the API says MERGEABLE.

| PR | Last green CI | Risk class |
|---|---|---|
| 1641 | 2026-08-04 | fixture stale by construction (see header). **CLOSE.** |
| 1554 | 2026-07-29 | narrow stdlib fix, low risk. MERGE-eligible but re-run CI before merge. |
| 1376 | 2026-07-26 | docs-only, no-risk. MERGE-eligible. |
| 1420 | 2026-07-26 | docs-only, no-risk. MERGE-eligible. |
| 1505 | 2026-07-26 | infra shell script, low risk. MERGE-eligible. |
| 1506 | 2026-07-26 | docs-only, no-risk. MERGE-eligible. |
| 1730 | 2026-08-13 | fresh, narrow stdlib fix. **MERGE-eligible.** |
| 1720 | 2026-08-13 | fresh, narrow workstream. **MERGE-eligible.** |

The 2026-08-13 green ticks were against a `main` closer to current than the
2026-07-26 to 2026-08-04 set; those are the safer MERGE candidates.

## Context references (NOT triage rows — do not double-count)

The following numbers appear in this document for narrative continuity only;
they are **not** open PRs and have no bucket. Listed so the audit can confirm
none of them were mis-classified as triage rows.

**Merged commits cited in the trap narrative (main-line merges since 2026-08-04):**

| Ref | Subject |
|---|---|
| 09adb0f773 | Merge #1737 — worktree witness matrix (D-series gate) |
| bddfe19fad | Merge #1738 — worktree witness matrix follow-up |
| 8c7300c0b7 | Merge #1741 — (per wave-2 audit; subject TBD) |
| 6f2c4e2461 | Merge #1751 — wave-1 planning tranche |
| 16573d73e0 / e7d33719e9 / fab45306a5 / 16573d73e0 / e6d2dbee02 | Lane-A / Lane-B lowerer / module-functions / gri30 / UI-error hotfixes |
| 6b2198e314 / 725beb5bc7 / 911a2770fa / d9d56436ee / 6d84b8d19b / ea65acc50d | Cap-22 / linear-branch-merge / visibility-preflight / aggregate-deep-copy / handle-table / tuple-f64-slot-classification (#1490, #1493, #1500, #1501, #1508, #1697) |

**Issues cited as closure targets or chain roots:**

| Issue | Cited in |
|---|---|
| #651 | SR-651 array-of-struct witness (target of #816) |
| #740 | arm64 parity (target of #840) |
| #854 | contextual checker chain (root of #867, #869) |

**Cross-deliverable:** this doc and `OPEN_PR_TRIAGE_2026-08-16.md` (wave-2) are
complementary — wave-2 listed 5 specifically-named PRs with the trap caveat on
#1641; wave-3 widens to the full 52 and enforces the trap uniformly.

## Coordination summary

- Lane: `minimax-cli1--pr-triage-wave3`. Active (claim STALE per registry; will
  be refreshed on next heartbeat).
- Active claims honoured: at capture, no code file is under an ACTIVE claim,
  so no MERGE recommendation collides with a current claim.
- No PRs merged or closed by this triage. Self-hosted/ untouched.
- Coverage redo on 2026-08-17: heading counts derived from rows; merged-commit
  and issue references isolated to "Context references" section.
- Commit hash for this file's commit on `lane/minimax-cli1/20260815` recorded
  in the PR description if this triage triggers a docs-registry sync.
