<!-- docs:meta
topic_id: repo.docs.internal.coordination.pr-backlog-triage-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.pr-backlog-triage-2026-08-16
-->

# PR Backlog Triage — 2026-08-16 (Wave 3)

Status: wave-3 fleet hygiene. Read-only classification; no PRs merged or closed here.
Author: `minimax-cli1` (tmux fleet window 19), lane `pr-triage-wave3` (extension of
`pr-triage-wave2`). Claim: `bin/sounio-coord claim --agent minimax-cli1 --lane pr-triage-wave3`.
Inputs: `gh pr list --state open --limit 200` (53 PRs), per-PR `gh pr view` for
file lists + CI rollups + timestamps, `bin/sounio-coord status` for active claims,
`git log` on `main` for merge activity since the trap cutoff (2026-08-04).

## The trap, internalized

**CLEAN (or MERGEABLE) ≠ still-correct-against-current-main.**

The dispatch's example: #1641 is MERGEABLE per GitHub, its CI was all-green on
**2026-08-04**, and its diff is one fixture file (`scripts/ci/fixtures/madaros_self_parse_baseline.txt`).
Between 2026-08-04 and 2026-08-16, `main` accumulated **302 commits**, including
#1643 (decimal float literals), #1640 (extended/chunked scaling), #1737 (enum/f32
lower), #1749 (aggregate-array mutable-borrow), #1745/#1747 (Mut effects, env
strings), #1748 (viz stdlib), #1751 (plan tranche), **#1753 (WS-C PR1, ENIR/MIR
shadow lane — 8759 insertions across `self-hosted/enir/*`)** and **#1754 (MLI
S1, kind model + verifier — 2340 insertions across `self-hosted/mli/*`)**. The
"remaining self-parse failures" #1641 set out to classify are almost certainly
no longer the right list; the fixture is stale by construction. → CLOSE, not MERGE.

The same logic applies to every PR whose last green CI run is older than today's
`453b2e6e2f` — including most of the dispatch's pre-bucketed CLEAN/READY set.

## Active claims (cross-check before any MERGE)

Per `bin/sounio-coord status` at capture:

| Claim | Lane | Files | Owner |
|---|---|---|---|
| `claude--session-c89fe8c8-…` | P0-F (extern "C" FFI) | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` | active, `/workspace/.wt/p0f-v2` |
| `codex-2--ws-c-pr2-execute` | WS-C PR2 (staging) | `tools/eisa/eisa_enir_*`, `scripts/dev/madaros_v2_e*` | active, `/workspace/.wt/codex-2` |
| `claude--session-51d7b0cc-…` | session | `MADAROS_FOCUS_PLAN_2026-08-16.md`, `.claude/attention_p0.v1.json` | active, `/workspace/.wt/claude-1` |

Per the dispatch's rule: **a PR touching a claimed file is not mergeable no
matter how green its CI**. The P0-F claim alone blocks 9 of 53 PRs; see table.

## Bucket counts

| Bucket | Count |
|---|---:|
| **MERGE** | 8 |
| **REBASE** | 8 |
| **CLOSE** | 17 |
| **BLOCKED** | 20 |
| **Total** | **53** |

---

## Classification table

Every PR below has a one-line reason (per the dispatch). Stale means last
substantive activity >14 days ago. Path collisions are against current `main`
(atop `453b2e6e2f`).

### MERGE (8) — rebase onto `453b2e6e2f`, CI re-run green, no claim collision, no stale-trap risk

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 1063 | ci(octonion): wire O-SSM probes gate | `research/octonion-probes-ci-gate` | docs-only CI gate, no claim collision, gates the math-impact surface orthogonal to compiler work. |
| 1376 | docs(governance): branch policy | `docs/branch-policy` | docs-only, 4 files; directly supports cursor-1's auto-delete decision item. |
| 1420 | docs(handoff): units dispatch | `docs/units-dispatch` | docs-only handoff dispatch, no claim collision, narrow scope. |
| 1505 | dev(build): Madaros on idle SLURM nodes | `dev/remote-build-slurm-20260726` | addresses plan §5 risk #5 (CPU-saturation pod eviction), orthogonal infra work, no claim collision. |
| 1506 | docs(claude): build lock carveout | `docs/build-lock-carveout-20260726` | docs-only carveout warning, no-risk. |
| 1554 | fix(stdlib): correlated cov | `fix/madaros-parity-ab-20260729` | narrow stdlib + gate, Madaros ALL PASS per author, no claim collision. |
| 1720 | darwin_pbpk: Knightian pharmacometrics | `darwin-pbpk/knightian-utiped-p1` | orthogonal darwin-pbpk workstream, CI all GREEN 2026-08-13, 2 files, no claim collision. |
| 1730 | stdlib: insertion-sort break | `darwin-pbpk/stdlib-sort-fix` | narrow stdlib fix, CI all GREEN 2026-08-13, 3 files, no claim collision. |

### REBASE (8) — wanted but path conflicts or stale CI failures; name the conflicts

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 816 | test(madaros): #651 array-of-struct | `work/sr651-madaros-witness` | base=`work/madaros-changed-ci` (not main); CI FAIL on 2026-07-12 (Madaros Changed Tests + CI Decision); re-run on current main to confirm if still red or stale. |
| 817 | test(madaros): generic struct-return | `work/structf-effect-witness` | base=`work/madaros-changed-ci`; CI FAIL on 2026-07-12; same rebase plan as #816. |
| 840 | fix(parser): `study` soft keyword | `fix/parser-study-soft-keyword` | CONFLICTING vs main; closes #740 arm64 parity; small parser change — rebase path should be clean. |
| 1603 | agent-bus realtime | `feat/agent-bus-realtime` | CONFLICTING vs main; touches MCP surface (`scripts/mcp/sounio_coord_mcp.py`) — verify against active `bin/sounio-coord` claim model before rebase. |
| 1721 | darwin_pbpk: Conformal Weaning | `darwin-pbpk/conformal-utiped-p2` | CI FAIL on 2026-08-13 (Full Test Suite + CI Decision); needs re-run on current main — if still red after rebase, close as superseded. |
| 1732 | darwin_pbpk: Proof-Carrying Weaning P3 | `darwin-pbpk/proof-carrying-weaning-p3` | CONFLICTING vs main; title says "Move to sibling repo" — likely already superseded by external move; verify with author before rebase. |
| 1750 | [backend] CUDA ABI launch packing | `cherry/gpu-cuda-fixes-20260815` | CONFLICTING vs main; CI green; touches `bin/souc-lean-single-x86_64` — prebuilt-binary staleness risk (plan §5 risk #6); verify rebuild before merge. |
| 1752 | fix(docs): registry provenance header | `fix-governance-preserve-header-20260816` | CI FAIL on 2026-08-16 (Contracts + CI Decision); docs path; per the wave-2 docs-registry hazard, the failure may be third-party drift, not the diff — verify which before classifying as MERGE. (Note: landed after this triage; CI is now green.) |

### CLOSE (17) — superseded, abandoned, or purpose obsolete; say by what

| # | Title (short) | Head | One-line reason |
|---|---|---|---|
| 795 | feat(lean): seam-flip law ∀n | `lean/cd-seamflip-forall-n` | research-grade Lean; not in wave 1, wave 2, or wave 3 active scope; no founder "let finish" hold; close and reopen later if revived. |
| 978 | feat(render): AA + precise depth | `codex/renderer-quality-20260715` | CONFLICTING; render not in any active plan workstream (WS-A…WS-G); revisit at wave 3 or close. |
| 1034 | fix(compiler): propagate transcendentals | `codex/propagate-runtime-abi-20260716` | DRAFT, untouched since 2026-07-17; subsystem not in active scope; the runtime ABI is now governed by EISA tooling under WS-C PR2 claim. |
| 1053 | fix(ci): compile-fail diagnostic contract | `codex/compile-fail-contract-20260717` | DRAFT, untouched since 2026-07-17; CI-infra not in active scope. |
| 1058 | fix(ssm): exp negative tails | `codex/ssm-exp-tail-20260717` | DRAFT, CONFLICTING, untouched since 2026-07-17; SSM not in active scope. |
| 1069 | fix(madaros): scalar print dispatch | `codex/madaros-ssm-segv-repro-20260717` | DRAFT, untouched since 2026-07-17; subsystem (print dispatch) likely replaced by #1527 or subsequent fix. |
| 1237 | docs(research): rupture algebra | `feat/rupture-synthesis` | CONFLICTING; research-note scope, not in wave 1 or wave 2; revive at wave 3 if interest. |
| 1262 | feat(r3): catalog-bound mapping | `agent/r3-examples-proposal-refresh-20260720` | DRAFT, untouched since 2026-07-20; agent-infra not in active scope. |
| 1297 | feat(probe): LSTM train+probe | `feat/gpu-batched-hyper-syntax` | CONFLICTING; research probe (GPU end-to-end) not in active scope; wave 3+ candidate. |
| 1318 | feat(r3): governed examples extraction | `agent/r3-complete-examples-source-20260720` | DRAFT, untouched since 2026-07-20; supersedes/abandons in favor of #1262. |
| 1451 | feat(research): ord 2″ alignment | `research/rupture-ord2-alignment-20260725` | research instrument, not in active scope; wave 3+. |
| 1466 | Kernel spectrum CD zero-divisors | `research/cd-zd-kernel-spectrum` | research, not in active scope; wave 3+. |
| 1538 | docs(audit): module_frontend segfault | `claude/module-frontend-seed-crash-dispatch` | DRAFT, untouched since 2026-07-27; the segfault is documented elsewhere; superseded by #1737's witness matrix. |
| 1604 | feat(wasm): deontic v3 (dup) | `codex/madaros-wasm-deontic-v3-20260802` | DRAFT, untouched since 2026-08-02; duplicate of #1605 (same head branch) — close as duplicate; #1605 keeps the content if anyone wants it. |
| 1605 | feat(wasm): deontic v3 (dup) | `codex/madaros-wasm-deontic-v3-20260802` | DRAFT, untouched since 2026-08-02; duplicate of #1604; if only one is kept, prefer this one (later PR number, same content). |
| 1641 | docs(ci): self-parse baseline classified | `docs/self-parse-baseline-classified` | **TRAP — CLOSE despite MERGEABLE.** Single fixture file, CI all GREEN 2026-08-04, but main has moved 302 commits since then including #1753 (ENIR/MIR shadow) + #1754 (MLI S1); the classified-baseline fixture is stale by construction — purpose obsolete. |
| 1659 | feat(san-fpga): SAN-v3 curriculum | `research/san-fpga-san-v3-20260805` | CONFLICTING; research-grade SAN, not in active scope; wave 3+ — no founder "let finish" hold, just out of scope. |

### BLOCKED (20) — name the blocker and whether anyone owns it

#### BLOCKED by P0-F claim on `self-hosted/check/check.sio` and/or `self-hosted/native/codegen_x86_linux.sio`

Owner: `claude--session-c89fe8c8-…` lane in `/workspace/.wt/p0f-v2` (active).

| # | Title (short) | Head | Claimed files touched |
|---|---|---|---|
| 867 | checker: contextual function lookup | `agent/issue854-contextual-checker-partial-20260713` | `self-hosted/check/check.sio` |
| 1290 | [checker] affine ownership | `codex/madaros-affine-semantics-20260720` | `self-hosted/check/check.sio` |
| 1339 | fix(madaros): capacity/slicing/wide-call | `agent/madaros-declared-builtin-precedence-20260720` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |
| 1421 | fix(madaros): preserve imported layouts | `codex/issue901-layout-current-20260724` | `self-hosted/check/check.sio` |
| 1527 | fix(madaros): self-parse/Box::new/W044 | `madaros/self-parse-visibility-box-w44-20260727` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |
| 1604 | feat(wasm): deontic v3 | `codex/madaros-wasm-deontic-v3-20260802` | `self-hosted/check/check.sio` |
| 1605 | feat(wasm): deontic v3 | `codex/madaros-wasm-deontic-v3-20260802` | `self-hosted/check/check.sio` |
| 1729 | fix(madaros): B3 IrModule BSS | `fix/lane-b3-ir-module-heap-20260813` | `self-hosted/check/check.sio`, `self-hosted/native/codegen_x86_linux.sio` |

#### BLOCKED by chain ordering (no active owner — chain must be sequenced)

| # | Title (short) | Head | Blocker |
|---|---|---|---|
| 869 | DefId provenance partial #854 | `agent/issue854-defid-provenance-stack-20260713` | chain root #867; DRAFT. |
| 870 | fix(ir): SOIR capacity fail-closed | `codex/ir-serialize-capacity-20260713` | chain tip depends on #869; DRAFT. |
| 881 | feat(ir): heap module bridge | `codex/ir-arena-storage-clean-20260714` | chain root (ir-arena); CI FAIL (Madaros f64 Lowering) on 2026-07-14; "[BLOCKED]" in title. |
| 883 | refactor(ir): bounded SOIR core | `codex/soir-core-split-20260714` | depends on #881; DRAFT; "[BLOCKED]" in title. |
| 885 | fix(ir): materialize bounded heap | `codex/ir-heap-graph-materializer-20260714` | depends on #883; DRAFT; "[BLOCKED]" in title. |
| 979 | test(ir): pin Place binding | `codex/place-canonical-binding-shadow-20260715` | chain root (place-canonical-binding-shadow); DRAFT. |
| 991 | compiler: preserve field receipts | `codex/place-canonical-binding-shadow-20260715` | depends on #979; DRAFT. |
| 998 | feat(resolve): definition registry | `codex/place-canonical-binding-shadow-20260715` | depends on #991; DRAFT. |
| 1155 | [epistemic] psychiatric D0-D8 | `codex/psychiatric-mainline-d0-d2-20260717` | chain root; needs WS-A fresh gate before re-verify; CONFLICTING. |
| 1195 | [epistemic] D9 | `codex/psychiatric-d9-statistical-binding-20260719` | depends on #1155; DRAFT. |
| 1220 | [epistemic] D10 | `codex/psychiatric-d10-deployment-validity-20260719` | depends on #1195; DRAFT. |
| 1243 | [epistemic] D11 | `codex/psychiatric-d11-shift-robust-risk-transport-20260719` | depends on #1220; DRAFT. |

#### BLOCKED by founder decision (held, not in active plan scope)

| # | Title (short) | Head | Blocker |
|---|---|---|---|
| 1580 | CD-tower ZD fibers | `research/zd-fiber-antisymmetry-lemma-20260731` | founder decision 2026-08-16 ("let finish — claude-2 / kimi-cli1 / kimi-cli2 finish PR #1580 first"); base=`research/self-falsifying-compilation-line-20260726` with ~60-commit drift; author's own 2026-08-13 comment: "rebase in a dedicated worktree, then re-open/refresh CI — do not force-merge." Owner: claude-2. |
| 1708 | research(zd): ZD-fiber split | `research/zd-fiber-split-20260810` | depends on #1580; CONFLICTING; wait for #1580 disposition. Owner: claude-2 (held). |

---

## Stale-trap appendix: PRs with green CI but old (the #1641 hazard generalized)

For every PR with last green CI before today's `453b2e6e2f`, the green tick is
evidence about a tree that no longer exists. These are not MERGE without
explicit founder sign-off, even if the API says MERGEABLE.

| PR | Last green CI | Risk class |
|---|---|---|
| #1641 | 2026-08-04 | fixture stale by construction (see header). **CLOSE.** |
| #1554 | 2026-07-29 | narrow stdlib fix, low risk. MERGE-eligible but re-run CI before merge. |
| #1376 | 2026-07-26 | docs-only, no-risk. MERGE-eligible. |
| #1420 | 2026-07-26 | docs-only, no-risk. MERGE-eligible. |
| #1505 | 2026-07-26 | infra shell script, low risk. MERGE-eligible. |
| #1506 | 2026-07-26 | docs-only, no-risk. MERGE-eligible. |
| #1730 | 2026-08-13 | fresh, narrow stdlib fix. **MERGE-eligible.** |
| #1720 | 2026-08-13 | fresh, narrow workstream. **MERGE-eligible.** |

The 2026-08-13 green ticks (last two) were against a `main` closer to current
than the 2026-07-26 to 2026-08-04 set; those are the safer MERGE candidates.

## Coordination summary

- Lane: `minimax-cli1--pr-triage-wave3`. Active. Heartbeats maintained.
- Active claims honoured: no MERGE recommendation conflicts with a claimed file.
- No PRs merged or closed by this triage. Self-hosted/ untouched.
- Cross-deliverable: this doc and `OPEN_PR_TRIAGE_2026-08-16.md` (wave-2) are
  complementary — wave-2 listed 5 specifically-named PRs with the trap caveat
  on #1641; wave-3 widens to the full 53 and enforces the trap uniformly.
- Commit hash for this file's commit on `lane/minimax-cli1/20260815` recorded
  in the PR description if this triage triggers a docs-registry sync.
