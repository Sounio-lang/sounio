<!-- docs:meta
topic_id: repo.docs.audit.rebracket-assoc-gates-audit-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: kimi-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.rebracket-assoc-gates-audit-2026-08-19
-->

# The ten rebracket/associator gates: wired 1/10, run 6/10, and what the four reds actually say (2026-08-19)

**Lane:** kimi-cli1 / `rebracket-gates-audit` (dispatch: `/tmp/dispatch_rebracket_claude1.md`)
**Measured:** `origin/main @ f9b3147364`, in a clean worktree off it.
**Governing concept:** `docs/internal/concepts/rebracketing-authority.md` — SOUNIO-REBRACKETING-AUTHORITY, `last_validated 2026-03-07` (five months stale).

## Semantic statement (mandatory per dispatch)

This audit **changed no gate, no wiring, and no source**. It ran gates read-only against main and classified the results. Nothing was reverted (founder rule, 2026-08-19).

## 1) Wiring census — 1/10, confirmed with a validated instrument

Instrument: `git grep -c <basename> origin/main -- .github/`. Validated per dispatch BEFORE trusting zeros: positive controls `sedenion_associator_1848_gate` → 1 (wired), `concept_status_gate` → 1 (entered ci.yml today) — the instrument sees wiring when it exists.

Only `sedenion_associator_1848_gate` is wired. **Unwired: 9/10 — including `exact_bitwise_rebracket_authority_gate.sh`, the gate the concept document itself names as its executable authority, and `proof_carrying_rebracketing_protocol_gate.sh`.** The concept's claim "executable under strict hash-bound gates" is currently enforced in CI by exactly one gate — and that one (sedenion 1848) validates associator combinatorics, not rebracketing authority.

## 2) All ten, run hermetically (rc per gate, not aggregated)

Environment: clean worktree off `f9b3147364`; `ulimit -s 1048576` + `MADAROS_STACK_KB=524288` (CI parity — a bare pod run segfaults everything on the 8 MB default); `SOUC_BIN` pinned to the worktree's `bin/souc` after the first pass was found contaminated: `resolve_souc.sh` walked up to `/workspace/sounio/bin/souc` (the control worktree, parked on a research branch). All rc's below are from the hermetic re-run; first-pass rc's matched, so the contamination did not change verdicts — but it is recorded as instrument finding.

| # | Gate | rc | Classification |
|---|---|---|---|
| 1 | sedenion_associator_1848_gate | **0** | VERDE (wired AND passing; cross-toolchain souc-vs-Python-oracle, 1848=11·168) |
| 2 | associator_gum_variance_gate | **0** | VERDE (experiment completes, verdict=PASS from receipt) |
| 3 | dyadic_relational_associator_gate | **0** | VERDE (Madaros-identity checks pass) |
| 4 | kretikos_kaxi_phase_z_assoc_gate | **0** | VERDE (Phase Z scaffold, 8-comp associator PBPK) |
| 5 | proof_carrying_policy_observation_associator_gate | **0** | VERDE (clinical policy observation rejection suite) |
| 6 | proof_carrying_rebracketing_protocol_gate | **0** | VERDE (protocol receipts; includes honest-limitations text anchors) |
| 7 | exact_bitwise_rebracket_authority_gate | **1** | **VERMELHO-PODRE** (gate rotted; the law it checks is intact — see §3) |
| 8 | exact_bitwise_rebracket_source_ir_gate | **1** | **VERMELHO-REAL do RUNTIME** (gate is fine; the Madaros stdout-to-file defect kills it silently — see §4) |
| 9 | native_v2_associator_gate | **1** | **VERMELHO-REAL do COMPILADOR** (driver broken by the #1717 SoA landing's effect cascade — see §5) |
| 10 | kretikos_associator_emit_gate | **1** | **VERMELHO-PODRE** (driver imports a file that no longer exists — see §6) |

The dispatch's suspicion was right in shape: total green would have meant a dead instrument. Six green is consistent with a working instrument; the four reds have four DIFFERENT root causes, which is what a real census looks like.

## 3) exact_bitwise_rebracket_authority_gate — VERMELHO-PODRE, and the law is INTACT

Fails at: `missing production anchor 'opt_cleanup_module_inplace(&! merged_module)' in module_frontend.sio`.

Root cause, verified in git history: commit `04c8ba89f4` (2026-07-20, "finalize multi-mod IR in place on &! IrModule") refactored the call from `(&! merged_module)` to `(&! (*module_box))` — an ownership fix, taking the module by in-place borrow instead of by value. **The cleanup call with the rebracket receipt still exists** (module_frontend.sio lines 6502/6588, `single-post-resolve` / `merged-post-finalize` receipts both print). The gate's text anchors were never updated after the refactor.

The law side of the gate (negative assertions: exact combiner must not admit Add/Mul; canonical scalar slice must refuse 16 opcodes; strict mode requires ELF magic + SHA-256-bound compiler + clean worktree) was not touched by the drift. **Fix: update 2 anchors to the current call shape (or anchor on the receipt calls, which are the semantic surface). Cheap, and this is the concept's named authority gate — first in the rewiring queue once rewiring is safe.**

## 4) exact_bitwise_rebracket_source_ir_gate — killed SILENTLY by the stdout defect

A/B, deterministic, both directions:

- stdout → file: **rc=1, zero output** (the CI shape)
- stdout → pipe: **rc=0** (the interactive shape)

This is the same Madaros runtime stdout-to-file defect family as the kaxi segfault in `DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-18.md` (PR #1914, merged) — here manifesting as a silent non-zero exit rather than a segfault. The gate's own logic never runs its assertions in the file case. **Classification: the gate is fine; the runtime defect owns this red. Dispatched already via #1914; re-ping after the runtime fix lands.**

Instrument warning this adds: **a gate that exits non-zero with an EMPTY log is not a measurement of anything.** Any harness treating rc alone would mis-file this as a real defect.

## 5) native_v2_associator_gate — the driver is broken by the SoA effect cascade

The spine dies at stage1: the driver `self-hosted/compiler/native_compile_driver.sio` fails type-check across 47 modules under a current-source Madaros. Errors (first of ~47):

```
error[E035] ... emit_instr ... (missing: Div) -- required by `ir_region_slot_w`
error[E035] ... emit_instr ... (missing: Div) -- required by `ir_arena_store`
error[E012] ... knowledge_runtime_guard_lowering_plan ... this type has no field named
```

This is the documented #1717 aftermath (IR arena/SoA landing, 2026-08-12): the arena accessors carry `Mut, Panic, Div`, every reader inherits them, and the cascade climbs the call graph — that landing took `123 → 10 → 4 → 0` declarations across three rounds **in the files it touched**, and a `pub(crate) fn` sweep escaped. `native_compile_driver.sio` never received its cascade round, and this gate — unwired — was the thing that would have caught it. Nothing rebracketing-specific: the driver cannot type-check at all. **Real compiler defect (E035 cascade + E012 renamed fields in knowledge_runtime_guard), owner: the SoA follow-up lane.**

## 6) kretikos_associator_emit_gate — driver imports an evaporated file

`./bin/kretikos emit-kaxi octonion_associator` → `error: failed to compile K-AXI emitter driver`; bare reproduction: the driver `self-hosted/gpu/kretikos_emit_kaxi.sio` imports `self-hosted/gpu/erdos90_hc_smoke_emit.sio` (line 34), **which does not exist on main** (history: removed in the erdos WIP-snapshot cleanup, `b8828063d6` era). Import evaporated under the driver. Gate rotted; the emitter driver needs either the import dropped or the file restored. Not a semantic defect.

## 7) THE DISPATCH'S ESCALATION QUESTION — is any red an EFFECTIVE REBRACKETING on a non-associative type?

**No. Verified per gate:**

- The two authority-family gates (7, 8) never reached their assertions — 7 on a stale text anchor before compiling anything, 8 on the stdout defect. Neither accuses the compiler of regrouping anything.
- The six greens include the two that would catch unauthorized regrouping if it compiled: `proof_carrying_rebracketing_protocol` (receipt discipline: check receipts, expected/found pairings, honest-limitations anchors) and `proof_carrying_policy_observation_associator` (rejection suite) — both green means the receipt protocol still holds for everything that compiles.
- The two broken-driver gates (9, 10) fail at **type-check / import** — before any lowering, any bracketing, any optimization. A driver that cannot type-check cannot silently regroup.

So: **no gate accuses the founder's thesis of being violated.** The accurate statement is worse in a different way: for the authority gate specifically, nothing is checking the law in CI at all (unwired), and the local copy rotted on an anchor — the thesis is unguarded, not violated.

## 8) Coverage map — what the ten validate as a set

| Layer | Gates | Status of the layer |
|---|---|---|
| **Rebracketing authority (the concept)** | exact_bitwise_rebracket_authority, exact_bitwise_rebracket_source_ir | **Uncovered in CI** (unwired); locally: one rotted anchor, one killed by the runtime stdout defect. The "strict hash-bound gates" claim is presently enforced by zero CI gates. |
| **Rebracket receipts / protocol discipline** | proof_carrying_rebracketing_protocol, proof_carrying_policy_observation_associator | Covered and green — for code that type-checks. |
| **Associator algebra (sedenion/octonion combinatorics)** | sedenion_associator_1848 (wired), dyadic_relational_associator | Covered, green, cross-toolchecked. |
| **Associator → GUM variance** | associator_gum_variance, kretikos_kaxi_phase_z_assoc | Covered, green (experiment + Phase Z scaffold). |
| **Associator → GPU emission (kretikos)** | kretikos_associator_emit (+ phase_z above) | Broken driver (evaporated import). |
| **Associator → native v2 science spine** | native_v2_associator | Broken driver (SoA cascade) — blocks the whole native-v2 spine it rides on, not just associator cases. |

**What non-associativity coverage lacks:** the algebra, the GUM propagation, and the receipt discipline are each independently checked; what has NO living enforcement is the compiler-side authority itself — the thing the concept document says a compiler may do "only for one identified occurrence after discharging the law, representation, ordering…". The e-graph's 1000+ unvalidated rewrite rules (dispatch context) sit exactly in that hole.

## 9) Rewiring ORDER (proposal only — nothing was wired, per dispatch §3)

Main is red for an unrelated reason (stale known-failure tag, other lane); wiring red gates now would misattribute. When that clears:

1. **exact_bitwise_rebracket_authority** — after the 2-anchor refresh (§3). The concept's own authority gate; highest semantic value per effort.
2. **kretikos_associator_emit** — after the evaporated import is resolved (drop import or restore file).
3. **native_v2_associator** — after the SoA cascade round for native_compile_driver lands; riding it earlier would gate main on a 47-module fix.
4. **exact_bitwise_rebracket_source_ir** — after the runtime stdout-to-file fix (PR #1914's dispatch); wiring it now would make CI fail on the stdout shape, not on source→IR truth.
5. The six greens: wire `proof_carrying_rebracketing_protocol` and `proof_carrying_policy_observation_associator` first among them — they are the thesis's actual living enforcement.

Separately: `docs/internal/concepts/rebracketing-authority.md` `last_validated` is 2026-03-07. Five months. The registry should not carry an executable-authority claim with zero wired gates — either the gates get wired or the concept's wording gets an honest "not currently enforced in CI" note. That is a governance decision, not mine to make.

## 10) Instrument findings (this audit's own traps, for the next lane)

1. `resolve_souc.sh` can walk out of the worktree to `/workspace/sounio/bin/souc` — pin `SOUC_BIN` absolutely in any gate measurement.
2. A gate rc=1 with an EMPTY log measures nothing (§4) — always `wc -l` the log before believing the rc.
3. The 8 MB pod stack segfaults every Madaros `run` — CI stack env is mandatory for any local gate run.
4. First-pass vs hermetic rc's matched here, but only because both were compared — record both, always.
