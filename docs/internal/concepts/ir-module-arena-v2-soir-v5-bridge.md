<!-- docs:meta
topic_id: repo.docs.internal.concepts.ir-module-arena-v2-soir-v5-bridge
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ir-module-arena-v2-soir-v5-bridge
-->

# IrModuleArena v2 to SOIR v5 Shadow Bridge

Status: executable shadow slice; not differential and not default-wired
Concept-ID: `SOUNIO-IR-STORAGE-OWNERSHIP`

This slice connects the generation-checked `IrModuleArenaV2ModuleId` to the
existing off-default SOIR v5 writer without transporting `IrModule`, a module
store, a projection aggregate, or a complete writer plan across calls.

The first implementation exposed a real native-v2 boundary: an aggregate plan
returned by preflight and borrowed by emit produced correct isolated results,
but a stale rejection followed by slot reuse in the same process returned an
incorrect scalar status. A mutated module likewise produced a corrupted failed
preflight plan status. In both cases the arena state, plan fields, and the full
131,072-byte canary remained unchanged.

The accepted bridge therefore uses the same ownership shape as the arena:

```text
PlanId = (plan_slot, plan_generation)

private plan columns =
  arena_identity
  module_slot
  module_generation
  mutation_epoch fingerprint
  start
  capacity
  required
  end
  version
```

Preflight returns a scalar status and publishes only `PlanId` through a caller
out-parameter after success. Emit revalidates the live PlanId, ModuleId identity
and generation, the exact-empty state plus its monotonic mutation epoch, and a
freshly computed writer plan before any byte is written. The full
`SoirWritePlan` exists only inside a single bridge call. The plan store is
deliberately bounded to two private scalar slots; exhaustion has a distinct
status, and explicit release increments the plan generation before reuse.

The bridge is currently origin-only. Direct scalar writer execution correctly
emits at a nonzero start, but forwarding the mutable 128 KiB buffer through the
additional bridge call loses the nonzero cursor in the current native backend.
The bridge therefore rejects `start != 0` before write.

## Evidence boundary

The gate mechanically extracts marked plan, shared primitive, and empty-module
blocks from `self-hosted/ir/soir_writer.sio` and the maximum-size and extension
count constants from `self-hosted/ir/soir_core.sio`. The fixed empty-size
arithmetic remains local to the writer extension. The gate does not copy wire
tags or maintain a parallel hand-written golden.

The source-fresh witness proves an exact 320-byte origin emission, deterministic
repeat, exact-capacity success, below-capacity rejection, full-buffer canary
preservation, and rejection of invalid, stale, reused, cross-arena, and mutated
module identities. A no-helper sequential probe proves stale rejection followed
by reused-slot rejection in the same process. Reversible BSS mutation is also
rejected because its epoch changes even when the visible value returns to zero.
A second preflight while nonempty returns the scalar module-contract status.
Plan capacity, failed same-out publication, release, and generation reuse have
their own executable lifecycle witness.

This is `shadow_canonical_not_differential`, where `canonical` means that the
bytes come from the existing writer-owned code path, not that byte correctness
or legacy parity has been established. The current positive byte evidence is
only deterministic self-repeat. The real legacy witness was run
with the exact parent compiler artifact and reached native execution, but its
large `IrModule` transport expanded to 2,048 functions, warned about
404,144,224-byte stack frames, and printed `FAIL soir_writer_v0_differential`.
The full imported writer-plus-bridge closure separately reached 1,867 functions
and stopped in the backend with `rc=19`. The executable matrix concatenates the
marked sources into one throwaway module, so it does not prove the imported
bridge-to-writer aggregate ABI. No legacy parity receipt exists, so no byte
parity or canonical replacement is claimed.

## Prior art boundary

Two-phase serialization, sized sections, versioned binary formats, canonical
encoding, and rejection before external commit are established techniques.
LLVM bitstream uses sized blocks and backpatching; WebAssembly uses versioned,
length-delimited sections without requiring one canonical byte encoding; MLIR
buffers bytecode before its final stream write; Cap'n Proto specifies exact
framing and an optional canonical form. The defensible Sounio contribution at
this stage is only the tested combination of a generation-bound scalar PlanId,
an exact semantic fingerprint, caller-owned output, and full-canary rejection
receipts without a SOIR-sized staging buffer. No priority, performance, or
generality claim follows from this bounded comparison.

Sources: LLVM Bitcode Format and BitstreamWriter; WebAssembly Core 3.0 binary
modules; MLIR Bytecode Format and BytecodeWriter; Cap'n Proto Encoding. The
bounded architecture packet is `/tmp/sounio-moduleid-soir-v5-mini-survey.md` in
the implementation environment and is not a repository authority.

```text
Semantic-Lane-ID: IR-MODULE-ARENA-V2-SOIR-V5-BRIDGE
Owner: codex/irmodule-soir-v5-bridge-20260715
Concept-IDs: SOUNIO-IR-STORAGE-OWNERSHIP
Intent-Preserved: module identity, generation, semantic fingerprint, writer version, exact length, and rejection-before-write remain observable.
Transformation: add a shadow-only scalar PlanId store and delegate exact-empty SOIR v5 bytes to the existing writer core.
Types-Changed: adds IrModuleArenaV2SoirPlanId only; canonical IrModule and SOIR types are unchanged.
Effects-Changed: none in the default compiler; shadow preflight and emit require existing mutation and bounds effects.
IR-Changed: none in the default compiler and no SOIR wire tag changes.
Claims-Introduced: the named source-fresh composite gate proves the bounded origin-only exact-empty writer-owned path and all listed identity adversaries.
Claims-Forbidden: legacy byte parity; general SOIR v5 parity; nonempty module support; nonzero-start support; default routing; canonical replacement; novelty priority; zero-copy or performance superiority.
Assumptions: caller initializes the bounded plan store and explicitly releases live plans; mutation epochs expand with every future semantic setter before nonempty support.
Write-Set: self-hosted/ir/arena_v2_shadow.sio, self-hosted/ir/soir_writer.sio, self-hosted/ir/arena_v2_soir_bridge.sio, tests/native-v2/ir_module_arena_v2_soir_v5_bridge_*.sio, scripts/ci/ir_module_arena_v2_shadow_gate.sh, scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh, docs/internal/concepts/ir-module-arena-v2-soir-v5-bridge.md.
Read-Set: self-hosted/ir/soir_core.sio, self-hosted/ir/serialize.sio, self-hosted/ir/heap_storage.sio, self-hosted/ir/mod.sio, self-hosted/compiler/main.sio.
Positive-Witness: exact length, deterministic self-repeat, exact capacity, writer-owned bytes, sequential stale then reuse rejection, reversible mutation rejection, mutation repreflight rejection, and bounded plan lifecycle.
Negative-Witness: invalid ID, stale ID, reused generation, cross-arena identity, semantic mutation, insufficient capacity, nonzero origin, and invalid PlanId fail before write.
Acceptance-Gate: SOUC_BIN=<exact-source-fresh-madaros> bash scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh.
Integration-Target: shadow-only stack on the IrModuleArena v2 parent; no default compiler reader changes.
Authoritative-Only-If: a real same-artifact legacy oracle emits bytes and the bridge matches them exactly before any reader switch.
```

## Integration receipt

```text
Semantic-Outcome: SHADOW_CANONICAL_NOT_DIFFERENTIAL
Concept-Status-Before: hypothesis
Concept-Status-After: hypothesis; no registry promotion
Distinctions-Added: aggregate plan transport != scalar generational plan identity; deterministic self-repeat != legacy differential parity
Distinctions-Preserved: ModuleId arena identity; module slot and generation; empty semantic counts; SOIR v5 writer ownership; exact required/end; fail-closed rejection
Distinctions-Erased: none
Evidence-Run: scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh with source-fresh compiler sha256 7ad961849c398ba6dc709dd327f14f14405062bb12638d48374d8619a1ba8372
Fallback-Path: none
Legacy-Kept: yes, unchanged as the required future differential oracle
Conflicting-Lanes: none in the declared shadow write set
Remaining-Blockers: real legacy byte oracle; byte correctness beyond self-repeat; nonzero-start nested mutable-buffer forwarding; imported bridge-to-writer ABI; full imported-closure backend rc19
Next-Semantic-Interface: extend scalar fingerprints and plan columns section by section, then rerun a real same-artifact differential oracle
```
