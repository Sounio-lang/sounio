# IrModuleArena v2 Shadow Kernel

Status: shadow prototype; not a canonical IR representation
Concept-ID: `SOUNIO-IR-STORAGE-OWNERSHIP`

The old heap bridge proved that moving a fixed `IrModule` off the stack does
not by itself establish one ownership model. Its raw two-GiB reservation is
released by `heap_free`, while the graph reachable through its aggregate
fields is owned separately by either `lean_single`'s never-reset `AGG_POOL` or
native-v2's `RuntimeContext` handle table. Issue #884 records the missing
repeated-use lifecycle. Issue #877 separately records that an owner value can
still be copied or aliased because linear state threading is not yet enforced.

`IrModuleArenaV2` starts a replacement architecture without changing that
legacy path. It is a bounded, columnar shadow kernel with one ownership
authority for its entire graph. Function and instruction slots belong to the
arena itself; there are no child allocations in another runtime. A handle is:

```text
FunctionHandle = (arena_identity, module_identity, function_slot, generation)
InstrHandle    = (arena_identity, module_identity, instruction_slot, generation)
```

The handle types are distinct. Every read and mutation validates arena state,
both stable identities, slot bounds, slot state, and generation. Removing a
slot increments its generation before reuse, so an old handle cannot revive
through ABA. Releasing the arena invalidates all live generations and performs
one logical reclaim. A repeated release returns `ALREADY_RELEASED` and cannot
record a second reclaim.

The tables store scalar columns rather than canonical aggregate values. Raw
setters are private and range-validated; public commands express operations,
and public getters borrow the arena shared. No operation loads a whole
`IrFunction` or `IrInstr`, mutates a local copy, and republishes it.

```text
Semantic-Lane-ID: IR-MODULE-ARENA-V2-SHADOW
Owner: codex/irmodule-arena-v2-shadow-20260714
Concept-IDs: SOUNIO-IR-STORAGE-OWNERSHIP
Intent-Preserved: IR identity and provenance must not disappear across a storage migration.
Transformation: introduce an unintegrated bounded ownership kernel with typed generational handles and one arena authority.
Types-Changed: adds shadow-only IrModuleArenaV2, IrModuleArenaV2FunctionHandle, and IrModuleArenaV2InstrHandle; canonical IR types are unchanged.
Effects-Changed: none in the default compiler; shadow mutation requires Mut, Panic, and Div.
IR-Changed: none in the default compiler or SOIR wire format.
Claims-Introduced: the bounded shadow model rejects stale, ABA, cross-arena, cross-module, released, capacity, and setter-range violations in its named witness.
Claims-Forbidden: canonical replacement; compile-time unique/noncopyable authority; physical memory reclamation; RuntimeContext lifecycle repair; unbounded capacity; SOIR parity; default-pipeline use.
Assumptions: positive arena_identity and module_identity are caller-assigned unique values; production identity minting is not specified by this prototype.
Write-Set: self-hosted/ir/arena_v2_shadow.sio, tests/native-v2/ir_module_arena_v2_shadow_witness.sio, scripts/ci/ir_module_arena_v2_shadow_gate.sh, docs/internal/concepts/ir-module-arena-v2.md.
Read-Set: self-hosted/ir/heap_storage.sio, self-hosted/native/runtime_context.sio, self-hosted/native/gc.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/compiler/lean_single.sio.
Positive-Witness: bounded source-check and runtime probe over typed handles, generation reuse, private validated mutation, capacities, and exactly-once logical release.
Negative-Witness: stale generation, cross-arena identity, cross-module identity, mutation range, capacity overflow, stale-after-release, and double release all fail closed.
Acceptance-Gate: bash scripts/ci/ir_module_arena_v2_shadow_gate.sh.
Integration-Target: shadow-only stack after draft #889; no default compiler integration.
Authoritative-Only-If: a later lane proves differential semantic parity with canonical IrModule and SOIR, resolves #877 authority linearity, resolves #884 repeated-use reclamation, and then explicitly switches readers.
```

## Differential migration

The legacy bridge remains executable and unchanged. It is the first oracle for
the new path, not dead code to be deleted early. A later parity lane should run
the same operation trace against both representations and compare observable
IR identity, DefId provenance, instruction payload, ordering, SOIR bytes, and
failure category. Only an exact receipt may move a reader to v2.

The intended sequence is:

```text
shadow kernel
-> differential operation trace
-> SOIR writer parity
-> bounded repeated-use lifecycle
-> selected read-path switch
-> legacy retirement only after explicit parity authority
```

## Honest boundary

This prototype demonstrates a coherent ownership state machine without using
the two current managed-handle mechanisms. It does not yet enforce unique
authority at the language type level: copying `IrModuleArenaV2` is forbidden
by the contract but remains mechanically possible until #877 is resolved. Its
release is logical because the bounded scalar tables are contained in the
owner value; it is not evidence that native-v2 now reclaims RuntimeContext
handles. Therefore #884 remains open.

```text
Semantic-Outcome: SHADOW_ONLY
Concept-Status-Before: hypothesis
Concept-Status-After: hypothesis; no registry promotion
Distinctions-Added: raw reservation ownership != child graph ownership; logical arena release != RuntimeContext reclamation; generational validity != type-level linear authority
Distinctions-Preserved: stable module identity; function/instruction type distinction; DefId-facing defining module identity; ordering; fail-closed access
Distinctions-Erased: none
Evidence-Run: scripts/ci/ir_module_arena_v2_shadow_gate.sh
Fallback-Path: none
Legacy-Kept: yes, as differential oracle
Conflicting-Lanes: none in this all-new write set
Next-Semantic-Interface: differential trace adapter shared with Place IR and SOIR Writer shadows
```
