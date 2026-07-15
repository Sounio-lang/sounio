<!-- docs:meta
topic_id: repo.docs.internal.concepts.ir-module-arena-v2-place-shadow
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ir-module-arena-v2-place-shadow
-->

# IrModuleArena v2 Place IR Shadow

Status: executable structural witness; off-default and not parity
Concept-ID: `SOUNIO-IR-STORAGE-OWNERSHIP`

This slice gives the Arena v2 module store a bounded, generation-checked Place
identity without importing it into the parser, canonical lowering, SOIR,
serialization, or a backend. It reuses the semantic vocabulary characterized
by draft PRs #894 and #910, but does not reuse their aggregate Place values or
test-local wire representation.

The representation is deliberately split:

```text
Place identity =
  PlaceId(slot, generation)
  + ModuleId(arena, slot, generation)
  + module mutation/layout epoch
  + root identity, type, layout
  + ordered field projections

Field projection identity =
  owner type, owner layout, field ordinal
  + result type, result layout

Access instance =
  read or write event, write policy, event order
```

Access events are not part of structural Place equality. A field-name hash is
stored only to construct a deliberate same-hash collision witness. It
never selects a field and cannot establish name equality by itself; the shadow
classifies only a same-hash structural collision. Field ordinal
is separate from spelling and from any future physical byte offset.

A Place is built and then finalized. Projection append is permitted only while
building; structural comparison and access events are permitted only after
finalization. Thus a published `PlaceId` cannot silently change structural
meaning after it becomes usable. Write authorization is supplied to the store
event rather than stored in Place identity.

Every semantic Place read or mutation first validates the live `ModuleId`, then the
live `PlaceId`, then the exact root binding and the module mutation epoch. A
module mutation therefore invalidates an earlier Place even when its module
slot and generation remain live. Place release increments its own generation
before slot reuse. A lifecycle-only discard validates only the generational
`PlaceId`, so an invalid module or epoch cannot leak bounded capacity; it does
not permit any semantic read. This authority remains copyable until the wider
Arena linearity blocker in issue #877 is resolved. The implementation uses
private scalar columns only; no
Place store, projection array, Place value, or projection value crosses an API
boundary.

## Executable differential

The structural shadow creates two modules with structs whose field spelling is
the same but whose owner type, layout, field ordinal, and result identity are
different. It records the second struct's declared ordinal 1, rejects hash-only
equivalence, validates a nested type/layout transition, records load before
store, and rejects stale ModuleId, reused ModuleId, stale PlaceId, capacity,
and projection overflow.

The comparable legacy program executes through the current compiler. Its typed
local control observes `B.shared == 42`, but `make_b().shared` exits with `11`.
The current lowering loses the nominal base of the call expression and its
global name fallback selects ordinal 0 from a different struct. This is a real
differentiating observation, not parity:

```text
mode=place_structural_witness_not_parity
shadow declared structural record: B.shared ordinal 1
legacy typed control: 42
legacy call expression: rc 11
```

The legacy path remains unchanged as a control and counterexample. This shadow
does not yet load or store the program value, emit a backend Place operation,
or serialize a logical Place. Root, type, layout, ordinal, and result IDs are
caller-supplied declarations checked for chain consistency; they are not yet
resolved against a canonical module/type/layout table. It therefore cannot
claim field selection authority or replace current lowering.

The full adversary matrix runs as a throwaway composite module. A separate
minimal imported probe executes calls carrying `ModuleId`, `PlaceId`, projection
scalars, finalization, and access arguments across the real module boundary.
That probe is an API transport smoke test, not imported execution of every
adversary or evidence of default wiring.

## Prior-art boundary

Root plus ordered projections is established prior art. Rust MIR represents a
Place as a local plus ordered projections and derives types stepwise. LLVM GEP
preserves a base pointer, source element type, address space, and ordered
structural indices. MLIR MemRef preserves explicit layout and memory-space
information. Swift SIL separates storage identity, access paths, formal access
scope, exclusivity, and value ownership.

The defensible Sounio question is narrower and remains unproven beyond this
bounded witness: can every serialization, rekey, lowering, and backend access
revalidate `(ModuleId, PlaceId, semantic/layout epoch)` and reject ABA or path
drift before executing the operation? This lane proves only the first
in-memory structural component.

Primary sources, pinned and accessed 2026-07-15:

- Rust MIR `Place` and `ProjectionElem`, rust-lang/rust
  `47101adcea71daee3c2879218f5b883bcdf180aa`.
- LLVM `getelementptr` specification and design notes, llvm-project
  `0350a23a8bbc091e646406a2aaeafa35dc0216d3`.
- MLIR LLVM `GEPOp`, MemRef and data-layout documentation, llvm-project at the
  same commit.
- Swift SIL memory-access and ownership documentation, swiftlang/swift
  `e9e1fa9e028bb8dcff68550afb939ac3103a8702`.

## Semantic lane

```text
Semantic-Lane-ID: IR-MODULE-ARENA-V2-PLACE-SHADOW
Owner: codex/place-ir-arena-v2-shadow-20260715
Concept-IDs: SOUNIO-IR-STORAGE-OWNERSHIP
Intent-Preserved: exact structural identity, projection order, and compiler-caused misselection remain observable rather than collapsing into a field spelling.
Transformation: add an off-default scalar PlaceId authority bound to a live ModuleId and mutation/layout epoch, plus an executable legacy counterexample.
Types-Changed: adds shadow-only IrModuleArenaV2PlaceId; canonical IR and borrow-checker Place types are unchanged.
Effects-Changed: none in the default compiler; shadow allocation, projection append, release, and access receipts require Mut.
IR-Changed: none in canonical IR, SOIR, serializer, writer, or backend routing.
Claims-Introduced: the exact source-fresh gate proves bounded caller-declared structural recording, self-consistent nested type transitions, access-event ordering, fail-closed generation/epoch adversaries, and a minimal imported API smoke.
Claims-Forbidden: first Place IR; canonical alias identity; provenance completeness; default-pipeline correctness; backend readiness; SOIR parity; serializer parity; performance superiority; replacement of legacy lowering.
Assumptions: module mutation epoch changes for every future type/layout-affecting mutation; caller-declared IDs are not canonical type-table proof; this first slice supports root-module-defined fields and two field projections only.
Write-Set: self-hosted/ir/arena_v2_place_shadow.sio, self-hosted/ir/arena_v2_place_import_probe.sio, tests/native-v2/place_ir_arena_v2_shadow_witness.sio, tests/native-v2/place_ir_legacy_nominal_collision_witness.sio, scripts/ci/place_ir_arena_v2_shadow_gate.sh, docs/internal/concepts/ir-module-arena-v2-place-shadow.md, generated docs governance metadata.
Read-Set: self-hosted/ir/arena_v2_shadow.sio, self-hosted/ir/lower.sio, draft PR #894, draft PR #910.
Positive-Witness: declared ordinal recording, self-consistent nested transition, finalized load-before-store, two-module structural distinction, Place generation reuse, stale lifecycle discard, distinct capacity receipts, and imported API transport smoke.
Negative-Witness: same-hash structural collision, type-chain mismatch, access before finalization, append after finalization, store-before-load, unauthorized store, stale module, reused module, stale Place generation, and module-epoch drift reject.
Acceptance-Gate: SOUC_BIN=<exact-source-fresh-madaros> bash scripts/ci/place_ir_arena_v2_shadow_gate.sh.
Integration-Target: shadow-only stack on PR #965; no default reader or writer switch.
Authoritative-Only-If: a later lane serializes and rekeys logical Places, executes the same values and mutations through both paths, matches every valid legacy observation, and explicitly resolves the current legacy misselection before any reader switch.
```

## Integration receipt

```text
Semantic-Outcome: PLACE_STRUCTURAL_WITNESS_NOT_PARITY
Concept-Status-Before: hypothesis
Concept-Status-After: hypothesis; no registry promotion
Distinctions-Added: field spelling != name hash != structural field identity; declared identity != canonical resolution; Place identity != access instance; live slot != live generation/epoch binding; differentiating observation != parity
Distinctions-Preserved: module identity and generation; projection order; type/layout transitions; legacy counterexample availability; fail-closed validation
Distinctions-Erased: none
Evidence-Run: scripts/ci/place_ir_arena_v2_shadow_gate.sh with source-fresh compiler sha256 7ad961849c398ba6dc709dd327f14f14405062bb12638d48374d8619a1ba8372
Fallback-Path: none
Legacy-Kept: yes, byte-identical to base and executed as the current control/counterexample, not a parity oracle
Conflicting-Lanes: draft PRs #894 and #910 remain discovery material only; neither was merged or cherry-picked
Next-Semantic-Interface: canonical module/type/layout resolver, then logical Place serialization and explicit rekey, followed by value/load/store differential execution and zero-backend-op rejection receipts
```
