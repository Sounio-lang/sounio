<!-- docs:meta
topic_id: repo.docs.decisions.adr-026-phase-arenas
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-026-phase-arenas
-->

# ADR-026: Phase arenas for Madaros self-compilation

**Status**: experimental (implementation pending)
**Date**: 2026-10-31
**Builds on**: [ADR-010](adr-010-madaros-region-reclamation.md) (function-return region reclamation)
**Grounded in**: [`docs/audit/MADAROS_MEMORY_FOOTPRINT_2026-10-22/README.md`](../audit/MADAROS_MEMORY_FOOTPRINT_2026-10-22/README.md)

## Context

ADR-010 reclaims young allocations at function return when dynamic escape
barriers prove the call's temporaries did not escape. That fixes a class of
short-lived helper temporaries, but the Madaros self-build has a larger lifetime
shape: one compiler phase builds a graph that the next phase consumes. HIR
builds MIR; MIR builds IR; IR builds codegen state. These products are too large
and too cross-cutting to be handled by per-function return reclamation alone.

The 2026-10-22 memory audit records two related facts:

- A gen3 self-build in CI fails with `madaros: arena full` (rc 181) even with a
  1.5 GiB stack, so the blocker is heap/arena lifetime, not just stack capacity.
- The seed and Madaros both preserve large compiler-state values too long:
  large by-value state creates BSS/spill/sret pressure in the seed and arena
  pressure in Madaros.

The audit is careful about what was measured and what was only reported. This
ADR inherits that caution: it does not cite the reported 15 GiB gen3 peak or
gen2/gen1 identity as established measurements. It records the design rule that
follows from the measured rc 181 and the localized lifetime mechanism.

## Decision

Madaros compiler phases will allocate phase-local graph data in explicit
**phase arenas**. A phase arena is an upward-only allocation level with a single
owner phase and a declared handoff boundary.

### 1. A level never points upward

Objects in an older or longer-lived level must not hold pointers into a younger
phase arena. A younger level may point down to interned data and other
long-lived tables, but the reverse is forbidden. When a phase hands data to the
next phase, the handoff copies or translates the surviving representation into
the recipient's arena. After that handoff, the producer arena is unreachable and
can die whole.

Legitimate long-lived levels are:

- interned data: strings, symbols, type names and other canonical atoms;
- provenance and diagnostics that must outlive a phase.

Long-lived provenance must use stable indices, not raw pointers into a phase
arena.

### 2. Phase arenas are separate mappings

Each phase arena is backed by its own mapping and is released with `munmap`
when the phase ends. Returning blocks to an allocator free list is not enough:
CI and containers enforce resident memory, and free-list reuse does not
necessarily lower RSS. A phase that is finished must give its pages back to the
operating system.

### 3. Handles and graph references are `u32` indices

Within a phase graph, references between nodes should be stored as `u32`
indices into phase-owned vectors rather than boxed pointers where the cardinality
fits in `u32`. This has three reasons:

1. it reduces per-edge storage from a pointer plus allocator metadata to four
   bytes;
2. it makes the graph relocatable across handoff boundaries;
3. it removes allocator addresses from iteration order, protecting the fixed
   point from allocation-layout changes.

When a graph can exceed `u32::MAX` elements, the phase must fail closed with a
capacity diagnostic or choose an explicitly wider index type. Silent wrap is not
allowed.

### 4. Handoffs are translation boundaries

A phase boundary is not a bag of borrowed pointers. It is a translation from one
owned representation to the next:

- parser/HIR data that is consumed by MIR lowering is copied or encoded into the
  MIR arena;
- MIR data consumed by IR lowering is copied or encoded into the IR arena;
- IR data consumed by codegen is copied or encoded into codegen-owned state.

The boundary is the proof point for releasing the producer arena. If a consumer
still needs part of the producer representation, that part is promoted into a
long-lived arena as indices or interned values before the producer arena is
unmapped.

### 5. Instrument before relying on the design

The compiler must be able to report, at each phase boundary:

- arena name;
- bytes reserved and committed;
- high-water allocation cursor;
- number of graph nodes/edges where applicable;
- bytes copied/promoted into the next phase;
- whether `munmap` was attempted and whether it succeeded.

The self-build gate should record these numbers before and after phase arenas
are introduced. The next investigation should be per-phase rather than only
per-process RSS.

### 6. Seed pressure is related but separate

The seed's BSS/spill/sret pressure is a separate implementation issue. Pooling
large `sret` and spill blocks per function on every target is compatible with
this ADR and should be tracked as a separate task. This ADR does not make that
seed change.

## Consequences

- ADR-010 remains the function-local reclamation rule. ADR-026 is the phase
  lifetime rule for data that crosses function calls but does not need to cross
  compiler phases.
- Compiler phase APIs must stop exposing raw pointers into phase-owned graphs as
  long-lived values.
- Provenance and diagnostics need explicit stable identities. They cannot rely
  on node addresses after a phase boundary.
- `u32` indices add capacity checks. A graph that outgrows its index type must
  be rejected with a clear diagnostic instead of wrapping.
- The first implementation can be incremental: instrument phase sizes, introduce
  one phase arena, prove its handoff contains no upward pointers, then expand to
  the next phase.

## Refusal cases

Implementation must fail closed when:

- a long-lived object would store a pointer into a phase arena;
- a phase graph exceeds the chosen index width;
- a phase boundary cannot translate all required survivors into the recipient
  arena or a long-lived arena;
- `munmap` fails;
- instrumentation is disabled in a self-build gate that depends on phase arena
  evidence.

These refusals should be diagnostics, not silent fallback to unbounded lifetime.

## Acceptance

1. A self-build run logs phase arena high-water marks and release points for at
   least HIR, MIR, IR and codegen phases.
2. At least one large producer phase is backed by a separately mapped arena and
   releases it with `munmap` after translating survivors into the next phase.
3. A test or gate demonstrates that no long-lived provenance record stores a raw
   pointer into a released phase arena; it uses stable indices instead.
4. A capacity test demonstrates that a phase graph exceeding its index width is
   refused rather than wrapping.
5. The gen3 rc 181 investigation is rerun with per-phase measurements, so the
   next report can attribute memory to phases rather than only to process RSS.

## Implementation tasks

Split implementation out of this ADR into separate cards:

- phase-size instrumentation and self-build reporting;
- first phase arena and `munmap` release at one producer/consumer boundary;
- `u32` graph indices plus capacity diagnostics;
- provenance/diagnostic promotion to stable identities;
- seed large-spill/sret pooling on non-macOS targets.

## Amendment (2026-11-09): field resolution

Added by [ADR-028](adr-028-rulings-2026-11-09.md); the sections above are unchanged.

A phase hands the next phase typed data, so a field is resolved by the type that owns it, never by its name. Resolving a field by a hash of its name, or by a majority vote across the structs that declare a field with that name, is forbidden in every phase of the HIR → MIR → IR → codegen pipeline. Where the type of a field access is unknown at lowering, lowering refuses rather than guesses.

Acceptance: the Stage 0 gate (`lower_refuse_untyped_field_hash` in `self-hosted/ir/lower.sio`, extracted from PR #3045) is on `main` and refuses an untyped field access that a name hash would have resolved.
