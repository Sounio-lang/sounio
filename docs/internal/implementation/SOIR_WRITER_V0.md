<!-- docs:meta
topic_id: repo.docs.internal.implementation.soir-writer-v0
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.soir-writer-v0
-->

# SOIR Writer v0

Status: off-default prototype

Base: draft PR #889, `17b0858f6e7d75c9cfc9e545b1b9f0805fa9d5d6`

Concept-ID: proposed `compiler.soir.writer.v0`

## Semantic Lane

```text
Semantic-Lane-ID: compiler-soir-writer-v0-20260714
Owner: Codex / soir_writer_v0
Concept-IDs: proposed compiler.soir.writer.v0
Intent-Preserved: SOIR v5 bytes and rejection semantics remain explicit and testable
Transformation: add an off-default planned writer over a caller-owned arena view
Types-Changed: none; introduces opaque SoirWritePlan and private SoirWriterCursor
Effects-Changed: none
IR-Changed: none
Claims-Introduced: checker-valid writer implementation; static wire-tag parity
Claims-Forbidden: runtime byte parity, default integration, v4 emission, performance
Assumptions: current SOIR v5 empty-extension layout is the differential baseline
Write-Set: four all-new files listed by this checkpoint
Read-Set: serialize.sio, soir_core.sio, heap_storage.sio, ir.sio
Positive-Witness: legacy-vs-writer length and byte comparison for exactly two v5 fixtures
Negative-Witness: full-buffer canaries for every declared input-rejection status exercised by v0
Acceptance-Gate: SOIR_WRITER_REQUIRE_DYNAMIC=1 bash scripts/ci/soir_writer_v0_gate.sh
Integration-Target: none; stacked draft over PR #889
Authoritative-Only-If: runtime differential gate passes locally and remotely
```

## Intent

SOIR Writer v0 preserves the existing SOIR v5 bytes while removing two
implementation accidents from the candidate write path:

1. the 128 KiB artifact buffer is never passed or returned by value;
2. writer failure state is not stored in a mutable global.

This prototype does not replace `ir::serialize` or `ir::soir_core`. The legacy
serializer remains the differential oracle and the default compiler path until
the new writer proves parity on a declared surface.

## Contract

The writer is split into two phases:

```text
preflight(module, arena_view) -> SoirWritePlan
emit(module, accepted_plan, caller_buffer) -> absolute_end | status
```

`SoirWritePlan` records `start`, `capacity`, `required`, `end`, `version`, and
`status`. Its fields are private. `SoirWriterCursor` is private and contains
only `pos`, `limit`, and `status`; it never owns the byte buffer.

Preflight validates before the first byte is written:

- module function/string counts and per-function instruction/parameter counts
  against `IR_MAX_FUNCS`, `IR_MAX_STRINGS`, `IR_MAX_INSTRS`, and
  `IR_MAX_PARAMS`;
- zero BSS, because SOIR v5 does not encode BSS sizes;
- empty epistemic and algebra extensions for the v0 subset;
- every function, instruction, and string `Name` length;
- every opcode against the existing SOIR v5 tag set;
- exact byte length inside the caller's `start + capacity` arena view.

Emit re-runs preflight and requires the result to match the opaque plan. An
input rejection therefore occurs before cursor creation, leaving every byte in
the caller buffer unchanged. Success returns the absolute end cursor; the byte
length is `end - start`.

## Supported Surface

Writer v0 emits only:

- SOIR v5;
- function and instruction prefix fields already encoded by the legacy path;
- the current stable opcode tags 0 through 36;
- strings;
- empty epistemic and algebra extension sections.

SOIR v4 is decode/golden-only. Writer v0 has no v4 emit entrypoint. The
differential witness derives a v4 artifact by removing the v5 provenance word
and verifies legacy decode behavior (`defining_module_id = UNKNOWN`).
The writer pins its own version constant to `5`; a future default SOIR version
cannot silently change the bytes emitted by this v0 contract.

## Input-Rejection Atomicity

Input-rejection atomicity is an observable contract, not an implementation comment.
The witness fills the caller buffer with a canary and checks all 131072 bytes
after rejected capacity, count bounds, module contract, stale plan,
unsupported-opcode, and invalid-name inputs.

The writer does not allocate a private 128 KiB scratch buffer. Its atomicity
comes from complete preflight, an opaque plan, and an exact-size cursor. Any
post-write cursor failure is an internal invariant breach and keeps this lane
from promotion; it is not a fallback to partial output. `SOIR_WRITER_INTERNAL_BOUNDS`
may be reported after bytes were written, so it does not carry the unchanged-buffer
guarantee and consumers must discard the entire candidate output.

## Differential Oracle

`self-hosted/test_soir_writer_v0.sio` compares length and every emitted byte
against `serialize_ir_module_into` for:

- an empty module (`320` bytes);
- one function, two instructions, and one string (`1632` bytes).

It also pins v5 anchors for magic/version, function provenance at offset 720,
and empty extensions at offset 1336, and repeats each accepted emission into a
distinct canary-filled buffer. Once the runtime witness passes, these checks
will establish deterministic parity only for these two fixtures. A broader
empty-extension subset matrix remains future qualification work.

## Non-Claims

- no default compiler integration;
- no replacement of the legacy serializer;
- no new SOIR version or opcode tag;
- no support for non-empty epistemic, algebra, claim, or BSS payloads;
- no performance claim until a source-fresh/Foundry benchmark exists;
- no proof of general failure atomicity; only the declared v0 input rejections
  promise an unchanged buffer;
- no parity claim beyond the two declared fixtures until a subset matrix runs.

## Promotion Rule

The writer may become authoritative only after:

1. byte-for-byte differential parity covers every supported v5 section;
2. rejection canaries cover every status category;
3. deterministic repeat emission is proven;
4. source-fresh and remote gates are green;
5. the legacy serializer remains available as a differential oracle during a
   bounded migration window.

Until then, `ir::serialize` is intentionally kept unchanged.

## Checkpoint Receipt

Status: `PARTIAL`

```text
Writer checker: PASS
Static architecture gate: PASS
Legacy tag-table parity: PASS
Witness checker: BASELINE-EQUIVALENT
Legacy serializer diagnostics: 22 E175, 7 E177
Witness diagnostics: 22 E175, 7 E177
Runtime differential: NOT RUN
Heavy/source-fresh build: NOT RUN
Default path: unchanged
Legacy oracle: kept
```

The standalone modular checker accepts `soir_writer.sio`. Importing the legacy
serializer into the differential witness reproduces the legacy serializer's
existing privacy diagnostics exactly; the witness adds no diagnostic category
or count. The gate therefore classifies runtime parity as not run rather than
converting checker-baseline equivalence into a byte-parity claim.

## Open Blocker

```text
Blocker-ID: BLK-20260714-soir-writer-v0-differential-runtime
Status: classified
Severity: B1
Class: harness-routing
Owner: /root integration shepherd
Lane: SOIR Writer v0 differential qualification
Worktree: /tmp/sounio-soir-writer-v0-20260714
Branch: codex/soir-writer-v0-20260714
Files-Owned: self-hosted/ir/soir_writer.sio, self-hosted/test_soir_writer_v0.sio, scripts/ci/soir_writer_v0_gate.sh, docs/internal/implementation/SOIR_WRITER_V0.md
Files-Read-Only: serialize.sio, soir_core.sio, heap_storage.sio, bootstrap_concat.sh
Do-Not-Touch: default serializer path and SOIR opcode tags
Repro: SOIR_WRITER_REQUIRE_DYNAMIC=1 bash scripts/ci/soir_writer_v0_gate.sh
Observed: writer check passes; witness matches 22 E175 + 7 E177 legacy diagnostics; runtime not run; gate rc=1
Expected: witness checker passes and prints PASS soir_writer_v0_differential
Acceptance-Gate: SOIR_WRITER_REQUIRE_DYNAMIC=1 bash scripts/ci/soir_writer_v0_gate.sh
Evidence-Level: E3
Evidence: gate output plus /tmp/soir-writer-v0-require-dynamic.log
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: run the differential through an isolated source-fresh integration surface without changing the default serializer in this lane
```

## Semantic Outcome

```text
Semantic-Outcome: implementation checkpoint; runtime qualification blocked
Concept-Status-Before: proposed
Concept-Status-After: proposed, checker-valid, not authoritative
Distinctions-Added: accepted plan vs emitted bytes; input rejection vs internal invariant breach
Distinctions-Preserved: wire value vs writer mechanism; compile success vs runtime parity
Distinctions-Erased: none
Evidence-Run: writer checker, static gate, tag parity, baseline diagnostic comparison
Fallback-Path: none
Legacy-Kept: yes, as default and differential oracle
Conflicting-Lanes: none in the declared write-set
Next-Semantic-Interface: IrModuleArena v2 read-only module view
```
