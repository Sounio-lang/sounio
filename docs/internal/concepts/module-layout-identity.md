<!-- docs:meta
topic_id: repo.docs.internal.concepts.module-layout-identity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.module-layout-identity
-->

# Module Layout Identity

Concept-ID: `SOUNIO-MODULE-LAYOUT-IDENTITY`

Status: hypothesis

An imported aggregate field is addressed by the declared layout of its direct
base expression. For a chained expression such as `outer.inner.field`, the
second lookup uses the declared nominal result type of `inner`; it is not a
global search for another field named `field`.

## Preserved Distinctions

```text
field spelling           != field identity
outer layout index       != inner layout index
known layout miss        != dynamic layout fallback
compile success          != imported runtime parity
aggregate payload        != scientific interpretation
```

A known nominal layout that lacks a requested field fails lowering. The legacy
name-only lookup remains confined to genuinely untyped/dynamic shapes; it may
not repair a typed layout miss by borrowing a same-spelled field from another
struct.

## Semantic Lane

```text
Semantic-Lane-ID: issue901-current-main-layout-and-chain-20260721
Owner: codex-root-issue901-20260721
Concept-IDs: SOUNIO-MODULE-LAYOUT-IDENTITY
Intent-Preserved: imported programs retain the authored nominal field layout through summary transport and native lowering
Transformation: fixed by-value layout storage becomes a complete paged catalog; field entries retain declared nominal result type for typed chained access
Types-Changed: StructFieldEntry gains value_type_name; StructLayoutTable gains paged catalog state
Effects-Changed: layout allocation is explicit through Alloc
IR-Changed: lowering-only layout metadata retains nominal field result identity; no SOIR serialization or scientific payload claim
Claims-Introduced: an imported two-module witness can distinguish same-spelled fields at different outer and inner offsets and execute the inner value
Claims-Forbidden: general aggregate ABI parity, arbitrary dynamic reflection, layout-capacity closure, D11/D12 runtime parity without their exact source lineage, physical or clinical validity, SOIR round-trip preservation, and closure of #901 without the separately scoped integration and Foundry evidence
Assumptions: registered field declarations retain nominal TypeNamed paths during one source-fresh compiler invocation
Write-Set: self-hosted/ir/ir.sio, self-hosted/ir/lower.sio, self-hosted/compiler/main.sio, tests/compiler/madaros_imported_runtime_acceptance/*
Read-Set: parser FieldDefList types, modular import closure, native-v2 field-get emission
Positive-Witness: issue_901_nested_field_chain_main.sio prints 520 and ISSUE_901_NESTED_FIELD_CHAIN_OK through the current-source imported path
Negative-Witness: a typed known-layout field miss lowers as an explicit error and never selects a same-named field from a different layout
Acceptance-Gate: source-fresh Madaros runs the forward-declared nested-field witness and the exact available D6 imported witness with no fallback marker
Integration-Target: origin/main
Authoritative-Only-If: current-source compiler, exact source inputs, executable ELFs, and no fallback marker prove each listed witness
```

## Boundary

This contract is about compiler provenance of aggregate layout identity. It does
not promote a library receipt to physical causality, alter values or uncertainty
semantics, or infer clinical validity from a successful native execution.
