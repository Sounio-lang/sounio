<!-- docs:meta
topic_id: repo.docs.internal.concepts.module-closure-authority
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.module-closure-authority
-->

# Module Closure Authority

Concept-ID: `SOUNIO-MODULE-CLOSURE-AUTHORITY`

Status: executable candidate

The compiler resolves an authored logical import to one physical source and one
`ModuleId` exactly once. The resulting `ModuleClosure`, including its parsed
`Program` carrier and authored edges, is the authority consumed by modular
checking and full-IR lowering.

## Preserved Distinctions

```text
logical module identity != physical source path
declared module identity != authored import spelling
authored import edge     != global name visibility
closed file closure      != valid export surface
compile success          != executable parity
legacy compact IR        != canonical lowering
```

Resolution fails closed when an import is unresolved or ambiguous, parsing
fails, or capacity saturates. Import-surface validation is a separate refusal:
the closure report remains structurally `complete`, records
`surface_status\tinvalid`, and compilation rejects the unavailable name with
`E137`. `pub use leaf::{x}` can forward `x`; it does not forward other public
names from `leaf`. A private `use` never expands the facade's public surface.
If topology is incomplete, surface validation has not run and the report says
`surface_status\tnot_evaluated`; absence of a surface error is never promoted to
an unevidenced `valid` claim.

`SOUNIO_LEGACY_COMPACT_IR=1` selects the old compact table only as an explicit
differential oracle. It cannot silently replace, or fall back into, the
canonical AST-closure full-IR path.

With `SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1`, the full-IR path emits one
`module_frontend_full_ir: lower_node` record per `ModuleId` and one
`module_frontend_full_ir: lower_edge` record per authored edge. Legacy compact
trace records are not evidence that the canonical lowering consumed a closure.

## Semantic Lane

```text
Semantic-Lane-ID: modulegraph-facade-vertical-r1
Owner: Codex-2 compiler lane
Concept-IDs: SOUNIO-MODULE-CLOSURE-AUTHORITY
Intent-Preserved: authored module order and closure-local definition identity survive into executable multimodule lowering
Transformation: repeated textual import discovery becomes one parsed closure whose module indices identify function definitions during one compile
Types-Changed: ModuleClosure carrier fields and IrFunction.defining_module_id in memory
Effects-Changed: none
IR-Changed: IrFunction.defining_module_id is authoritative in memory for multimodule lowering and merge; SOIR v4 does not serialize it
Claims-Introduced: the exact vertical gate proves closure-local function identity is consumed by lowering and merge when its #854 context witnesses resolve and execute
Claims-Forbidden: SOIR round-trip preservation, compiler-wide ModuleId preservation, general visibility correctness, the complete ModuleGraph epic, lean_single reexports, large-graph capacity, or #991 receipt semantics
Assumptions: authored imports resolve within the declared module-root set or established local/package paths
Write-Set: self-hosted/compiler/module_frontend.sio, self-hosted/compiler/module_native_driver.sio, self-hosted/compiler/main.sio, self-hosted/compiler/module_parse.sio, self-hosted/parser/items.sio, self-hosted/ir/ir.sio, self-hosted/ir/lower.sio, self-hosted/ir/serialize.sio, self-hosted/ir/optimize.sio, self-hosted/ir/ssa.sio
Read-Set: checker, resolver, native backend, legacy compact importer
Positive-Witness: scripts/ci/module_graph_facade_vertical_gate.sh reports context_state=resolved, runtime_state=pass, and the facade ELF prints 42
Negative-Witness: the same gate rejects missing, non-reexported, private, unresolved, ambiguous, or context-partial states without accepting fallback
Acceptance-Gate: scripts/ci/module_graph_facade_vertical_gate.sh
Integration-Target: origin/main
Authoritative-Only-If: source-fresh compiler passes the exact positive and negative witnesses with context_state=resolved, runtime_state=pass, and no fallback marker
```

## Current Boundary

The `ModuleId` stored in `IrFunction.defining_module_id` is an index in the active
compile's closure. It is not a persistent module identifier and is not authority
outside that closure. Deserializing SOIR v4 explicitly restores
`IR_DEFINING_MODULE_UNKNOWN`, because the v4 wire format has no provenance field.
This lane therefore makes no SOIR round-trip or compiler-wide preservation claim.

This contract establishes the carrier, in-memory lowering identity, and refusal
boundary for the exact gate. It does not close the remaining ModuleGraph epic
work: normalized physical identity, graph digest, large-closure capacity, SOIR
installation, compiler-wide identity propagation, scientific metadata, numeric
payload, or complete receipts remain separate executable milestones.
