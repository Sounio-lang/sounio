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
same spelling            != same binding
binding resolution       != visibility authorization
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

`SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1` selects the compact table only as an
explicit differential oracle. The experimental route fails closed; it cannot
replace or fall back into the canonical AST-closure full-IR path.

Within one collected closure, bounded lexical lookup for functions, struct
constructors, enum paths, and enum variants first selects the definition owned
by the current `ModuleId`. If no local definition exists, the checker accepts a
global fallback only when the relevant namespace has one candidate. Concrete
lexical bindings shadow variants; pass-1 `TyUnknown` stubs do not. Visibility
authorization runs after lookup. Bare-variant ambiguity fails closed with E137,
and a valid local enum struct variant takes precedence over a remote
same-spelled struct.

This rule is not aggregate value identity. `TypeEntry` has no `ModuleId`, so
lookups driven by `TypeEntry.name` remain outside this authority, including
field access, patterns, return/parameter compatibility, layout, and linearity.
It is also not a canonical import-binding graph: the definition tables do not
record which authored `use` edge introduced a unique global fallback.

Imported named-function references expose a separate opt-in checker receipt.
The HOF gate requires the `TyUnknown` import stub to resolve to `TyFn` through
both the in-place and remaining by-value expression paths, and requires both
paths to reject a private imported function reference with E175. Its executable
witness then proves that a named imported function value survives the canonical
full-IR path. These are deliberately independent claims: lowering still
performs its own name lookup, so the gate does not claim direct `TypeEntry` to
IR identity continuity, capturing closures, or a function-value ABI contract.

With `SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1`, the full-IR path emits one
`module_frontend_full_ir: lower_node` record per `ModuleId` and one
`module_frontend_full_ir: lower_edge` record per authored edge. Legacy compact
trace records are not evidence that the canonical lowering consumed a closure.

## Semantic Lane

```text
Semantic-Lane-ID: modulegraph-facade-vertical-r1
Owner: Codex-2 compiler lane
Concept-IDs: SOUNIO-MODULE-CLOSURE-AUTHORITY
Intent-Preserved: authored module order and closure-local definition identity are retained through checking and presented to executable multimodule lowering
Transformation: repeated textual import discovery becomes one parsed closure whose module indices carry definition identity during one compile; bounded lexical checker lookup is local-ModuleId-first and global-unique-only
Types-Changed: ModuleClosure carrier fields and IrFunction.defining_module_id in memory
Effects-Changed: none
IR-Changed: IrFunction.defining_module_id is carried in memory for multimodule lowering and merge; its identity consumption is not yet attested and SOIR v4 does not serialize it
Claims-Introduced: the exact vertical gate proves closure-local identity is consumed by checker lookup and that the same ordered closure is presented to lowering before executable behavior; check-only #854 witnesses cover bounded lexical struct/enum/variant constructor selection through both in-place and remaining by-value checker paths; the HOF gate independently proves imported-stub refinement, function-reference privacy, and named function-value execution
Claims-Forbidden: direct checker TypeEntry-to-IR identity continuity, capturing closures, function-value ABI generality, TypeEntry-derived aggregate identity, cross-module transport or inspection of same-spelled aggregates, tuple-payload enum typing, canonical import binding, SOIR round-trip preservation, compiler-wide ModuleId preservation, general visibility correctness, the complete ModuleGraph epic, lean_single reexports, large-graph capacity, or #991 receipt semantics
Assumptions: authored imports resolve within the declared module-root set or established local/package paths
Write-Set: self-hosted/compiler/module_frontend.sio, self-hosted/compiler/module_native_driver.sio, self-hosted/compiler/main.sio, self-hosted/compiler/module_parse.sio, self-hosted/parser/items.sio, self-hosted/check/check.sio, self-hosted/check/defs.sio, self-hosted/check/mod.sio, self-hosted/ir/ir.sio, self-hosted/ir/lower.sio, self-hosted/ir/serialize.sio, self-hosted/ir/optimize.sio, self-hosted/ir/ssa.sio
Read-Set: resolver, native backend, legacy compact importer
Positive-Witness: scripts/ci/module_graph_facade_vertical_gate.sh reports context_state=resolved, runtime_state=pass, aggregate_witness_mode=check-only with exact aggregate surface passes, and the facade ELF prints 42; scripts/ci/madaros_full_path_imported_hof_gate.sh reports checker_refinement=TyUnknown-to-TyFn paths=inplace+by-value and runtime_exit=42
Negative-Witness: the vertical rejects missing, non-reexported, private function/struct/unit-enum/structured-enum, unresolved, ambiguous-global, ambiguous-bare-variant, or context-partial states; the HOF gate rejects imported private function references with E175x2
Acceptance-Gate: scripts/ci/module_graph_facade_vertical_gate.sh plus scripts/ci/madaros_full_path_imported_hof_gate.sh
Integration-Target: origin/main
Authoritative-Only-If: one source-fresh compiler passes both gates with context_state=resolved, runtime_state=pass, aggregate_witness_mode=check-only, checker_refinement=TyUnknown-to-TyFn, paths=inplace+by-value, private_fn_ref=E175x2, runtime_exit=42, and no fallback marker
```

## Issue #901 Stable Item Ownership Lane

```text
Semantic-Lane-ID: modulegraph-issue901-stable-items
Owner: Codex root compiler lane
Concept-IDs: SOUNIO-MODULE-CLOSURE-AUTHORITY
Intent-Preserved: every authored item in an imported module remains visible to closure discovery, explicit import authority, checker preparation, and canonical full-IR lowering
Transformation: ModuleGraph publishes its ItemList handle from the parsed pinned Program before the legacy Program-array compatibility copy; later closure traversal reads the handle and does not replace it from the aggregate carrier
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the #901 item-chain witness proves that two exports after the first item and a nested import after that first export reach a native ELF through the full noncompact path
Claims-Forbidden: persistent ModuleId, canonical import-binding serialization, complete ModuleGraph epic closure, generic aggregate transport, global fallback authorization, general large-graph capacity, or scientific validation of the imported library
Assumptions: parsed Box-backed Program and ItemList storage remain valid for the duration of one compiler invocation
Write-Set: self-hosted/compiler/module_frontend.sio, self-hosted/compiler/module_parse.sio, scripts/ci/madaros_imported_runtime_acceptance_gate.sh, tests/compiler/madaros_imported_runtime_acceptance/*
Read-Set: checker module array compatibility path, lowerer ItemList-handle consumption, native full-IR driver
Positive-Witness: madaros_imported_runtime_acceptance_gate.sh reports issue_901_item_chain=elf+exit0+42_LF and issue_901=elf+exit0+PROB_STDLIB_OK_LF with compact=disabled and fallback=none
Negative-Witness: existing #862 private-import witness remains E175 with no ELF; missing or ambiguous imports remain fail-closed
Acceptance-Gate: scripts/ci/madaros_imported_runtime_acceptance_gate.sh with one explicit source-fresh compiler SHA
Integration-Target: origin/main
Authoritative-Only-If: the source-fresh gate passes #921, #901 item-chain, real #901 prob stdlib, and #862 privacy without a compact or fallback marker
```

## Current Boundary

The `ModuleId` stored in `IrFunction.defining_module_id` is an index in the active
compile's closure. It is not a persistent module identifier and is not authority
outside that closure. Deserializing SOIR v4 explicitly restores
`IR_DEFINING_MODULE_UNKNOWN`, because the v4 wire format has no provenance field.
This lane therefore makes no SOIR round-trip or compiler-wide preservation claim.

This contract establishes the carrier, bounded contextual checker lookup,
in-memory identity presented to lowering, and refusal boundary for the exact gate. It does
not close the remaining ModuleGraph epic work: a ModuleId-bearing aggregate type
carrier, canonical import bindings, normalized physical identity, graph digest,
large-closure capacity, SOIR installation, compiler-wide identity propagation,
scientific metadata, numeric payload, or complete receipts remain separate
executable milestones.
