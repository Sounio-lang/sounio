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

`SOUNIO_LEGACY_COMPACT_IR=1` selects the old compact table only as an explicit
differential oracle. It cannot silently replace, or fall back into, the
canonical AST-closure full-IR path.

## Semantic Lane

```text
Semantic-Lane-ID: modulegraph-facade-vertical-r1
Owner: Codex-2 compiler lane
Concept-IDs: SOUNIO-MODULE-CLOSURE-AUTHORITY
Intent-Preserved: authored module order, visibility, and identity survive into executable lowering
Transformation: repeated textual import discovery becomes one parsed closure with stable ModuleIds
Types-Changed: ModuleClosure carrier fields only
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a passing vertical gate proves one facade closure checks, lowers, and executes without fallback
Claims-Forbidden: this alone fixes lean_single reexports, all visibility semantics, large-graph capacity, or #991 receipt semantics
Assumptions: authored imports resolve within the declared module-root set or established local/package paths
Write-Set: self-hosted/compiler/module_frontend.sio, self-hosted/compiler/module_native_driver.sio, self-hosted/compiler/main.sio, self-hosted/compiler/module_parse.sio, self-hosted/parser/items.sio
Read-Set: checker, resolver, native backend, legacy compact importer
Positive-Witness: root -> public facade -> leaf -> ELF prints 42
Negative-Witness: missing, non-reexported, private, unresolved, and ambiguous imports produce no ELF
Acceptance-Gate: scripts/ci/module_graph_facade_vertical_gate.sh
Integration-Target: origin/main
Authoritative-Only-If: source-fresh compiler passes positive and negative witnesses with no fallback marker
```

## Current Boundary

This contract establishes the carrier and refusal boundary. It does not close
the remaining ModuleGraph epic work: normalized physical identity, graph digest,
large-closure capacity, SOIR installation, scientific metadata, numeric payload,
or complete receipts remain separate executable milestones.
