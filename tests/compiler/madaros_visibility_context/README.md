# Madaros module-binding identity reducer

This fixture family began as a reducer for issue #854. It separates a false
privacy diagnostic caused by name-only lookup from genuine cross-module private
access, then gates the bounded contextual-checker implementation described
below.

The one-call reducer mirrors the E5 test. The root and imported leaf each own a
private `abs_f64`; in the pinned baseline the leaf's one internal call resolves
to the root's earlier table entry and emits one false E175.

The matrix mirrors the public example exactly at the causal level:

```text
eisa::format::abs_f64       1 internal call
eisa::evm::print_i64_dec   17 internal calls
                            -----------------
                            18 false E175 diagnostics
```

The fixture uses the same two duplicate private names and the same `1 + 17`
call topology. The baseline is classified with `madaros check`, before
lowering or runtime can replace the diagnostic exit status. Only after both
reducers check clean does the gate execute them. Different return values then
make a superficial "allow access" change insufficient: the resolved state
must print the exact PASS markers from the module-owned bindings.

Run the classifier against a current-source Madaros:

```bash
MADAROS_RAW_BIN=/path/to/madaros \
  bash scripts/ci/madaros_visibility_context_gate.sh
```

Require the implementing acceptance state with:

```bash
MADAROS_RAW_BIN=/path/to/madaros \
SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved \
  bash scripts/ci/madaros_visibility_context_gate.sh
```

`MADAROS_RAW_BIN` is mandatory so the gate cannot silently use the checked-in
or otherwise stale compiler. The gate accepts only two coherent
classifications: the exact `1/18` E175 check baseline, or two clean checks
followed by exact global-function and bare-variant E137 rejections, check-only
aggregate-constructor witnesses, and both executable function witnesses. In
either state it requires the canonical E175, E176, and E177 true-private
negative controls.
Mixed states, fatal logs, diagnostic-count drift, successful checks that
retain privacy diagnostics, and runtime exits without exact markers are
rejected.

The ModuleGraph facade vertical invokes this gate with `EXPECT=resolved` once a
current-source raw compiler is available. The checked-in prebuilt remains a
valid historical-baseline classifier; it is not evidence for the implementing
state. The enclosing vertical accepts only the exact aggregate
`aggregate_witness_mode=check-only` receipt, both ambiguity controls, and the
runtime function receipt. It republishes those facts with `visibility_854_*`
keys in its own receipt.

## Contextual checker port boundary

The 2026-07-19 port onto `b75376c9e` restores the checker half of the proven
#867/#869 stack after closure-local function identity made the runtime witnesses
pass. The lookup rule is deliberately bounded:

1. Ordinary function names, struct constructors, enum paths, and enum variant
   constructors first resolve inside the current `ModuleId`.
2. If no local definition exists, a global fallback is accepted only when the
   relevant lexical namespace has exactly one candidate in the closure.
3. Concrete lexical bindings shadow enum variants. Pass-1 `TyUnknown` stubs do
   not: they allow contextual function and variant lookup to recover identity.
4. A bare variant must be unique inside the selected namespace; ambiguity emits
   E137 instead of degrading to a shared `TyUnknown` binding.
5. For `E::V { ... }`, a valid local enum variant is selected before a remote
   same-spelled struct can emit a false E176.
6. Definition visibility is then checked with the real current `ModuleId`.

This is sufficient for the bounded lexical namespace above: it distinguishes
same-spelled private locals, rejects genuine cross-module private access, and
refuses collection-order choice where the gate has an ambiguity witness. It is
**not** aggregate value identity. `TypeEntry` carries a name but no `ModuleId`,
so field access, pattern checking, return/parameter compatibility, layout,
linearity, and other lookups derived from `TypeEntry.name` remain P2 and
Claims-Forbidden. The aggregate fixtures only construct their module-local type
and return `i64`; no same-spelled aggregate crosses or is inspected across a
module boundary.

This is also not a canonical import-binding graph: the selected `use` edge is
not yet carried into the definition tables, and a unique global fallback does
not prove which import introduced the binding. Enum tuple-payload declarations
remain parser work; this wave proves only the existing zero-argument variant
call form, not tuple-payload typing. The facade surface validator remains
facade-only and cannot replace this checker pass for mixed implementation
modules.

```text
Semantic-Lane-ID: issue-854-contextual-checker-port-r2
Owner: Codex-2 compiler lane
Concept-IDs: SOUNIO-MODULE-CLOSURE-AUTHORITY
Status: implementation-candidate; source-fresh acceptance pending
Intent-Preserved: binding resolution precedes visibility authorization
Transformation: name-only lexical constructor lookup -> local ModuleId first, global unique-only fallback
Types-Changed: none
Effects-Changed: none
IR-Changed: none in this port; closure-local IR identity is an input prerequisite
Claims-Introduced: bounded lexical function/struct/enum/variant constructor lookup distinguishes module-local identities; bare variant ambiguity fails closed with E137
Claims-Forbidden: TypeEntry-derived aggregate identity; cross-module transport or inspection of same-spelled aggregates; tuple-payload enum typing; canonical import binding; general qualified-name resolution; general re-export correctness; issue #854 closed before the source-fresh acceptance gate passes
Assumptions: the modular checker stamps each collected definition and current checker pass with the closure-local ModuleId
Write-Set: self-hosted/check/check.sio; self-hosted/check/defs.sio; self-hosted/compiler/main.sio; scripts/ci/madaros_visibility_context_gate.sh; scripts/ci/module_graph_facade_vertical_gate.sh; tests/compiler/madaros_visibility_context/*
Read-Set: self-hosted/check/mod.sio; self-hosted/compiler/module_frontend.sio; self-hosted/ir/ir.sio; self-hosted/ir/lower.sio; native codegen
Positive-Witness: duplicate_private_single_main.sio and duplicate_private_18_main.sio execute exact PASS markers; duplicate_private_struct_main.sio and duplicate_private_enum_main.sio pass check-only local-constructor witnesses, including ExprLoop by-value bridges in root and leaf
Negative-Witness: ambiguous_public_main.sio=E137; ambiguous_bare_variant_main.sio=E137; private function=E175; private struct=E176; private unit enum=E177; private structured enum=E177
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved bash scripts/ci/madaros_visibility_context_gate.sh
Integration-Target: origin/main
Authoritative-Only-If: the source-fresh acceptance gate reports context_state=resolved, runtime_state=pass, both ambiguity controls, aggregate_witness_mode=check-only with every aggregate surface pass, and exact E175/E176/E177 controls including the structured-variant E177 without compiler fallback
Fallback-Path: unique-only global lookup; rejected when more than one candidate exists
Legacy-Kept: name-only lookup remains for consumers outside the contextual checker surface
LLM-Offload: not-required (compiler binding mechanics; no math, clinical pathway, or external-facing claim)
```

```text
Semantic-Outcome: implementation candidate; source-fresh semantic evidence pending
Concept-Status-Before: SOUNIO-MODULE-CLOSURE-AUTHORITY executable-candidate with checker read-only
Concept-Status-After: SOUNIO-MODULE-CLOSURE-AUTHORITY executable-candidate with bounded contextual checker ownership
Distinctions-Added: same spelling != same binding; lexical constructor identity != carried aggregate value identity; binding resolution != visibility authorization
Distinctions-Preserved: authored import edge != global name visibility; compile success != runtime parity
Distinctions-Erased: none
Evidence-Run: semantic scanner, diff checks, shell syntax gates, and historical-prebuilt baseline classifier; source-fresh acceptance pending
Fallback-Path: compiler fallback none; checker lookup fallback global-unique-only and fail-closed on duplicates
Legacy-Kept: name-only helper and SOIR v4 path remain outside the proven contextual surface
Conflicting-Lanes: scanner reported observational historical overlaps; no active ownership conflict established
Next-Semantic-Interface: ModuleId-bearing TypeEntry replacement plus canonical import-binding records tied to authored closure edges
```

## Exact baseline receipt

The initial classifier run used the `madaros-current-source-f64-lowering`
artifact produced for exact `origin/main` SHA
`e6725240d266353621639419f1c4fd859d90caad` by CI run `29270956296`
(artifact ID `8287540274`). The compiler ELF was 98,413,103 bytes with SHA-256
`7e6203c08b254ea46cbae4094a0109317ac28063e4bb3b84209d51eeef2ae8fa`.

```text
context_state=baseline
runtime_state=not-run-baseline
single_e175=1
matrix_e175=18
true_private_fn=E175
true_private_struct=E176
true_private_enum=E177
```

The same artifact with `SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved`
exits 1 and emits the blocker ID without attempting reducer runtime. A
synthetic resolved-state harness proves the gate orders clean checks before
the two exact runtime markers; it is a gate self-test, not evidence that the
compiler is fixed. A separate synthetic compiler that accepts every input is
rejected when the E175 negative control returns 0. This pins the state machine
and anti-weakening behavior independently of the current compiler result.

## Historical blocker handoff

```text
Blocker-ID: BLK-20260713-MADAROS-EISA-E5-VISIBILITY-E175
Status: classified
Severity: B1
Class: compiler-semantics
Owner: Codex root coordinator (implementation dispatch pending)
Lane: Madaros module-binding identity / issue #854
Worktree: /tmp/sounio-issue-854-reducer-20260713
Branch: codex/issue-854-reducer-20260713
Files-Owned: tests/compiler/madaros_visibility_context/*; scripts/ci/madaros_visibility_context_gate.sh
Files-Read-Only: self-hosted/check/check.sio; self-hosted/check/defs.sio; self-hosted/check/mod.sio; self-hosted/ir/lower.sio; EISA witnesses and stdlib
Do-Not-Touch: self-hosted/check/check.sio while PR #814 owns it; canonical compiler artifacts; EISA mathematical thresholds and receipt semantics
Repro: MADAROS_RAW_BIN=<current-source-madaros> bash scripts/ci/madaros_visibility_context_gate.sh
Observed: name-only FnSigTable lookup selects the first same-name declaration even while checking a different defining module
Expected: an unqualified call resolves through the caller module/import context before visibility is evaluated
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved bash scripts/ci/madaros_visibility_context_gate.sh
Evidence-Level: E3
Evidence: exact-main CI run 29270956296 artifact 8287540274 plus the baseline receipt in this README
Fallback-Path: none
Legacy-Kept: yes; existing checker, EISA, and negative visibility paths are unchanged
LLM-Offload: not-required (diagnostic fixtures/gate only; no math, clinical, or external claim change)
Next-Action: implement module-aware function binding lookup after PR #814 releases checker ownership
```

## Reducer semantic lane declaration (historical)

`SOUNIO-MODULE-BINDING-IDENTITY` is proposed here for the future implementation
lane. It is not yet registered as an executable concept, and this reducer does
not alter its semantics.

```text
Semantic-Lane-ID: issue-854-module-binding-identity-reducer
Owner: Codex issue #854 reducer lane
Concept-IDs: proposed SOUNIO-MODULE-BINDING-IDENTITY
Intent-Preserved: source-level module ownership remains part of symbol identity; privacy is checked after contextual binding resolution
Transformation: none; executable characterization only
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: current-source Madaros reproduces exact 1-call and 18-call E175 name-collision boundaries while true private access remains rejected
Claims-Forbidden: EISA runtime parity; general import/re-export correctness; implementation of module-aware bindings; weakening private visibility
Assumptions: root module is collected before imported leaves; unqualified same-module declarations are in lexical scope
Write-Set: tests/compiler/madaros_visibility_context/*; scripts/ci/madaros_visibility_context_gate.sh
Read-Set: self-hosted/check/check.sio; self-hosted/check/defs.sio; self-hosted/check/mod.sio; self-hosted/compiler/main.sio; EISA fixtures and modules
Positive-Witness: acceptance-only; after clean checks, duplicate_private_single_main.sio and duplicate_private_18_main.sio must execute their exact PASS markers
Negative-Witness: visibility_fn_private_main.sio=E175; visibility_struct_private_main.sio=E176; visibility_enum_private_main.sio=E177
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved bash scripts/ci/madaros_visibility_context_gate.sh
Integration-Target: future checker/module-binding implementation lane after PR #814
Authoritative-Only-If: both contextual witnesses execute and all three true-private diagnostics remain exact
```

## Integration receipt

```text
Semantic-Outcome: exact blocker reduced; implementation intentionally deferred
Concept-Status-Before: unregistered proposal
Concept-Status-After: unregistered proposal with executable characterization
Distinctions-Added: same spelling != same binding; binding resolution != visibility authorization
Distinctions-Preserved: private same-module access != private cross-module access; compile success != runtime binding parity
Distinctions-Erased: none
Evidence-Run: baseline classifier against exact-main CI artifact; resolved-state and accept-all harness self-tests
Fallback-Path: none
Legacy-Kept: all existing checker, EISA, and visibility paths
Conflicting-Lanes: PR #814 owns self-hosted/check/check.sio; no overlap in this phase
Ordinary-CI: not wired in this diagnostic lane; required wiring belongs to the implementation PR
Next-Semantic-Interface: module-qualified binding records and caller-context lookup
```
