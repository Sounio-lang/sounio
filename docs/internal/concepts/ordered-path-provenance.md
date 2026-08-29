<!-- docs:meta
topic_id: repo.docs.internal.concepts.ordered-path-provenance
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ordered-path-provenance
-->

# Ordered Path Provenance

Concept-ID: `SOUNIO-ORDERED-PATH-PROVENANCE`

Status: executable for the bounded fixture described here. Promotion was based
on the strict current-source receipt from source commit `5db068512f55fa832ea190e71a2498dc5cbc1f6f`
and Madaros SHA-256 `32ea8cc250b70f0ac632fe8084f537e3f6179d85200f704d1d4da8367b315997`:
two source checks, four nominal category rejections, exact single-module and
imported runtime outputs, two exact 26-link traces, and `fallback=0` produced
`merge_ready=1`. Executable status is limited to that synthetic, bounded
source-to-IR witness; it is not a compiler-wide preservation theorem.

## Founder Intent

A scalar observation does not exhaust functional state. If a model declares
that `A` followed by `B` and `B` followed by `A` produce distinct states, Sounio
must preserve the ordered path instead of treating the states as interchangeable.
If the ordered leaves are the same but their grouping differs, Sounio must also
preserve the grouping witness.

This concept refines `SOUNIO-NONASSOCIATIVE-ORDER`. It does not authorize any
algebraic rewrite and does not infer a causal or biological mechanism from a
function body.

## Two Distinct Questions

Order sensitivity and nonassociativity are not synonyms:

```text
A then B != B then A                 order sensitivity / noncommutation
(A then B) then C != A then (B then C)  parenthesization sensitivity
```

The first witness changes the ordered leaves. The second keeps the exact leaf
sequence `A, B, C` and changes only an explicit `LeftGrouping` versus
`RightGrouping` receipt. A compile-fail control prevents an
`OrderSensitivityWitness` from substituting for a `NonAssociativityWitness`.

## Executable Representation

The focused source witnesses use ontology classes as nominal types for steps,
whole-path order receipts, observations, grouping receipts, and functional
states. `OrderABReceipt` and `OrderBAReceipt` accompany the exact positional
sequences `A,B` and `B,A`; neither representation substitutes for the other.
Both orderings and both groupings deliberately collide at the immediate scalar
projection. Their state types remain distinct, and the same fixed synthetic
later context `7` is encoded in state-specific continuations that diverge.

The compiler already exports every checked ontology-typed parameter as the
audit-only triple:

```text
(lowered function name, source parameter index, ontology class name)
```

`SOUNIO_ORDERED_PATH_TRACE=1` prints those exact triples after cleanup and, for
imports, after modular finalization. No diagnostic hash is needed to establish
the bounded fixture identity. Every trace line declares `authority=0`: the
metadata can be inspected but cannot authorize optimization or code generation.
The summary also requires `bounded=1`; an out-of-range table count emits no link
claims. The existing `IrOntologyTable` has a fixed 128-link capacity. This lane's
26-link fixture is deliberately below that boundary and does not claim that the
table records arbitrary larger programs without refusal or loss.

In this document, a path identity "survives" only when the strict gate observes
the expected checked triples re-emitted from the audit table at those two named
compiler phases. It is a bounded executable observation, not a soundness theorem
or a compiler-wide preservation result.

The evidence has two layers. The four compile-fail programs exercise the
pre-existing ontology typechecker and show that the concept can be encoded with
nominal category boundaries. The new compiler behavior is the post-cleanup and
post-merge trace; the strict gate would fail on a compiler that typechecks the
same sources but lacks that trace or emits a changed triple. Runtime arithmetic
establishes only the exact scalar collision and later divergence. It is not used
as evidence that the provenance metadata survived; the trace matrix supplies
that independent observation.

The optimized runtime witness stays inside the compiler's admitted exact-bitwise
slice. It returns a branchless certificate formed by OR-ing each observed versus
expected XOR delta; the certificate is zero exactly when all eight scalar checks
match. The generic current-source `-O` failures that motivated this bounded form
are tracked separately in [issue #1070](https://github.com/Sounio-lang/sounio/issues/1070).
That issue covers ontology-free arithmetic crashes and scalar call/guard
miscompilations. This lane uses no silent fallback and makes no claim that those
broader compiler paths are correct.

Ontology return types are enforced by the source checker. In this first slice,
the resulting state identity re-enters the IR as the typed parameter of its
projector and continuation. Direct ontology-result metadata is a separate future
interface rather than an implicit claim of this lane.

## Falsification Conditions

The focused claim fails if any of the following occurs:

- `StateAfterAB` is accepted where `StateAfterBA` is required.
- `LeftGroupedABCState` is accepted where `RightGroupedABCState` is required.
- an occupancy observation is accepted as a functional state.
- order sensitivity is accepted as a nonassociativity witness.
- the immediate scalar projections do not collide exactly.
- the shared fixed later context does not distinguish the typed continuations.
- the `A,B` and `B,A` parameter triples or their distinct order receipts become
  equal or disappear after cleanup.
- the identical `A,B,C` leaf sequence or the distinct grouping receipt disappears
  after imported-module finalization.
- the trace is consumed as transformation authority.

## Claims Forbidden

- A clinical, diagnostic, pharmacological, or biological prediction.
- A claim that receptor occupancy determines functional state.
- Automatic inference of pathway causality from arbitrary source code.
- A general theorem that all stateful composition is noncommutative or
  nonassociative.
- Compiler-wide source-to-native semantic preservation.
- Cryptographic uniqueness or collision resistance of diagnostic output.
- Optimizer authority from ontology identity, a typed state, or a public trace.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: ordered-path-provenance-source-ir-v1
Owner: Codex ordered-path provenance lane
Concept-IDs: SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: order and parenthesization remain semantic state when the source model declares them
Transformation: none; print checked audit-only ontology parameter triples after native-v2 cleanup and modular finalization
Types-Changed: source witnesses add nominal path-step, whole-path order-receipt, grouping-receipt, observation, and functional-state classes; compiler IR layout is unchanged
Effects-Changed: none
IR-Changed: no opcode or executable field; bounded read-only accessors expose existing IrOntologyParameterLink entries to an opt-in trace
Claims-Introduced: the strict focused gate re-observes A-B/B-A and left/right-grouped A-B-C audit identities after optimized cleanup and imported merge while exact immediate scalar projections coincide
Claims-Forbidden: clinical meaning, inferred causality, ontology-derived optimizer authority, general noncommutativity/nonassociativity, compiler-wide preservation
Assumptions: ontology parameter links are collected only after authoritative typecheck; parameter indices retain source signature order; merge deduplicates only exact function/index/class triples; typed projectors and continuations make result-state identity visible as a later signature input
Write-Set: self-hosted/ir/ir.sio; self-hosted/compiler/module_frontend.sio; tests/compiler/ordered_path_provenance_*; tests/compile-fail/ordered_path_*; scripts/ci/ordered_path_provenance_source_ir_gate.sh; docs/internal/concepts/ordered-path-provenance.md; docs/internal/concepts/registry.tsv; docs/internal/concepts/bindings.tsv
Read-Set: self-hosted/check/mod.sio; self-hosted/ir/opt_cleanup.sio; scripts/lib/resolve_souc.sh; bin/souc
Positive-Witness: exact bitwise scalar collision at 877; same fixed later context 7 yields AB=874, BA=879, left=5, right=866; exact source parameter triples and distinct OrderABReceipt/OrderBAReceipt classes survive single and imported optimized paths
Negative-Witness: four compile-fail category boundaries for AB/BA state, left/right grouping state, order/nonassociativity witness, and observation/functional state
Acceptance-Gate: strict ordered_path_provenance_source_ir_gate.sh with an expected source Git SHA and a current-source Foundry compiler at the expected SHA-256
Integration-Target: default native-v2 single-module and imported/merged optimized source paths
Authoritative-Only-If: the source and compiler SHAs match explicit expected values, both ELF witnesses return zero with exact output, all four category rejections hold, and no fallback occurs
```
