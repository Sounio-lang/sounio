<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-rebracketing-protocol-d7-2026-07-15
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-rebracketing-protocol-d7-2026-07-15
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# D7: Proof-Carrying Rebracketing Protocol

Date: 2026-07-18
Evidence level: executable bounded synthetic model protocol
Concept-ID: `SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL`
Related compiler concept: `SOUNIO-REBRACKETING-AUTHORITY`

## Question

What evidence should a model retain when comparing `((a * b) * c)` with
`(a * (b * c))`, without pretending that a public model receipt is permission
for the compiler to rewrite source or IR?

D7 answers with a nominally separated protocol for local equality, semantic
refusal, model replay, cross-occurrence refusal, and compiler-promotion
abstention. It is deliberately not a compiler authority implementation.

## Frozen Program

The ordered D6 atom identities are:

| Atom | ID | Role |
|---|---:|---|
| `a` | 9101 | policy context |
| `b` | 9102 | commit boundary |
| `c` | 9103 | pending synthetic probe |

Both D6 parenthesizations are defined. D7 consumes those exact IDs from the D6
non-associativity receipt; it does not infer them from a checksum or replace
them with hardcoded equality booleans.

The flat control recomputes:

```text
flat_left  = (9101 * 101 + 9102) * 101 + 9103 = 93767706
flat_right =  9101 * 101^2 + (9102 * 101 + 9103) = 93767706
```

The grouping audit recurrence remains non-associative:

```text
tree_left  = (9101 * 31 + 9102) * 31 + 9103 = 9037326
tree_right =  9101 * 31 + (9102 * 31 + 9103) = 573396
```

The six frozen numeric label orders produce six distinct flat checksums in this
finite fixture. Because the labels exceed base `101`, this is not a general
injectivity, identity-recovery, or collision-resistance claim.

## Declared Cases

| Case | Fixture occurrence | Model operator | Result |
|---|---:|---:|---|
| flat equality | 11001 | 9801 | local equality decision 10201 and model replay 10301 |
| semantic inequality | 11002 | 9601 | semantic refusal 10401 |
| compiler promotion request | origin 11001 | 9801 | promotion abstention 10601 |
| wrong-occurrence control | requested 11003 | 9801 | replay refusal 10801 |

The first three are the declared decision cases. D7 records all three; it does
not claim to generate an exhaustive universe of candidate rewrites. The
independent oracle enumerates all `2^6 = 64` inputs to the declared Boolean
local-decision predicate and confirms that only the all-true vector is admitted.
That is predicate coverage, not a compiler completeness theorem.

## Exact Bounded Codes

The semantic carrier is encoded for audit as:

```text
code(status, survivor, burden, count, evidence) =
    ((((status * 31) + survivor) * 31 + burden) * 31 + count)
    * 1000000 + evidence
```

With `status <= 3`, `survivor <= 3`, `burden <= 7`,
`count <= 2`, and `evidence < 1000000`, the maximum code is
`92475999999`, below signed `i64`. Base-31 components and the final
base-`1000000` component are decoded by the oracle, so this bounded code is
injective on the asserted component domain.

The frozen semantic results are:

```text
left  = code(2, 3, 3, 1, 8101)   = 62559008101
right = code(3, 2, 7, 2, 259234) = 91514259234
```

The diagnostic difference bitset is `1 + 2 + 4 + 8 = 15` for status,
survivor mask, burden, and evidence-count differences. It is not a metric,
numeric associator, or independence claim.

## Refusal Receipts

Semantic refusal mask `31` records:

- `1`: bounded semantic results differ;
- `2`: committed evidence would change under the proposed model tree swap;
- `4`: grouping-retained payloads differ;
- `8`: D6 declares the partial operator non-associative on this witness;
- `16`: no universal associativity law was supplied.

Compiler-promotion abstention mask `63` records:

- `1`: the evidence is local to one model fixture;
- `2`: universal quantification is absent;
- `4`: compiler binding is absent;
- `8`: a sealed compiler capability is absent;
- `16`: native `Contest/TyContest/IrContest` evidence is absent;
- `32`: compiler operator-admission evidence is absent.

Wrong-occurrence replay mask `3` records the fixture mismatch and the refusal
of cross-occurrence reuse. No model replay or compiler authority is issued.

## Canonical Compiler Separation

The current `SOUNIO-REBRACKETING-AUTHORITY` is a compiler-private structural
capability. Its bounded executable slice revalidates live IR and admits exact
bitwise AND, OR, and XOR transactions. It does not return or serialize its
private capability, and diagnostic hashes cannot authorize mutation.

D7 leaves that row, contract, bindings, implementation, and gates unchanged.
Its public model decision is neither that private capability nor evidence that
operator `9801` or `9601` is admitted by the compiler. D7 also does not
instantiate native `Contest`, lower to `TyContest` or `IrContest`, or add
an epistemic index.

## Literature Compass

The protocol is adjacent to several stronger or differently scoped ideas:

- Necula's proof-carrying code lets a consumer check a proof that supplied code
  obeys a safety policy. D7 has public executable receipts but no small trusted
  proof checker. [Proof-Carrying Code](https://doi.org/10.1145/263699.263712)
- Translation validation checks a particular compiler run rather than proving
  the compiler correct for every program. D7 borrows the local/global
  distinction but does not formalize source and target semantics as a
  refinement relation. [Translation Validation](https://doi.org/10.1007/BFb0054170)
- Alive2 performs bounded translation validation for LLVM IR. D7 has no SMT
  semantics, poison/undefined-behavior model, or general optimization checker.
  [Alive2](https://doi.org/10.1145/3453483.3454030)
- The current LLVM LangRef says `reassoc` permits algebraically equivalent
  floating-point transformations such as reassociation and can significantly
  change results. D7 shares only the principle that permission must be
  explicit; it is not LLVM-compatible and has no floating-point authority.
  [LLVM Language Reference](https://llvm.org/docs/LangRef.html#fast-math-flags)
- CompCert represents the stronger rival of machine-checked semantic
  preservation. D7 establishes no verified compiler theorem.
  [A Formally Verified Compiler Back-end](https://doi.org/10.1007/s10817-009-9155-4)

These sources motivate boundaries; none entails the D7 construction or proves
its novelty.

## Exact Supported Claim

For one frozen three-atom model, Sounio can type and replay one local flat
equality decision, type one semantic refusal, refuse one mismatched fixture
replay, and abstain from one attempted compiler-authority promotion. Nominal
types reject substituting these receipts for a global law, private compiler
capability boundary, native Contest carrier, empirical or causal artifact,
clinical action, or ontology capability claim.

There is no actual compiler rewrite. Runtime evidence is the standalone scalar
mirror plus the independent oracle; the reusable imported path is check-only.

## Claims Not Supported

D7 does not establish:

- universal associativity or a total algebra;
- soundness, completeness, or usefulness of a general rewrite validator;
- collision resistance for diagnostic checksums;
- linear or single-use consumption of decision receipts;
- authenticity or unforgeability of public receipt values;
- a compiler operator-admission theorem or sealed occurrence capability;
- native Contest/IR or runtime ontology transport;
- a proof-assistant theorem or verified compiler;
- an active optimizer/IR integration;
- empirical psychiatric equivalence, causal mechanism, real suffering, consent,
  diagnosis, prognosis, treatment, or clinical action.

## Falsifiers

The bounded claim fails if:

- the two flat evaluations cease to agree;
- either bounded semantic code is wrong or the codes become equal;
- the D6 ordered atom IDs are not consumed by the local request;
- a wrong fixture occurrence records a model replay;
- a protocol receipt typechecks as a compiler capability, native Contest, global
  law, empirical/causal interpretation, or clinical action;
- the canonical compiler authority row or binding is changed by D7;
- the oracle disagrees with the native mirror on any exact checksum or mask.

## Validation Contract

Acceptance requires the D7 gate, the recursive D6-D0 gate, focused imported
check-only harness evidence, default parallel ontology validation, documentation
and semantic-registry checks, and mandatory independent LLM math review.

The selected engine must be Madaros. Backend dispatch fallback is reported
separately if observed; D7 does not use an unqualified `fallback=0` claim.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: psychiatric-regime-D7-rebracketing-protocol
Owner: Codex scientific protocol lane; compiler capability owner remains codex-2
Concept-IDs: SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL, SOUNIO-POLICY-OBSERVATION-ASSOCIATOR, SOUNIO-REBRACKETING-AUTHORITY
Intent-Preserved: parenthesization and committed evidence cannot be erased by a public model receipt
Transformation: add model equality/refusal/replay/promotion-abstention receipts without compiler mutation
Types-Changed: new stdlib and parallel ontology types only
Effects-Changed: none
IR-Changed: none
Claims-Introduced: exact bounded D7 supported claim above
Claims-Forbidden: compiler capability, global associativity, verified compiler, native Contest bridge, empirical or clinical meaning
Assumptions: exact frozen integer fixture and current D6 receipt contract
Write-Set: D7 stdlib, ontology, concept/spec, tests, oracle, gate, bindings, governance metadata, offload log
Read-Set: D6 kernel/tests/gate, canonical rebracketing contract/registry/binding, Contest frontend evidence
Positive-Witness: imported check-only API witness, native scalar mirror, finite independent oracle, parallel ontology witness
Negative-Witness: nominal clinical/Contest/compiler/law/ontology boundaries plus wrong-occurrence replay refusal
Acceptance-Gate: scripts/ci/proof_carrying_rebracketing_protocol_gate.sh
Pending-Interface: sealed-receipt-and-compiler-capability-bridge
```

## Integration Receipt

```text
Semantic-Outcome: bounded public model protocol; canonical compiler authority remains separate
Distinctions-Added: equality decision != model replay != semantic refusal != compiler-promotion abstention
Distinctions-Preserved: public receipt != private capability; checksum != identity; model != empirical claim
Distinctions-Erased: none
Runtime-Path: scalar native mirror plus independent oracle
Imported-Path: check-only; multimodule runtime blocker remains explicit
Ontology-Path: parallel nominal evidence; runtime transport false
Compiler-Path: unchanged; capability issued 0; rewrites 0
Contest-Path: unchanged; Contest/TyContest/IrContest receipts 0
Legacy-Kept: D6-D0 and all compiler reassociation paths
```
