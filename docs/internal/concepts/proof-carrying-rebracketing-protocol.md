<!-- docs:meta
topic_id: repo.docs.internal.concepts.proof-carrying-rebracketing-protocol
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.proof-carrying-rebracketing-protocol
-->

# Proof-Carrying Rebracketing Protocol

Concept-ID: `SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL`

Status: executable bounded model protocol.

Canonical surface:
`stdlib/epistemic/proof_carrying_rebracketing_protocol.sio`

Related compiler concept:
`SOUNIO-REBRACKETING-AUTHORITY`.

## Meaning

D7 records three model-level decisions without collapsing them:

1. exact equality for one declared flat-trace fixture supports a local equality
   decision and an independently recomputed model replay;
2. semantic inequality on the D6 policy-observation carrier produces a typed
   refusal;
3. promotion of local model evidence to a global law or compiler capability
   produces a typed abstention.

A fourth control refuses replay when the requested fixture occurrence differs
from the occurrence carried by the decision.

## Structural Custody

The local request consumes the D6 non-associativity receipt as well as the flat
control. It carries the exact ordered atom IDs `9101, 9102, 9103`, the two
grouping-tree audit values, the operator, protocol `8700`, and fixture
occurrence `11001`. The semantic request uses fixture occurrence `11002`.

Exact atom fields are authoritative inside this frozen model. Flat, tree,
decision, replay, refusal, promotion, and abstention checksums are diagnostic;
none is used to recover identity or claimed generally collision-free. The
bounded semantic result code is a mixed-radix encoding whose component bounds
are checked before encoding and decoded by the independent oracle.

## Authority Boundary

The public model receipts are not compiler capabilities. Construction is
unsealed, the model operator IDs `9801` and `9601` are not accompanied by
compiler operator-admission evidence, and no receipt is bound to the private
occurrence capability owned by `SOUNIO-REBRACKETING-AUTHORITY`.

D7 does not instantiate native `Contest`, `TyContest`, or `IrContest`.
It emits no Contest index, performs no source/IR mutation, consumes no compiler
capability, and establishes no compiler-wide authority. The canonical compiler
concept remains unchanged and separately admits a narrow private structural
transaction for exact bitwise operations.

The protocol is therefore related evidence, not a bridge. A future bridge would
need sealed construction, live source/compiler identity, exact structural
occurrence custody, current-IR revalidation, operator admission, and private
capability consumption. A public D7 checksum can never substitute for those
obligations.

## Runtime And Ontology Boundary

The reusable kernel and imported conformance witness are check-only evidence on
the current compiler surface. Runtime evidence comes from the standalone scalar
mirror and independent oracle. Imported multimodule execution remains covered
by `BLK-20260718-D6-MULTIMODULE-RUNTIME`; it is not silently counted as D7
runtime evidence.

The D7 ontology is a parallel nominal fixture. It establishes sibling category
boundaries but no runtime kernel-to-ontology transport and no representation of
the compiler's private capability.

## Scientific Boundary

D7 proves no universal associativity law, general translation validator,
proof-assistant theorem, causal mechanism, empirical psychiatric equivalence,
suffering state, consent, diagnosis, treatment, or clinical action. The
six-Boolean truth table exhausts only the declared local-decision predicate, and
the three decision cases are recorded rather than generated exhaustively.

## Acceptance Surface

The D7 gate must:

- typecheck the kernel, ontology module, and imported API witness;
- execute the scalar mirror and independent finite oracle;
- recompute both flat folds, both grouping trees, both bounded semantic codes,
  all refusal masks, and the wrong-occurrence control;
- reject protocol receipt substitutions into compiler, Contest, law, empirical,
  causal, clinical, and ontology claim categories;
- prove that the canonical compiler authority registry row and binding remain
  unchanged and that the protocol has its own Concept-ID;
- recursively keep D6 through D0 green.

## Pending Interface

`sealed-receipt-and-compiler-capability-bridge` remains pending. It belongs to
the compiler owner and must preserve the private canonical authority rather than
promoting this public model protocol.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
