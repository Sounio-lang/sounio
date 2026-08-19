<!-- docs:meta
topic_id: repo.docs.audit.typekind-concept-cross-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.typekind-concept-cross-2026-08-19
-->

# TypeKind × concept registry — spec skeleton

**This is a census.** It does **not** reclassify `docs/internal/concepts/registry.tsv`. Rows below are observational. The founder decides whether a TypeKind is a missing Concept-ID, or a concept is a missing TypeKind.

**SHA of `origin/main`:** `2b4d217a04` (v2 re-evaluation; v1 rows were at `98eb2b4f41`)  
**Counts measured:** TypeKind = **99** (`self-hosted/check/types.sio`). TypeExprKind = **54** (`self-hosted/parser/ast.sio`). Concept-IDs on this worktree = **24**. `origin/main` at this SHA also carries `SOUNIO-EXACTNESS` (#1941) — a 25th row. This census does **not** add a cross for it and does **not** reclassify `registry.tsv`.

Family G positions (protocol v2, run this turn): [`TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.md`](TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.md).  
Machine table: [`TYPEKIND_CONCEPT_CROSS_2026-08-19.tsv`](TYPEKIND_CONCEPT_CROSS_2026-08-19.tsv).

## The inversion

The usual failure in this repository is documentation claiming more than the compiler does. The TypeKind enum is the other way around.

- 99 TypeKinds in the checker.
- 24 Concept-IDs in the registry.
- 22 TypeKinds with no documentation (founder brief).
- Family G adds the sharper cut: **privacy and justice have TypeKinds and no Concept-ID at all.** `DiffPrivate`, `DPBudget`, `FairPrediction`, `FairnessGap` do not appear in `registry.tsv`. The word "privacy" in the concept tree is a zero-event gate path, not ε. The word "transport" in the registry is D11 shift-robust *risk* transport, not Pearl/Bareinboim `TyTransportable`.

If the suspicion is right — founder concepts that gained an enum variant and never gained a registry row — the concept registry is **under-representing the language**. That is the inverse of the problem we usually have. This table is the skeleton a spec would have to walk before it declared 99 types.

`Fit` is not a promotion:

| Fit | Meaning |
|---|---|
| embodies | a reader of the concept contract would look for this TypeKind first |
| related | same neighbourhood; not the same object |
| none | no TypeKind carries this concept |
| undocumented-kind | TypeKind is in the 22; concept column is the best *name-level* neighbour, or `none` |

## 1. For each of the 24 concepts: is there a TypeKind that embodies it?

| Concept-ID | registry status | TypeKind that would embody it | fit | why this is not a promotion |
|---|---|---|---|---|
| SOUNIO-ZERO-PROVENANCE | executable | none | none | Surface is `stdlib/epistemic/zero_event.sio`. `TyUnobserved` is "not yet inspected", not "zero is not a value". |
| SOUNIO-EPISTEMIC-NUMERIC-VALUE | executable | TyKnowledge | embodies | `Knowledge<T>` is the inhabited numeric carrier. Ran this turn: `measure` constructs it; coerce to f64 is E001. |
| SOUNIO-NONASSOCIATIVE-ORDER | executable | TyHyper | related | Associator lives in stdlib algebra; Hyper is the algebra tag, not the order. |
| SOUNIO-RNA-CD-INDUCTIVE-BIAS | hypothesis | none | none | Research manifest. No TypeKind. |
| SOUNIO-EXPLICIT-DISCHARGE | executable | TyDeferred | related | Discharge is an act. Deferred is a certificate type. Family B owns the kind. |
| SOUNIO-PHYSICAL-OBSERVATION | hypothesis | none | none | `stdlib/physics`. No observation TypeKind. |
| SOUNIO-PRECISION-PRESERVATION | executable | TyF128, TyF256 | related | Format-identity kinds exist; the concept's surface is `qd128.sio`, not those kinds. |
| SOUNIO-DYADIC-NONREDUCTION | executable | none | none | Stdlib epistemic. No TypeKind. |
| SOUNIO-RELATIONAL-ASSOCIATOR | executable | none | none | Stdlib epistemic. No TypeKind. |
| SOUNIO-PROOF-CARRYING-INFERENCE | executable | TyProof, TyContest, TyRobust | related | Proof/Contest/Robust are carriers; the concept's surface is a stdlib contest protocol. |
| SOUNIO-SECOND-ORDER-COMPILATION | hypothesis | none | none | Architecture doc. No TypeKind. |
| SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE | executable | TyHyper | related | ZD evidence is stdlib/eisa; Hyper is the algebra tag. |
| SOUNIO-SCIENCE-RESEARCH-BOUNDARY | executable | none | none | Docs/policy. No TypeKind. |
| SOUNIO-REBRACKETING-AUTHORITY | hypothesis | none | none | IR (`opt_cleanup.sio`). No TypeKind. |
| SOUNIO-ORDERED-PATH-PROVENANCE | executable | none | none | IR path. No TypeKind. Family G's import-FO work preserved this concept without adding a kind. |
| SOUNIO-REFLEXIVE-INQUIRY | executable | none | none | Stdlib protocol. No TypeKind. |
| SOUNIO-ENDOGENOUS-OBSERVABILITY | executable | TyUnobserved | related | Unobserved is type-state; the concept is a proof-carrying protocol. |
| SOUNIO-POLICY-STATE-FEEDBACK | executable | TyPolicy, TyDecisionPolicy | related | Concept surface is stdlib proof-carrying policy. TypeKinds are Family B. DecisionPolicy is undocumented. |
| SOUNIO-POLICY-OBSERVATION-ASSOCIATOR | executable | TyMonitoringPolicy, TyObservedTransition | related | Same split: stdlib protocol vs Family B kinds. Both kinds undocumented. |
| SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL | executable | TyProof | related | Proof is a certificate kind; the concept is a protocol. |
| SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION | executable | TyIntervention, TyCounterfactual, TyCausalEffect, TyCondIndep | related | Causal TypeKinds exist; CondIndep is undocumented; the concept's surface is a stdlib protocol, not those kinds. |
| SOUNIO-PROOF-CARRYING-STATISTICAL-COVERAGE-EMPIRICAL-BINDING | executable | TyDistribution, TySample | related | Probabilistic kinds exist; the concept is a sealed-coverage protocol. |
| SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY | executable | TyValidated, TyValidation, TyAdmissible | related | Validity kinds exist; the concept is affine warrant + institutional authority. |
| SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT | executable | TyTransportable, TySelectionDiagram | related **by name only** | Registry transport is D11 *risk* transport (`proof_carrying_shift_robust_risk_transport.sio`). TyTransportable is Pearl/Bareinboim S-admissibility. Same English word, two objects. Both TypeKinds are undocumented. |

**Summary of 24:** embodies=1 (Knowledge). related=12. none=11.

Eleven founder concepts have **no** TypeKind. That is the familiar direction (concept without a type). The inversion sits in the next table.

## 2. For each of the 22 undocumented TypeKinds: is there a concept under another name?

The 22 from the archaeology directive. "docs hits" this turn counted `docs/` excluding `training/` and `archive/`. A hit of 1 is almost always the June 2026 coverage map listing the kind as empty.

| TypeKind | docs (excl. training/archive) | concept under another name | fit | note |
|---|---|---|---|---|
| ModelFamily | 2 | none | undocumented-kind | Sibling of TyModel. No Concept-ID for model families. |
| AcquisitionPolicy | 1 | SOUNIO-POLICY-STATE-FEEDBACK | related | Family B Garden this turn. Policy concept is a stdlib protocol, not this kind. |
| RecoursePolicy | 1 | SOUNIO-POLICY-STATE-FEEDBACK | related | Family B Garden. |
| AlternativePolicy | 1 | SOUNIO-POLICY-STATE-FEEDBACK | related | Family B Garden. |
| TransitionPolicy | 1 | SOUNIO-POLICY-STATE-FEEDBACK | related | Family B Garden. |
| MonitoringPolicy | 1 | SOUNIO-POLICY-OBSERVATION-ASSOCIATOR | related | Family B Garden. |
| DecisionPolicy | 1 | SOUNIO-POLICY-STATE-FEEDBACK | related | Family B Garden. |
| DeferralPolicy | 2 | SOUNIO-EXPLICIT-DISCHARGE | related | Family B Hypothesis (E097 names the *item*, not the TypeKind). |
| GradedEffect | 1 | none | undocumented-kind | Effect-row grade. No Concept-ID. Not an effect in CLAUDE.md's nine. |
| SessionEnd | 1 | none | undocumented-kind | Session type-state. No Concept-ID. |
| CondIndep | 1 | SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION | related | Conditional independence as a type. Concept is a path-identification protocol. |
| Transportable | 1 | SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT | related by name only | Pearl transport ≠ D11 risk transport. Do not collapse. |
| SelectionDiagram | 1 | SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT | related by name only | Same collision. |
| FairnessGap | 1 | **none** | undocumented-kind | Family G Garden this turn. **No justice / fairness Concept-ID.** |
| ConditionalDist | 1 | none | undocumented-kind | Sprint 23 P. No Concept-ID. |
| VecShaped | 1 | none | undocumented-kind | Sprint 23 V. No Concept-ID. |
| MatrixShaped | 1 | none | undocumented-kind | Sprint 23 V. No Concept-ID. |
| VariationalFamily | 1 | none | undocumented-kind | Sprint 23 X. No Concept-ID. |
| MarkovChain | 1 | none | undocumented-kind | Sprint 24 B. No Concept-ID. |
| StationaryDist | 1 | none | undocumented-kind | Sprint 24 B. No Concept-ID. |
| SliceMut | 0 | none | undocumented-kind | Language primitive. No Concept-ID, and none should be invented by this census. |
| RawPtr | 0 | none | undocumented-kind | Language primitive (`*const T` / `*mut T`). Same. |

**Family G kinds that are documented only as comments / lying tests, and still have no Concept-ID:**

| TypeKind | family-G v2 | deepest named layer | Concept-ID | note |
|---|---|---|---|---|
| DiffPrivate | Executable | checker (no HLIR) | **none** | Privacy is not in the registry. Certo=`as`+id passes; meaning-errado (compose) also passes. |
| DPBudget | Executable | checker (no HLIR) | **none** | Two queries do not spend. No concept to under-represent; the kind is ahead of both docs and registry. |
| FairPrediction | Garden | checker (no TypeExpr, no HLIR) | **none** | Justice is not in the registry. Ghost-identical to NoSuchType. |
| FairnessGap | Garden | checker (no TypeExpr, no HLIR) | **none** | In the 22. Zero docs. Zero concept. |

## 3. Parser surface vs enum (why 99 ≠ 54)

54 TypeExpr kinds parse. 99 TypeKinds exist. The 45 without a TypeExpr can only appear as `TyNamed("TheirName")` if the user writes the English word — which is what FairPrediction / FairnessGap / NoSuchType did this turn (all three check OK as signatures).

TypeExpr present, relevant to the inversion (not a complete 54-list):

- Present: Knowledge, DiffPrivate, DPBudget, Model, the Family B Policy/Plan cluster, Contest, Robust, Intervention, Counterfactual, Validated, Admissible, Deferred, Chan, Session, CausalEffect, Aleatoric, PotentialOutcome, Proof, Lemma, Axiom, RawPtr.
- Absent (so the TypeKind is unreachable as itself): FairPrediction, FairnessGap, ModelFamily, GradedEffect, SessionEnd, CondIndep, Transportable, SelectionDiagram, Distribution, Sample, ConditionalDist, Entropic, MutualInfo, KLBounded, VecShaped, MatrixShaped, Broadcastable, ELBO, VariationalFamily, Differentiable, Gradient, Jacobian, MarkovChain, SDE, Martingale, StationaryDist, BigO, Amortized, Unobserved, plus most numeric aliases that share a primitive TypeExpr.

A spec that listed the 99 would be listing the enum. A spec that listed the 24 would be listing the concepts the founder has claimed. Neither list is the language. The language is the intersection that survives the ladder **and** still has a name in every layer.

## 3b. Layer debt (founder rule 3) — measured this turn

Directive: every type must exist in every layer. A checker kind the IR does not name is erasure.

| measure | n |
|---|---:|
| TypeKind | 99 |
| HlirTypeKind unique | 42 (`Contest`/`Robust` duplicated in source → 44 variants) |
| same stem checker→HLIR | **19** (Array Bool Contest Counterfactual F32 F64 I128 I32 I64 I8 Intervention Knowledge Robust Tuple U128 U32 U64 U8 Validated) |
| HLIR-only unique | **23** (founder brief said 24; measured 23) |
| of which algebra-only (Octonion, Sedenion, Quat*, Dual, Vec*, Mat*) | **17** |

Family G is checker-only in the first direction: four TypeKinds, zero `HlirTypeKind` names, zero IR/codegen names. The inverse debt is the 17 algebra kinds the backend names and the language does not. Both directions are debt. Neither is a Family G Concept-ID.

Compile of `1.0 as DiffPrivate<f64>` / `as DPBudget<f64>` / `as FairPrediction<f64>` all emitted 8648-byte ELFs this turn. The program survives; the name does not.

## 4. What a spec may not say

- That Sounio has differential privacy as a type. DiffPrivate is Executable under protocol v2 (certo passes, meaning-errado also passes). DPBudget does not spend.
- That Sounio has fairness as a type. FairPrediction and FairnessGap are Garden.
- That the (as, coerce) pair is Claim-ready. `NoSuchType` has the same pair.
- That the 24 concepts cover the type system. Eleven concepts have no TypeKind; a cluster of privacy / justice / causal-transport / session / shape / process TypeKinds have no Concept-ID.
- That `TyTransportable` is `SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT`. Same word, two objects.

The decision to add Concept-IDs for privacy and justice, or to delete the TypeKinds, is the founder's. This census does not do either.
