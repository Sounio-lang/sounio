<!-- docs:meta
topic_id: repo.docs.audit.modular-pipe-coverage-map-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.modular-pipe-coverage-map-2026-06-01
-->

# Modular Compiler — Per-Type Pipe Coverage Map (2026-06-01)

**Headline:** Landing G1 brings **9 of 61** broken types (15%) fully alive end-to-end with **zero new pipe work** — they already carry all six layers (token -> constructor/producer -> checker -> dedicated IR opcode -> lowering -> native codegen). The other **52 (85%)** still need work: **19** type-check but dead-end before the backend (no IR opcode / no lowering / no codegen), **2** parse a token but never produce a value, and **31** are inert enum slots wired only to dead helpers.

> Scope note: the prompt references "63 broken types" but the supplied DATA array contains **61 entries**; counts below are over the 61 actually traced. Honest accounting — `TyAlternativeOption` is deliberately **not** counted COMPLETE: it reaches codegen only via the generic `IrCopy` path and has **no dedicated IR opcode**, so it is CHECK_ONLY per the six-layer spec.

## Verdict tally

| Verdict | Count | Meaning | What G1 alone does |
|---|---|---|---|
| COMPLETE | 9 | All 6 layers present (dedicated opcode + lowering + codegen) | **Comes alive end-to-end** |
| CHECK_ONLY | 19 | L1-L3 reachable; missing real backend (L4 opcode / L5 lowering / L6 codegen) | Still type-checks only; **needs a backend pipe** |
| FRONT_ONLY | 2 | Token exists but no reachable value/type producer | **Needs producer + full backend** |
| ENUM_ONLY | 31 | Enum slot + dead/zero-caller helpers only | **Needs everything (token/parse/check/IR/lowering/codegen)** |
| **TOTAL** | **61** | | |

## Layer-presence summary (per verdict)

| Verdict | L1 token | L2 ctor/producer | L3 checker | L4 dedicated opcode | L5 lowering | L6 codegen |
|---|---|---|---|---|---|---|
| COMPLETE (9) | yes | yes (node or builtin producer) | yes | yes | yes | yes |
| CHECK_ONLY (19) | mostly yes | producer for some, none for others | **yes** | mixed (some declare opcode but no builder) | mostly 0 | mostly none |
| FRONT_ONLY (2) | yes | no reachable producer | partial/none | no | 0 | no |
| ENUM_ONLY (31) | no | dead builders, 0 callers | no reachable producer | no | 0 | no |

## WHAT LANDING G1 UNLOCKS — the COMPLETE list (9)

These already have every layer; once G1 (the modular front-half) lands, they compile and run with no further pipe work.

- **TyContest** — reference-complete: ExprContest node, IrContest opcode (ir.sio:298), lowering lower.sio:5803-5818, codegen lower_ir.sio:266.
- **TyRobust** — produced by `prove_robust` builtin; IrProveRobust (ir.sio:304), lowering lower.sio:4613-4629, codegen lower_ir.sio:269. (Producer is builtin-call, not a dedicated Expr node — still all six layers present.)
- **TyValidated** — `validate_manifest` builtin; IrValidated (ir.sio:309), lowering lower.sio:4639-4654, codegen lower_ir.sio:272.
- **TyAdmissible** — `admit_action` builtin; IrAdmitAction (ir.sio:317), lowering lower.sio:4657-4684, codegen lower_ir.sio:275.
- **TyDeferred** — `defer_action` builtin; IrDeferAction (ir.sio:325), lowering lower.sio:4686-4713, codegen lower_ir.sio:278.
- **TyAcquisitionPlan** — `plan_acquisition` builtin; IrPlanAcquisition (ir.sio:332), lowering lower.sio:4730-4756, codegen lower_ir.sio:281.
- **TyRecoursePlan** — `plan_recourse` builtin; IrPlanRecourse (ir.sio:339), lowering lower.sio:4811-4891, codegen lower_ir.sio:284.
- **TyAlternativeSet** — `propose_alternatives` builtin; IrProposeAlternatives (ir.sio:346), lowering lower.sio:4892-4999, codegen lower_ir.sio:287.
- **TyTransitionPlan** — `commit_alternative` builtin; IrCommitAlternative (ir.sio:353), lowering lower.sio:5100-5143, codegen lower_ir.sio:290.

Pattern: the COMPLETE set is exactly the **decision/validation + plan** family — the *plan/result* nullable-value producers (the `*Plan`/`*Set` outputs of builtins), plus the four contest/robust/validate/admit/defer primitives. Their paired *policy* inputs are NOT complete (see below).

## WHAT STILL NEEDS A PIPE

### CHECK_ONLY (19) — type-checks, needs a backend (L4 opcode / L5 lowering / L6 codegen)

Two sub-shapes here: (a) **policy/metadata** types that are compile-time-only by design (no runtime value pipe), and (b) two types with a **declared opcode but no builder/lowering** (the closest to "just add the backend").

- **TyObservedTransition** — *closest to unlockable*: opcode IrObserveTransition **declared** (ir.sio:360, has normalize/serialize/machine_ir handling) but L5=0 (no `ir_observe_transition` builder, no lowering, no dispatch), L6 none. Needs builder + lowering + codegen.
- **TyRollbackCertificate** — *closest to unlockable*: opcode IrRollbackTransition **declared** (ir.sio:367) but L5=0, L6 none. Same gap: builder + lowering + codegen.
- **TyPolicy** — annotation + `policy` item parse; ty_policy ctor; only metadata struct IrPolicyInfo (ir.sio:909), no value opcode/lowering.
- **TyDecisionPolicy** — annotation only; metadata IrDecisionPolicyInfo (ir.sio:935); no value pipe.
- **TyDeferralPolicy** — annotation only; metadata IrDeferralPolicyInfo (ir.sio:953) consumed by IrDeferAction; not itself a value.
- **TyAcquisitionPolicy** — annotation + item; ty_acquisition_policy producer; metadata IrAcquisitionPolicyInfo (ir.sio:973); passed as compile-time-only arg (lower.sio:4749).
- **TyRecoursePolicy** — annotation + item; metadata IrRecoursePolicyInfo (ir.sio:995); compile-time-only.
- **TyAlternativePolicy** — annotation + item; metadata IrAlternativePolicyInfo (ir.sio:1397); never emitted as runtime value.
- **TyTransitionPolicy** — annotation + item; metadata IrTransitionPolicyInfo (ir.sio:1535); read at check-time only.
- **TyMonitoringPolicy** — annotation + item; metadata IrMonitoringPolicyInfo (ir.sio:1657); compile-time metadata only.
- **TyAlternativeOption** — projection producer; **does lower** (lower.sio:5008-5034) but via **generic IrCopy** (no dedicated opcode) -> below COMPLETE bar; effectively runs but lacks its own pipe.
- **TyIntervention** — annotation-only wrapper; lower_intervention_type + .value/.epsilon member access; **no IR opcode, 0 lowering**.
- **TyCounterfactual** — annotation-only wrapper; checker present; no opcode/lowering.
- **TyCausalEffect** — annotation-only; identifiability logic operates on a runtime CausalGraph struct, not the type; no dedicated opcode.
- **TyPotentialOutcome** — annotation-only; ty_potential_outcome built; transform `potential_outcome_to_ate` has zero callers; no backend.
- **TyModel** — annotation-only; ty_model real producer; only trace is model_count folded into IrContest metadata; no value pipe.
- **TyModelFamily** — synthesized from `models` item; IrModelFamilyInfo metadata side-table (ir.sio:889); ~2 metadata-read lines, no instruction emission.
- **TyChan** — annotation-only (parse_chan_type); ty_chan producer; **no IrChan opcode**, 0 lowering.
- **TySessionEnd** — no token/parse, but **reachably produced** (ty_session_end from TypeSession degeneration + lower_chan_type None branch); no backend. (Labeled CHECK_ONLY as the honest closest bucket — checked + reachable + no backend.)

### FRONT_ONLY (2) — token parses but no reachable value/type producer

- **TySession** — TokenKind::Session + `session Name {}` item parse, but the type-annotation dispatch has **no Session arm** (TypeSession never built); ItemSession is a checker no-op; ty_session only reached via unreachable `session_dual`. Needs a real producer + full backend.
- **TySample** — TokenKind::Sample present and reserved, but **never consumed by any parse path**; ty_sample / sample_from_distribution orphaned. Needs parse path + producer + full backend.

### ENUM_ONLY (31) — enum slot + dead helpers only; needs the whole pipeline

No token, no parse path, no reachable producer; constructors/validators exist but have **zero callers**; the only spine touch is usually a pretty-printer case.

- Causal: **TyATE, TyCondIndep, TyTransportable, TySelectionDiagram** (dead transforms `potential_outcome_to_ate`/`transport_to_ate`, caller-less `is_valid_*`).
- Probability/info: **TyDistribution, TyConditionalDist, TyEntropic, TyMutualInfo, TyKLBounded, TyELBO, TyVariationalFamily** (orphaned `epistemic.sio` algebra, no parse path).
- Fairness/epistemic-extra: **TyFairPrediction, TyFairnessGap** (handlers exist but 0 call sites), **TyEpistemic** (distinct from the working `with Epistemic` effect; ty_epistemic 0 callers).
- Effects: **TyEffectVar, TyEffectRow, TyGradedEffect, TyScopedEffect** (row-walkers/subsumption helpers only call each other).
- Tensor/shape: **TyVecShaped, TyMatrixShaped, TyBroadcastable, TySingleton** (richest dead machinery — matmul/transpose/broadcast result-type helpers, all 0 callers; the "BRIDGE" note is a comment).
- Autodiff/stochastic: **TyDifferentiable, TyGradient, TyJacobian, TyMarkovChain, TySDE, TyMartingale, TyStationaryDist** (dead bridge chains `gradient_bridge_complete` -> `langevin_from_gradient` etc., 0 callers).
- Complexity: **TyBigO, TyAmortized** (validators `bigO_compose`/`matmul_complexity`/`check_amortized_type`, 0 callers).

## Remaining work after G1 — prioritized

1. **Cheapest wins (2):** `TyObservedTransition` and `TyRollbackCertificate` already have **declared opcodes** wired into normalize/serialize/machine_ir — they only lack the L5 builder + lowering dispatch and the L6 codegen case. Finishing these mirrors the existing `*Plan` lowerings (lower.sio:4730+) and is the highest-leverage next step.
2. **Decide policy semantics (9 policy types):** the `*Policy` family is intentionally compile-time metadata (Ir*PolicyInfo side-tables). If they should stay metadata-only, they are *done as designed* and should arguably be re-labeled out of "broken." If any must become first-class runtime values, each needs a value opcode + lowering + codegen.
3. **Epistemic wrappers as runtime values (4):** `TyIntervention`/`TyCounterfactual`/`TyCausalEffect`/`TyPotentialOutcome` parse + check but have no backend; give them either dedicated opcodes or a shared epistemic-wrapper lowering (they already expose `.value`/`.epsilon`).
4. **Front-end gaps (2 FRONT_ONLY):** wire `TySession` into the type-annotation dispatch and give `TySample` a parse path before any backend matters.
5. **Lowest priority (31 ENUM_ONLY):** these are inert enum slots backed by zero-caller helpers. Each needs the full pipeline (token/parse/check producer/IR/lowering/codegen). Treat as a design decision per family (especially the tensor-shape and autodiff families, which have the most pre-written but dead helper math) rather than mechanical pipe work — none come alive from G1.
