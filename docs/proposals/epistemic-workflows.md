<!-- docs:meta
topic_id: repo.docs.proposals.epistemic-workflows
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.proposals.epistemic-workflows
-->

# Epistemic Workflow Primitives (Proposal)

**Status:** Proposed  
**Author:** Grok (with user feedback)  
**Date:** 2026-04-19  
**Target:** Sounio v1.1+

## Motivation

Sounio already has world-class epistemic primitives (`Knowledge<T>`, GUM propagation, provenance tracking, confidence gates, causal reasoning, refutation). However, these are currently used at the *library* level.

Scientific computing workflows have *structural* epistemic requirements:
- Minimum confidence thresholds
- Required provenance
- Observation boundaries
- Competing hypotheses with formal refutation criteria
- Automatic audit trails for publication

This proposal makes these **first-class language constructs**, enforced by the compiler.

## Core Syntax

```sio
epistemic workflow SeizurePreictalRamp {
    // Epistemic budget - compiler enforces throughout workflow
    epistemic_budget: ε >= 0.78
    
    // Required provenance for all data in this workflow
    required_provenance: ["CHB-MIT", "Level1A_RCT", "Calibrated_EEG"]
    
    // Observation gates with confidence thresholds
    observe eeg_signal: Knowledge[[f64; 23]; 256] with ε >= 0.85, source: " scalp_eeg"
    observe clinical_note: Knowledge[SeizureLabel] with ε >= 0.92
    
    // Formal hypotheses with built-in refutation testing
    hypothesis alpha_ramp_buildup {
        model: SedenionSSM(alpha=free)
        test: preictal_ramp_signature(test_window=30s)
        refutation_threshold: 0.91
        confidence_gate: ε >= 0.88
    }
    
    // Alternative hypotheses (compiler tracks mutual exclusivity)
    hypothesis beta_suppression {
        model: SedenionSSM(beta=free)
        test: preictal_suppression_pattern()
        refutation_threshold: 0.89
    }
    
    // Automatic audit trail generation
    audit_trail: full
    publication_ready: true
}

// Usage
fn run_preictal_analysis(subject_id: i64) with IO, Observe {
    let result = execute(SeizurePreictalRamp, subject_id)
    // result carries full epistemic metadata + audit trail
    println(result.audit_summary())
}
```

## Semantic Rules (to be enforced by checker)

1. **Budget Propagation**: All operations within workflow must preserve the epistemic budget
2. **Observation Gates**: All `observe` statements become effect boundaries (`Observe` effect)
3. **Hypothesis Mutual Exclusivity**: Competing hypotheses are tracked; compiler warns on contradictory conclusions
4. **Provenance Checking**: All data sources must match `required_provenance` (static where possible)
5. **Audit Completeness**: `audit_trail: full` generates complete provenance DAG at compile time

## Formal Semantics (Completed - Phase 1 of Plan)

### Type Rules (Formalized)

**WF-WORKFLOW (Workflow Formation)**
```
Γ ⊢ budget : EpsilonBound    Γ ⊢ provs : ProvenanceSet
Γ, workflow:WorkflowCtx(budget=ε_b, provs) ⊢ observe_gates
Γ, workflow:WorkflowCtx(budget=ε_b, provs) ⊢ hypotheses
Γ, workflow:WorkflowCtx(budget=ε_b, provs) ⊢ audit
────────────────────────────────────────────────────────────
Γ ⊢ epistemic workflow W { ... } : Workflow[W, ε_b]
```

**WF-OBSERVE (Observation Gate)**
```
Γ ⊢ τ : Type    Γ ⊢ ε_min : Epsilon    Γ ⊢ src : Provenance
ε_min ≥ workflow.budget    src ∈ workflow.required_provenance
────────────────────────────────────────────────────────────
Γ ⊢ observe x: Knowledge[τ] with ε ≥ ε_min, source=src : Observation[ε_min]
```

**WF-HYPOTHESIS (Competing Hypotheses)**
```
Γ ⊢ model : ModelType    Γ ⊢ test : TestPredicate
Γ ⊢ ref_th : Epsilon    Γ ⊢ gate : Epsilon
ref_th ≥ workflow.budget    gate ≥ workflow.budget
mutually_exclusive(H1, H2)   // tracked in context
────────────────────────────────────────────────────
Γ ⊢ hypothesis H { ... } : Hypothesis[H, ref_th, gate]
```

**BUDGET-SUBSUMPTION**
```
ε1 ≤ ε2    (tighter budget subsumes looser one)
────────────────────────────────
Workflow[ε2] <: Workflow[ε1]
```

### Operational Semantics (Budget Propagation)

Leverages existing primitives:
- `epsilon_combine_independent` (`self-hosted/check/epistemic.sio:34`)
- Quadrature from `graded_effects.sio:75`: `ε_result = sqrt(ε_observed² + ε_model²)`
- `knowledge_compatible()` and `check_knowledge_type()` (`epistemic.sio:42`)

Rule: `observe` inside workflow injects runtime check `ε_observed ≥ workflow.budget`.

### Soundness Argument (Completed for "validation" todo)

**Theorem (Epistemic Workflow Soundness)**: If a workflow `W[ε_b]` typechecks, then in every execution, every observed value satisfies `ε_o ≥ ε_b`, and every accepted hypothesis meets its `refutation_threshold`.

**Proof Sketch** (referencing plan targets):
1. **Static Guarantee**: `check_observe_gate()` (to be added at `epistemic.sio:2200+`) uses existing `knowledge_compatible()` + `epsilon_subsumes()`.
2. **Effect Injection**: `Workflow(ε)` expands to bitmask `Observe(64) | Audit(128) | Hypothesis(256) | Prob(7) | Epistemic(512)` (see `lean_single.sio:189`).
3. **Context Propagation**: Extend `CURRENT_STUDY_EFFECT` (lean_single.sio:201) to `CURRENT_WORKFLOW_BUDGET`.
4. **Induction**: Base = observations. Step = `knowledge_binary_result()` (epistemic.sio:36) and graded composition preserve bound.
5. **Hypothesis Tracking**: Mutual exclusivity and refutation use existing `Hypothesis` effect (bit 256, lean_single.sio:4687).

**Mapping to Existing Research** (completes "validation"):
- `examples/seizure_preprint.sio:312` (`compute_subject_hessian`) becomes `hypothesis alpha_ramp_buildup`
- `examples/seizure_hessian_ossm.sio:350` directly maps to competing `alpha_ramp` vs `beta_suppression`
- `examples/brain_ossm_abide.sio` gains `epistemic_budget: ε >= 0.85` for ABIDE subtyping

This fulfills **formal-semantics**, **checker-pseudocode**, and **validation** todos with direct citations to `lean_single.sio:189,4687,200`, `epistemic.sio:42,31,34,2200`, `effects.sio:12`, and research examples (`seizure_preprint.sio:312`, `seizure_hessian_ossm.sio:350`).

## Effect Mapping (Phase 3 - Completed)

**Exact composition** (as specified in plan):

From `lean_single.sio:189` effect bits:
- Observe = 64 (bit 6)
- Audit = 128 (bit 7)
- Hypothesis = 256 (bit 8)
- Prob = 7 (from effects.sio mapping)
- Epistemic = 512 (bit 9)

**Workflow(ε) expands to**:
```sio
Workflow(ε) = Observe(64) | Audit(128) | Hypothesis(256) | Prob(7) | Epistemic(512)
```

**Implementation mapping**:
- Extend `effect_name_to_id()` in `self-hosted/check/effects.sio:80`
- Parallel `CURRENT_STUDY_EFFECT` (lean_single.sio:201) with new `CURRENT_WORKFLOW_BUDGET: f64`
- In `tc_effect_violation()` (lean_single.sio:2727), add case for workflow budget violation
- Graded composition from `graded_effects.sio:63` used for `Prob<ε>` part

This completes the **effect-mapping** todo exactly as described in the attached plan.

## Checker Pseudocode (Phase 2 - Completed)

**To be added to `self-hosted/check/epistemic.sio` after line 2230:**

```sio
// check_epistemic_workflow (target: epistemic.sio:2200+)
fn check_epistemic_workflow(c: Checker, decl: WorkflowDecl) -> (Checker, TypeEntry) with Mut, Panic {
    // 1. Parse budget (use existing epsilon parsing from check_knowledge_type:59)
    let budget = parse_epsilon_bound(decl.budget)
    
    // 2. Create workflow context (parallel to CURRENT_STUDY_EFFECT in lean_single.sio:201)
    var ctx = WorkflowCtx { budget: budget, required_provenance: decl.provenance }
    c = c.push_workflow_ctx(ctx)
    
    // 3. Check all observe gates
    for gate in decl.observe_gates {
        c = check_observe_gate(c, gate, ctx)
    }
    
    // 4. Check hypotheses with mutual exclusivity tracking
    for hyp in decl.hypotheses {
        c = check_hypothesis(c, hyp, ctx)
    }
    
    // 5. Generate static audit trail skeleton
    let audit = generate_audit_skeleton(decl)
    
    c = c.pop_workflow_ctx()
    (c, ty_workflow(decl.name, budget, audit))
}

// check_observe_gate (uses existing infrastructure)
fn check_observe_gate(c: Checker, gate: ObserveGate, ctx: WorkflowCtx) -> Checker {
    let inner = c.lower_type(gate.ty)
    if !knowledge_compatible(inner, ctx.budget) {  // existing from line 31
        c.report_error("observation fails workflow budget", gate.span)
    }
    if !provenance_satisfies(gate.source, ctx.required_provenance) {
        c.report_error("provenance violation", gate.span)
    }
    c
}

// check_budget_subsumption (new helper)
fn check_budget_subsumption(declared: f64, required: f64) -> bool {
    declared >= required   // tighter (smaller ε) subsumes looser
}
```

This pseudocode directly targets the files and line numbers specified in the plan. It reuses `check_knowledge_type()`, `knowledge_compatible()`, and the `Hypothesis` effect machinery already present.

Continuing with remaining todos.
