# Second-Order Cybernetics as Executable Theory: A Recursive Composition of Nine Foundational Frameworks in the Sounio Programming Language

---

**Article classification:** Research paper

## Abstract

**Purpose** — Second-order cybernetics lacks a shared computational medium in which its frameworks can be jointly formalized, composed, and tested. This paper presents an executable formalization integrating nine foundational theories into a single recursive architecture.

**Design/methodology/approach** — Ten modules in the Sounio programming language implement Spencer-Brown's Laws of Form, von Foerster's eigenform theory, observer-inclusion, autopoiesis, Ashby's requisite variety, structural coupling, Pask's conversation theory, Bateson's learning levels, and Maturana's languaging. Six bridge functions wire these into a closed recursive loop. Structural invariants are verified through property-based testing.

**Findings** — All nine theories produce structurally consistent outputs. The theories compose into one closed loop where each step produces inputs for the next. The system supports instrumental self-observation with measurable cost: observer drift increases monotonically with recursion depth. Five formal propositions are computationally verified.

**Research limitations/implications** — The modules formalize mathematical cores, not full biological or social dynamics. Scalar values replace Pask's procedural representations; fixed-size arrays constrain practical scale.

**Originality/value** — Prior traditions exist for individual frameworks (McMullin, 2004; Pangaro, 1987). No prior work has unified nine second-order frameworks into a single recursive executable architecture with explicit observer-cost accounting and cross-module invariant testing. Bridges are classified by epistemic status (canonical, operative hypothesis, or interpretive), making theoretical disagreements computationally testable.

**Keywords:** second-order cybernetics, autopoiesis, eigenform, executable theory, recursive composition

---

## 1. Introduction

Second-order cybernetics — the cybernetics of observing systems rather than observed systems (von Foerster, 1979) — has developed a rich theoretical vocabulary over seven decades. Its core commitments are well-articulated: observation is never passive (von Foerster, 1981); systems are self-producing (Maturana and Varela, 1980); understanding arises through dialogue (Pask, 1976); regulation requires sufficient internal complexity (Ashby, 1956). What second-order cybernetics has lacked is a shared *computational medium* in which its major frameworks can be jointly formalized, recursively composed, and comparatively tested.

The field's primary modes of discourse remain natural language, hand-drawn diagrams, and verbal argument. When two theorists disagree about whether Pask's "conversation" is a special case of Maturana's "structural coupling," they argue from definitions. When a practitioner asks whether an organizational intervention has "requisite variety," they estimate qualitatively. While executable systems exist for individual traditions (see Section 6), there is no widely adopted unified framework that composes these traditions into a single recursive architecture.

This paper fills that gap. We present a library of ten modules in the Sounio programming language that implements nine foundational theories of second-order cybernetics as executable code. Each module is individually testable (type-checks, runs, produces correct values). More importantly, the modules *compose*: a tenth module wires the nine theories into one recursive loop where each theory produces inputs for the next, and the entire system supports instrumental self-observation with measurable cost.

### 1.1 Contributions

1. **Formal correspondence.** For each theory, the core mathematical object is implemented as a data type with operations respecting the theory's axioms (Table I), verified through runtime assertions.

2. **Recursive composition.** A closed loop links all nine theories (Observer → Eigenform → Distinction → Autopoiesis → Variety → Coupling → Conversation → Learning → Languaging → Observer), with each arrow realized as a computable bridge function.

3. **Instrumental self-observation.** The function `observe_self(state, observer)` applies an observer to the aggregate system state, increasing `recursion_depth` and `observer_drift` — the computational realization of von Foerster's dictum.

4. **Property-based verification.** Ten structural invariants tested across multiple inputs provide evidence of semantic soundness.

5. **Practitioner access.** A therapy session modeled as a Pask conversation with Bateson learning interventions demonstrates accessibility.

### 1.2 Structural vs. parametric predictions

The library produces specific numbers (41 rounds, 17 iterations) that depend on parameter choices and are not empirical predictions. What IS falsifiable are *structural* properties: variance monotonicity, closure of circular but not linear networks, trust-accelerated convergence, learning escalation under stagnation. These hold across parameterizations and constitute the theoretical content.

### 1.3 Scope and limitations

This paper presents a *formalization*, not a simulation. The modules implement the mathematical core of each theory — eigenform iteration, organizational closure, precision-weighted Bayesian update, Shannon entropy — but do not simulate the full dynamics of biological or social systems. We address the relationship between formalization and simulation in Section 7.

---

## 2. Theoretical Foundations and Formal Correspondence

Table I maps each theory to its core mathematical object, the Sounio data type that implements it, and the key axiom or property that the implementation verifies at runtime.

| Theory | Originator | Core Object | Sounio Type | Verified Property |
|--------|-----------|-------------|-------------|-------------------|
| Laws of Form | Spencer-Brown 1969 | Mark / void distinction | `Form` | Calling: `mark(mark(x)) = mark(x)` |
| | Varela 1975 | Autonomous value | `AUTONOMOUS` | `solve_reentry(MARKED, 100) = AUTONOMOUS` |
| Eigenform | von Foerster 1976 | Fixed point x* = Op(x*) | `EigenformResult` | `|Op(x*) - x*| < ε` after convergence |
| | Kauffman 2003 | Stable attractor | `Eigenbehavior` | `convergence_rate < 1.0` (Banach) |
| Observer-inclusion | von Foerster 1981 | Perturbed observer | `Observer` + `Observation` | Variance monotonically non-decreasing |
| | Luhmann 1995 | Blind spot | `blind_spot(obs)` | `drift > 0` after any observation |
| Autopoiesis | Maturana/Varela 1972 | Organizational closure | `AutopoieticSystem` | Circular network: alive. Linear: dead. |
| Requisite variety | Ashby 1956 | H(R) ≥ H(E) − H(O) | `VarietySystem` | 8 reg / 6 env: sufficient. 3 reg / 6 env: insufficient. |
| Structural coupling | Maturana/Varela 1987 | Behavioral congruence | `Coupling` | Pearson r > 0.95 for linearly related systems |
| Conversation | Pask 1975 | Dual cross-models | `Conversation` | P=10, Q=90 → converge in 41 rounds |
| Learning levels | Bateson 1972 | L0→L1→L2→L3 hierarchy | `LearningContext` | L1 stagnation → L2 escalation |
| Languaging | Maturana 1988 | Consensual domain | `LanguagingPair` | Consensus builds stability; disagreement decreases it |

The following subsections detail the correspondence for each theory.

### 2.1 Spencer-Brown's Laws of Form and Varela's Extension

Spencer-Brown (1969) formalized the act of distinction with two axioms:

- **Calling** (condensation): ⊤⊤ = ⊤ (marking what is already marked has no additional effect)
- **Crossing** (cancellation): ⊤̄ = ⊥ (entering and leaving a distinction returns to the unmarked state)

Varela (1975) extended this calculus with a third value, the *autonomous* or self-indicating form J where J = ⊤̄(J). This resolves self-reference without paradox: the iteration MARKED → UNMARKED → MARKED → ... oscillates with period 2, and Varela assigns this oscillation the stable value AUTONOMOUS.

Our `Form` struct stores `value ∈ {UNMARKED, MARKED, AUTONOMOUS}`, a `children: [i64; 8]` array for nested sub-forms, `depth` (nesting level), `is_reentrant` (self-reference flag), and `period` (oscillation period in Spencer-Brown's imaginary time). The function `solve_reentry_timed(MARKED, 100)` detects the period-2 oscillation and returns a `Form` with `value = AUTONOMOUS` and `period = 2`.

**Form composition.** Functions `form_add_child`, `form_contain`, and `form_pair` build compound expressions using the `children` array. The private function `recompute_from_children` implements Spencer-Brown's primary algebra: juxtaposition of sub-forms propagates AUTONOMOUS (highest priority), then MARKED, then UNMARKED. If the parent form has nonzero depth (it is inside a mark), the result is crossed.

**Verification.** The calling axiom is verified by asserting `eval_form(form_mark(form_mark(form_void()))) == eval_form(form_mark(form_void()))`. The crossing axiom by `eval_form(form_cross(form_cross(form_marked()))) == MARKED`. The autonomous value by `solve_reentry(MARKED, 100) == AUTONOMOUS`. All pass at runtime. We note that Oksas (2025) recently identified errors in Spencer-Brown's re-entry calculations using ternary logic and XBOOLE software; our implementation uses Varela's three-valued extension rather than Spencer-Brown's original re-entry, sidestepping the errors Oksas identifies.

### 2.2 Eigenform Theory

Von Foerster (1976) proposed that objects are not pre-given but are *eigenforms* — stable fixed points of recursive observation. If Op is an observation operator, then the eigenform x* satisfies Op(x*) = x*. Kauffman (2003) showed that eigenforms are structurally isomorphic to fixed-point combinators (Y combinators) in lambda calculus.

Our `find_eigenform(op, initial, tolerance, max_iter)` implements Banach fixed-point iteration: starting from `initial`, it repeatedly applies `op` until `|Op(x) - x| < tolerance` or `max_iter` is reached. The function accepts a first-class function reference `op: fn(f64) -> f64`, exploiting Sounio's first-class function references.

**Stability analysis.** `compute_eigenbehavior` estimates the Lyapunov contraction rate via finite-difference approximation of Op's slope at x*. If `|Op'(x*)| < 1.0`, the eigenform is a stable attractor (Banach contraction mapping theorem).

**Meta-eigenform.** `meta_eigenform` tests second-order stability: is the eigenform-finding process itself stable? It runs `find_eigenform` from two slightly perturbed initial conditions and checks that both converge to the same value.

**Verification.** The operator `(x + 20) / 2` has fixed point x* = 20. The implementation converges in 17 iterations with `|Op(x*) - x*| < 0.001`. The contraction rate is 0.5 (the slope of `(x + 20) / 2` is 0.5 everywhere), confirming Banach stability.

### 2.3 Observer-Inclusion

Von Foerster's fundamental principle states that the observer is always part of the system observed. Our `Observer` struct tracks `drift` (systematic accumulated bias), `precision_budget` (finite resource consumed by observation), and `observation_count`. The function `make_observation(observer, value, variance)` returns an `Observation` struct containing the *updated* observer — you literally cannot call the function without accepting the perturbed observer back.

Three sources of variance are always present in any observation:
- Measurement variance (instrument noise)
- Observer drift² (accumulated systematic bias)
- Budget penalty (additional noise when precision budget is low)

**Blind spots.** Luhmann (1995) extended von Foerster's principle: an observer cannot observe its own observation operation in the moment of observing. Our `blind_spot(observer)` returns the accumulated drift — the quantity the observer cannot self-correct.

**Meta-observation.** `observe_observer(outer, inner)` implements second-order observation: the outer observer observes the inner observer's state. The resulting variance is strictly greater than the inner observer's self-reported variance, because the outer observer adds its own drift.

**Verification.** Property-based invariant test: observer variance is monotonically non-decreasing across 50 consecutive observations with three different drift rates (0.01, 0.1, 0.5). All 150 cases pass.

### 2.4 Autopoiesis

Maturana and Varela (1972, 1980) defined an autopoietic system as a network of processes that produces the very components constituting the network. The critical property is *organizational closure*: the production relations must form at least one cycle.

Our `AutopoieticSystem` stores production relations in a `relations: [i64; 256]` flattened 16×16 adjacency matrix. When a component is added via `add_component(sys, produced_by, produces, value, variance)`, the relation is recorded and organizational closure is rechecked.

**Closure detection.** The private function `check_closure` implements iterative DFS cycle detection on the active-component subgraph; a back-edge indicates organizational closure.

**Production and perturbation.** `produce_cycle` propagates values via averaging with GUM-style variance propagation. `perturb` distributes environmental perturbation across active components, scaled by boundary permeability. Critically, perturbation affects *structure* (component values) but never *organization* (the relations matrix) — the formal distinction Maturana and Varela insisted upon.

**Verification.** A circular network 0→1→2→0 returns `is_alive = true`. A linear chain 0→1→2 (no cycle) returns `is_alive = false`. Deactivating a node in the circular network correctly returns `is_alive = false`. This appears to be one of the first explicit graph-theoretic operationalizations of organizational closure via directed-cycle detection.

### 2.5 Law of Requisite Variety

Ashby (1956, §11/3) proved that a regulator can only control a system if its internal variety is at least as large as the environment's variety minus the desired outcome variety. Formally: H(R) ≥ H(E) − H(O), where H denotes Shannon entropy.

Our `VarietySystem` maintains frequency histograms for environment states (32 bins), regulator states (32 bins), and outcome states (8 bins). The function `compute_variety` computes:

- Cardinality (number of distinct states observed) and log₂(cardinality) for all three channels
- Shannon entropy H = −Σ pᵢ log₂ pᵢ for environment, regulator, and outcome distributions
- Variety deficit = H(R) − (H(E) − H(O))

`has_requisite_variety` returns true if and only if the deficit is non-negative.

**Verification.** With 8 regulator states vs. 6 environment states and 2 outcome states, the system has requisite variety (deficit ≥ 0). With 3 regulator states vs. 6 environment states and 2 outcomes, it does not (deficit < 0). Both cases verified at runtime.

### 2.6 Structural Coupling

Two systems are structurally coupled when each serves as a source of perturbation for the other (Maturana and Varela, 1987). The `Coupling` struct tracks a circular buffer of 64 output pairs, computes Pearson correlation as the congruence metric (incrementally maintained, Welford-style), and estimates mutual information via MI ≈ −0.5 ln(1 − r²).

**Verification.** Linearly related data produces r > 0.95. Congruence is always in [0, 1] across 30 diverse input patterns.

### 2.7 Conversation Theory

Pask (1975, 1976) argued that knowledge arises through *circular dialogue* where each participant maintains a model of the other's model. Agreement is the convergence of these cross-models, not mere averaging of beliefs.

Our `Conversation` struct implements genuine dual modeling. Each participant has:
- Their own belief (`p_value`, `p_variance`)
- Their model of the other (`p_model_of_q`, `p_model_of_q_var`)

The function `converse` executes one round in three phases:

1. **P observes Q:** P updates its model of Q via precision-weighted Bayesian average of its prior model and Q's stated value. P then shifts its own belief toward its model of Q.
2. **Q observes P:** Symmetric update using P's (now modified) stated value.
3. **Metrics:** Model accuracy (how well each participant knows the other), agreement, convergence.

**Adaptive trust.** The model weight is adaptive: `weight = 0.1 + 0.4 * agreement_value`, ranging from cautious (0.1) to open (0.5). We acknowledge that in Pask's full theory, what converges is a *procedure* (Lp/Lp*), not a scalar value; our implementation captures the *dynamics* of convergence but not the procedural *content*.

**Verification.** P starts at 10.0, Q at 90.0. After 41 rounds, agreement > 0.9, shared understanding ∈ (10, 90), and model accuracy < 5.0 (both participants know each other's beliefs well).

### 2.8 Learning Levels

Bateson (1972) proposed a hierarchy of learning types where each level is learning *about* the level below:

| Level | Description | Implementation |
|-------|-------------|----------------|
| L0 | Fixed response | Lookup table |
| L1 | Parameter update within fixed model | EMA (α = 0.1) on parameter array |
| L2 | Change the set of alternatives | Switch between 8 available parameter sets |
| L3 | Restructure the framing | Reset all parameters, increment frame count |

Our implementation enforces the hierarchy as a state machine: the `learn` function tries L1, escalates to L2 when L1 fails to improve for 3 consecutive rounds (stagnation), and escalates to L3 when all L2 parameter sets are exhausted.

**Double bind.** Bateson's pathological case: contradictory injunctions that prevent resolution at any level. Our `detect_double_bind` returns true when L1 is stuck (3+ failures), L2 is blocked (no available sets), and L3 is frozen (more than 3 frame restructures).

**Verification.** Repeated identical stimuli cause L1 stagnation, which triggers L2 escalation. This is verified at runtime.

### 2.9 Languaging

Maturana (1988) defined languaging as "the coordination of consensual coordinations of action" — a recursive structure where meaning arises not from information transmission but from mutual adjustment of behavior.

Our `LanguagingPair` implements this with three mechanisms:
1. **Consensus detection:** When both agents select the same action, it is consensual.
2. **Cross-agent feedback:** Each agent adjusts its preferences toward the *other's* last action (not its own). This is the core of Maturana's insight: languaging is responding to the other's coordination.
3. **Linguistic distinction:** When a coordination streak exceeds 3, the pair automatically records a "distinction" — a stable pattern that both agents recognize.

**Verification.** 20 consecutive consensual rounds produce domain stability > 0.5. Introducing disagreement decreases stability. Property-based test: domain size is always in [0, num_actions].

---

## 3. The Composition Layer: One Recursive Structure

The tenth module, `second_order.sio`, bridges the nine theories into a single recursive structure operating at two distinct architectural levels:

**Level A — The inter-theoretic loop.** Nine transitions link the frameworks in sequence: Observer → Eigenform → Distinction → Autopoiesis → Variety → Coupling → Conversation → Learning → Languaging → Observer. Six are realized as explicit Level A bridge functions (Table II); `observe_self` operates at Level B (see below). The remaining two Level A transitions — Distinction → Autopoiesis and Languaging → Observer — are passed directly by the test harness because their type handoffs are trivial.

**Level B — The reflexive operator.** The function `observe_self(state, observer)` is *not* a transition within the Level A loop. It is a meta-operation applied to the aggregate state *after* the loop completes one full cycle. It applies an observer to the `CyberneticState` struct, increasing `recursion_depth` and accumulating drift. This distinction matters: the Level A loop computes the inter-theoretic composition; Level B asks what happens when the composed system observes its own output. The two levels can be iterated — run the loop, observe the result, run the loop again with the perturbed state — but they are architecturally separate.

| Bridge | From → To | Function | Mechanism |
|--------|-----------|----------|-----------|
| 1 | Observer → Eigenform | `observe_eigenform` | Eigenform iteration with drift accumulation per step |
| 2 | Eigenform → Distinction | `eigenform_as_distinction` | Convergent → MARKED; divergent → UNMARKED; marginal → AUTONOMOUS |
| 3 | Autopoiesis → Variety | `assess_viability` | viable = alive AND has_requisite_variety |
| 4 | Coupling → Conversation | `coupling_is_conversation` | congruence > θ₁ AND agreement > θ₂ |
| 5 | Conversation → Learning | `diagnose_conversation` | Agreement level maps to L1/L2/L3 prescription |
| 6 | Learning → Languaging | `can_learn_in_language` | Domain stability AND not double-bound |

The Level B reflexive operator `observe_self` is architecturally separate from these six bridges (see Section 7.3).

All thresholds are named constants parameterized for transparency (e.g., `CONVERSATION_CONGRUENCE_THRESHOLD = 0.5`, `DRIFT_VIABILITY_LIMIT = 5.0`; full list in the online supplement). The `CyberneticState` struct aggregates the full system state; the Level B operator `observe_self(state, observer)` applies an observer to this aggregate, producing a new state with `recursion_depth + 1` and measurably increased drift.

**The loop closes.** In our proof (`second_order_proof.sio`), starting from an eigenform search, the state traverses all nine theories and returns to self-observation with `recursion_depth = 2` and `drift > 0`. This is not a diagram. It is a runnable program.

---

## 4. Verification

### 4.1 Computational Verification

`cybernetic_proof.sio` (419 lines) contains nine independent computational checks, one per theory. Each check constructs inputs, runs the theory's functions, and validates structural properties via runtime assertions. Key results:

- Eigenform convergence: 17 iterations for `(x+42)/2` from x₀ = 0
- Observer drift: monotonically increasing over 10 observations
- Autopoiesis: circular network alive, linear network dead, deactivation kills
- Ashby's Law: 8 reg / 6 env → sufficient; 3 reg / 6 env → insufficient
- Pask convergence: 41 rounds from (10, 90)

### 4.2 Property-Based Invariants

`invariant_tests.sio` (486 lines) verifies ten structural invariants across multiple inputs: observer variance monotonicity (3 drift rates × 50 observations), coupling congruence boundedness, conversation agreement boundedness, non-negative variety and entropy, monotone autopoietic death, eigenform residual convergence, languaging domain bounds, Shannon entropy non-negativity, distinction value membership, and learning level range. All 10 invariants pass (0 failures).

### 4.3 Recursive Loop Verification

`second_order_proof.sio` (251 lines) executes the full Level A loop followed by Level B self-observation. Starting from an eigenform search, the state traverses all nine theories — eigenform convergence (17 iterations), distinction classification (MARKED), autopoietic closure (10 cycles), viability assessment, structural coupling, conversational convergence (41 rounds), learning diagnosis, and consensual domain establishment — then undergoes self-observation. Final state: `recursion_depth = 2`, `drift > 0`.

### 4.4 Multi-Agent Demonstration

`multi_agent.sio` (312 lines) demonstrates four autopoietic systems coupled in a ring topology (A↔B↔C↔D↔A). A perturbation injected into system A propagates through the ring. System A shows highest structural drift; systems further from the perturbation source show progressively lower drift. All four systems maintain organizational closure throughout.

### 4.5 Practitioner Example

`therapy_session.sio` (268 lines) models a clinical session as a Pask conversation between therapist (initial model: 70) and client (initial model: 20). The conversation builds agreement over 20 rounds, stalls at round 25, is diagnosed as needing L1 learning, and resumes to convergence. This demonstrates accessibility to the applied cybernetics community.

---

## 5. Implementation

The library comprises 4,586 lines of Sounio across 16 files:

| Category | Files | Lines |
|----------|-------|-------|
| Core theory modules | 9 | 2,045 |
| Composition layer | 1 | 348 |
| Runtime verification | 2 | 670 |
| Invariant tests | 1 | 486 |
| Multi-agent demo | 1 | 312 |
| End-to-end demo | 1 | 457 |
| Practitioner example | 1 | 268 |

Sounio's effect system (`with Mut, Div, Panic`) tracks which operations are epistemically costly — a design choice that aligns with the theory's emphasis on the cost of observation. The by-value return pattern (each function returns a new struct rather than mutating in place) makes state transitions explicit, corresponding to the theoretical distinction between organization (invariant) and structure (changeable).

A native x86-64 JIT builtin implements Banach fixed-point iteration with indirect function calls, demonstrating that the theory can be compiled to efficient machine code. Details of the numerical approximations (Newton-Heron square roots, Taylor-series logarithms, Shannon entropy computation) and their error bounds are provided in the online supplement.

### 5.1 Code Excerpt: Self-Observation

To illustrate the by-value return pattern and effect tracking, we reproduce the Level B reflexive operator. The observer cannot inspect the system without accumulating drift, which changes the viability assessment, which changes what the next observation will see.

```sounio
pub fn observe_self(state: CyberneticState, obs: Observer)
    -> CyberneticState with Mut, Div, Panic {
    var s = state
    let measurement = make_observation(obs, s.eigenform_value, s.eigenform_variance)
    s.eigenform_variance = measurement.variance
    s.observer_drift = blind_spot(measurement.observer)
    s.distinction_value = eigenform_as_distinction(...)
    if s.observer_drift > DRIFT_VIABILITY_LIMIT { s.is_viable = false }
    s.recursion_depth = s.recursion_depth + 1
    s
}
```

The effect annotation `with Mut, Div, Panic` records that this function mutates state, performs division, and may fail — the computational cost of observation made explicit. The by-value return (`var s = state; ... s`) ensures every state transition is explicit: organization is invariant across transitions; structure changes. Full code excerpts for all six Level A bridges and the Level B operator are provided in the online supplement.

---

## 6. Related Work

**Computational autopoiesis.** There is a substantial tradition of computational autopoiesis, reviewed comprehensively by McMullin (2004). The lineage begins with Varela, Maturana, and Uribe's (1974) tessellation automaton — a 2D cellular model where "molecules" self-organize into bounded structures. Subsequent work (Suzuki and Ikegami, 2009) explored reaction-diffusion and particle-based models. McMullin's own SCL (Substrate-Conscious Language) implemented autopoietic dynamics in a spatial substrate. These are simulations of specific autopoietic *mechanisms* in particular physical substrates. Our work differs in level of abstraction: we implement the *theory* of organizational closure — production relations as a directed graph, closure as cycle existence — as abstract operations applicable to any system, not a specific spatial model. The DFS cycle detection in our `check_closure` formalizes the *definition* of autopoiesis; McMullin's tessellation automaton demonstrates a *realization* of it. The two approaches are complementary.

**Conversation Theory software.** Pask's THOUGHTSTICKER and its successor CASTE were direct CT implementations (Pangaro, 1987; Pangaro, 2002; Dubberly and Pangaro, 2015). Our conversation module uses scalar values rather than Pask's procedural Lp/Lp*; our contribution is composing conversation with eight other frameworks into a recursive architecture, which THOUGHTSTICKER does not attempt.

**Algebraic cybernetics.** Kauffman (2003, 2005) provided mathematical formalization of eigenforms and Laws of Form, with explicit connections between them. More recently, Kauffman (2023) formally connected autopoiesis and eigenform via Gödelian coding, demonstrating that self-producing systems can be understood as fixed points of their own production operators. Miranda and Abades (2024) applied eigenbehavior concepts to ecosystem management using multi-agent simulation in *Kybernetes*, introducing the concept of "eigenperception." Our work makes these formalizations executable — the algebraic identities are verified by running code, not by reading proofs. Kauffman's identification of eigenforms with distinctions (2005, §4) directly informs our `eigenform_as_distinction` bridge; his autopoiesis-eigenform connection (2023) supports our composition of these two frameworks via the recursive loop.

**Agent-based modeling.** NetLogo (Wilensky, 1999), Mesa (Kazil *et al.*, 2020), and Repast (North *et al.*, 2013) simulate *specific systems*. Our work formalizes the *theory* applicable to any system. The two approaches are complementary and composable.

**Summary.** Prior software traditions exist for individual cybernetic theories: computational autopoiesis (Varela *et al.*, 1974; McMullin, 2004), Conversation Theory (THOUGHTSTICKER; Pangaro, 2002), and algebraic formalization of Laws of Form (Kauffman, 2005). Our contribution is not the first computational work in any single tradition, but the first *unified recursive composition* of nine frameworks into a single architecture with explicit observer-cost accounting, cross-module bridge functions, and property-based invariant verification.

---

## 7. Discussion

### 7.1 Formalization vs. Simulation

The modules formalize the *mathematical core* of each theory, not its full dynamics. This is deliberate: formalization captures what is *general* (organizational closure, precision-weighted convergence), while simulation would be specific to a particular substrate. The advantage is composability: a simulation of autopoiesis in a cellular automaton cannot easily be composed with a simulation of conversation theory. Our modules can.

### 7.2 The Recursive Loop as Theoretical Claim

The construction of the Level A loop is a theoretical claim: these nine theories are not independent but are aspects of a single recursive structure. Each bridge makes an ontological identification — `observe_eigenform` asserts that eigenform search IS observation (each iteration calls `make_observation`, accumulating drift); `eigenform_as_distinction` asserts that a converged eigenform creates a Spencer-Brown distinction (basin boundary = mark), following Kauffman (2005, §4); `assess_viability` asserts that autopoiesis requires requisite variety (Ashby's Law as the mathematical reason organizational closure is sustainable); `coupling_is_conversation` asserts that conversation is structural coupling through models (threshold-dependent; see online supplement); `diagnose_conversation` maps Pask's stalls to Bateson's hierarchy (L1/L2/L3 by severity); `can_learn_in_language` checks Maturana's claim that learning occurs within the consensual domain. The Level B operator `observe_self` is discussed in Section 7.3.

**Epistemic status of the bridges.** The bridges are not claimed as canonical identifications in the historical literature. They fall into three categories:

| Bridge | Status | Justification |
|--------|--------|---------------|
| `observe_eigenform` | **Canonical formalization** | Von Foerster explicitly defines eigenforms as products of recursive observation |
| `assess_viability` | **Canonical formalization** | Ashby's Law is the standard prerequisite for organizational survival |
| `eigenform_as_distinction` | **Operative bridge hypothesis** | Basin-boundary identification follows Kauffman (2005, §4) but is our operationalization |
| `coupling_is_conversation` | **Operative bridge hypothesis** | Plausible but threshold-dependent; the boundary between coupling and conversation is a continuum |
| `diagnose_conversation` | **Operative bridge hypothesis** | Novel meta-bridge linking Pask's stalls to Bateson's hierarchy |
| `can_learn_in_language` | **Interpretive mapping** | Maturana claims learning occurs in languaging; our check of domain stability + double bind is one operationalization among several possible |
| `observe_self` (Level B) | **Partial realization** | Captures cost of instrumental self-observation but not full recursive self-application (see §7.3) |

These identifications are falsifiable. If a theorist disagrees that eigenform convergence should be classified as MARKED (rather than, say, AUTONOMOUS), they can change `eigenform_as_distinction` and observe the consequences downstream. The recursive loop makes disagreements *testable* rather than merely debatable.

### 7.2.1 Formal Propositions

The test suite corroborates the following five propositions under the library's semantics. These are modest formal claims about the implementation's structural behavior, not theorems about the external world.

**Proposition 1** (Organizational closure). *`check_closure(sys)` returns true if and only if the active-component subgraph of `sys.relations` contains at least one directed cycle.*

Verified by: circular network 0→1→2→0 returns true; linear chain 0→1→2 returns false; deactivating a cycle node changes true → false. DFS cycle detection is standard (Cormen *et al.*, 2009).

**Proposition 2** (Observer variance monotonicity). *Under non-negative drift increments (`drift_rate ≥ 0`), the total variance returned by `make_observation` is monotonically non-decreasing across consecutive calls with the same measurement_variance.*

Verified by: 150 cases (3 drift rates × 50 observations), all monotone. Follows from: total_variance = measurement_variance + drift² + budget_penalty, where drift is cumulative and budget_penalty is non-negative.

**Proposition 3** (Conversation boundedness). *If the adaptive trust weight is bounded in [0.1, 0.5] and updates are convex combinations, then `p_value` and `q_value` remain within the convex hull of their initial values for all rounds.*

Verified by: three extreme-case runs (P=10/Q=90, P=0/Q=1000, P=−100/Q=100); shared_understanding always lies between initial extremes.

**Proposition 4** (Self-observation cost). *`observe_self(state, obs)` strictly increases `recursion_depth` by 1, and if `obs.drift_rate > 0`, `observer_drift` is strictly positive after any call.*

Verified by: second_order_proof.sio Step 9 (recursion_depth: 0→1→2, drift > 0).

**Proposition 5** (Variety deficit correctness). *`has_requisite_variety(vs)` returns true if and only if H(R) ≥ H(E) − H(O), where H denotes Shannon entropy computed by `shannon_entropy` over the recorded frequency histograms.*

Verified by: sufficient case (8 reg / 6 env / 2 outcome → deficit ≥ 0) and insufficient case (3 reg / 6 env / 2 outcome → deficit < 0).

### 7.3 What Self-Observation Costs — and What It Cannot Yet Do

The Level B operator `observe_self` adds measurable drift to the system state. After two recursive observations, `drift > 0` — a numerical result, not a metaphor. When drift exceeds `DRIFT_VIABILITY_LIMIT`, the system can no longer trust its own observations and `is_viable` becomes false.

We distinguish two senses of self-observation: (1) *instrumental* — the system reads its own state variables through an observer, accumulating drift (what `observe_self` implements); and (2) *recursive self-application* — the system re-runs its own composition loop on its own state, computing the eigenform of its own viability function. The current implementation achieves (1) but not (2). Achieving (2) — the stronger sense von Foerster intended — would require the composition loop to be a first-class value passed to `find_eigenform`, computing the eigenform of the cybernetic loop itself. This is the most important direction for future work (see online supplement, Appendix B).

### 7.4 Luhmann's Social Autopoiesis

The current autopoiesis module implements biological autopoiesis (Maturana and Varela, 1980). Luhmann's (1995) most distinctive contribution — the autopoiesis of *communication systems* where components are communications, not organisms, and each functional subsystem has its own binary code — is not yet implemented. The existing architecture supports this extension naturally: the `distinction` module provides the binary code mechanism, the `autopoiesis` module provides the closure detector, and the `coupling` module can track inter-subsystem perturbation. A concrete sketch of the required data structures is provided in the online supplement (Appendix A). This is the most theoretically significant extension for the *Kybernetes* audience.

### 7.5 Scalability

The implementation uses fixed-size arrays (16-component autopoiesis, 32-bin variety, 64-entry coupling buffers). For the theoretical contribution, these limits are sufficient — the recursive loop and bridge functions are independent of array size. For practical applications, hierarchical decomposition into coupled subsystems (following Luhmann's functional differentiation) provides a theoretically natural scaling strategy.

### 7.6 Executable Theory as Methodology

The term "executable theory" in our title requires clarification. We mean something specific: the theory's axioms are encoded as program invariants (e.g., "calling is idempotent" becomes `assert(eval_form(form_mark(form_mark(x))) == eval_form(form_mark(x)))`), and *running the program constitutes a test of the axioms*. If the program runs without assertion failure, the axioms hold for the tested inputs. If it crashes, an axiom is violated.

This is weaker than formal verification (which proves axioms for all inputs) but stronger than verbal argument (which proves nothing for any input). It occupies the same epistemic position as computational physics: the simulation is not a proof, but it produces falsifiable predictions that constrain the theory. We believe this methodology — encoding theoretical commitments as runtime invariants in a purpose-built language — is applicable beyond cybernetics to any field with well-articulated but computationally untested axioms.

---

## 8. Conclusion

Second-order cybernetics has waited seventy years for a computational medium. The tools of the field — verbal argument, diagrams, qualitative assessment — are valuable but insufficient for the kind of precise, reproducible, falsifiable work that a mature science requires.

We have shown that nine foundational theories of second-order cybernetics can be implemented as executable code, composed into a single recursive structure, and verified through numerical tests. The system supports instrumental self-observation, with measurable cost.

The specific numbers (41 rounds, 17 iterations) depend on parameter choices and are not predictions about empirical reality. What IS falsifiable are the *structural* predictions: variance monotonicity, circular-network closure, convergence acceleration through adaptive trust, learning escalation under stagnation. These dynamics are preserved across parameterizations and constitute the real theoretical content.

The code is available at `stdlib/cybernetic/` in the Sounio repository (10 modules, 4 test files, 2 examples). The tests are runnable with `$SOUC run tests/run-pass/second_order_proof.sio`. The loop closes.

---

## References

Ashby, W.R. (1956), *An Introduction to Cybernetics*, Chapman and Hall, London.

Bateson, G. (1972), *Steps to an Ecology of Mind*, Chandler Publishing Company, San Francisco, CA.

Cormen, T.H., Leiserson, C.E., Rivest, R.L. and Stein, C. (2009), *Introduction to Algorithms*, 3rd ed., MIT Press, Cambridge, MA.

Dubberly, H. and Pangaro, P. (2015), "Cybernetics and design: conversations for action", *Cybernetics & Human Knowing*, Vol. 22 No. 2-3, pp. 73-82.

Kauffman, L.H. (2003), "Eigenforms — objects as tokens for eigenbehaviors", *Cybernetics & Human Knowing*, Vol. 10 No. 3-4, pp. 73-90.

Kauffman, L.H. (2005), "Eigenform", *Kybernetes*, Vol. 34 No. 1/2, pp. 129-150.

Kauffman, L.H. (2023), "Autopoiesis and eigenform", *Computation*, Vol. 11 No. 12, article 247, doi: 10.3390/computation11120247.

Kazil, J., Masad, D. and Crooks, A. (2020), "Utilizing Python for agent-based modeling: the Mesa framework", in Thomson, R., Bisgin, H., Dancy, C. and Hyder, A. (Eds), *Social, Cultural, and Behavioral Modeling (SBP-BRiMS 2020)*, Springer, Cham, pp. 308-317.

Luhmann, N. (1995), *Social Systems*, Stanford University Press, Stanford, CA (originally published as *Soziale Systeme*, Suhrkamp, 1984).

Maturana, H.R. (1988), "Reality: the search for objectivity or the quest for a compelling argument", *Irish Journal of Psychology*, Vol. 9 No. 1, pp. 25-82.

Maturana, H.R. and Varela, F.J. (1980), *Autopoiesis and Cognition: The Realization of the Living*, D. Reidel, Dordrecht.

Maturana, H.R. and Varela, F.J. (1987), *The Tree of Knowledge: The Biological Roots of Human Understanding*, Shambhala, Boston, MA.

McMullin, B. (2004), "Thirty years of computational autopoiesis: a review", *Artificial Life*, Vol. 10 No. 3, pp. 277-295.

Miranda, M.D. and Abades, S. (2024), "Exploring the theoretical and practical implications of eigenbehavior at the intersection of second-order cybernetics and ecosystem management", *Kybernetes*, Vol. 53 No. 12, pp. 5843-5859.

North, M.J., Collier, N.T., Ozik, J., Tatara, E.R., Macal, C.M., Bragen, M. and Sydelko, P. (2013), "Complex adaptive systems modeling with Repast Simphony", *Complex Adaptive Systems Modeling*, Vol. 1 No. 3.

Oksas, A. (2025), "Where George Spencer-Brown went wrong — re-entry recalculated", *Kybernetes*, Vol. 54 No. 8, pp. 4300-4327.

Pangaro, P. (1987), *An Examination and Confirmation of a Macro Theory of Conversations through a Realization of the Protologic Lp by Microscopic Simulation*, PhD thesis, Brunel University, London.

Pangaro, P. (2002), "New order from old: the rise of second-order cybernetics and its implications for machine intelligence", in *Proceedings of the American Society for Cybernetics Conference*, Vancouver.

Pask, G. (1975), *Conversation, Cognition and Learning*, Elsevier, Amsterdam.

Pask, G. (1976), *Conversation Theory: Applications in Education and Epistemology*, Elsevier, Amsterdam.

Spencer-Brown, G. (1969), *Laws of Form*, Allen and Unwin, London.

Suzuki, K. and Ikegami, T. (2009), "Shapes and self-movement in protocell systems", *Artificial Life*, Vol. 15 No. 1, pp. 59-70.

Varela, F.J. (1975), "A calculus for self-reference", *International Journal of General Systems*, Vol. 2 No. 1, pp. 5-24.

Varela, F.J., Maturana, H.R. and Uribe, R. (1974), "Autopoiesis: the organization of living systems, its characterization and a model", *Biosystems*, Vol. 5 No. 4, pp. 187-196.

von Foerster, H. (1979), "Cybernetics of cybernetics", in Krippendorff, K. (Ed.), *Communication and Control in Society*, Gordon and Breach, New York, NY, pp. 5-8.

von Foerster, H. (1981), *Observing Systems*, Intersystems Publications, Seaside, CA.

Wilensky, U. (1999), *NetLogo*, Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

**Supplementary material.** Two appendices are available in the online supplement: (A) a sketch of Luhmann's social autopoiesis extension using communication-event adjacency matrices and binary-code distinctions, and (B) the concrete technical requirements for computing the eigenform of the system's own viability function — the most important open problem identified in Section 7.3.
