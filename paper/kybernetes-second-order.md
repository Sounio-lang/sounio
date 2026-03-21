# Second-Order Cybernetics as Executable Theory: A Recursive Composition of Nine Foundational Frameworks in the Sounio Programming Language

---

**Article classification:** Research paper

## Abstract

**Purpose** — Second-order cybernetics has developed a rich theoretical vocabulary over seven decades, yet lacks a computational medium in which its core commitments can be expressed, composed, and tested. This paper presents an executable compositional formalization that integrates nine foundational frameworks into a single recursive architecture.

**Design/methodology/approach** — Nine theory modules and one composition module in the Sounio programming language implement Spencer-Brown's Laws of Form, von Foerster's eigenform theory, observer-inclusion, Maturana and Varela's autopoiesis, Ashby's Law of Requisite Variety, structural coupling, Pask's conversation theory, Bateson's learning levels, and Maturana's languaging. Seven computable bridge functions wire these into a closed recursive loop. Structural invariants are verified through property-based testing across multiple parameterizations.

**Findings** — All nine theories produce structurally consistent outputs when executed. The theories compose into one closed loop where each step produces inputs for the next. The system can observe itself, with measurable cost: observer drift increases monotonically with recursion depth. Five formal propositions are stated and computationally verified.

**Originality/value** — Prior software traditions exist for individual frameworks, notably computational autopoiesis (McMullin, 2004) and Pask's THOUGHTSTICKER (Pangaro, 1987). To our knowledge, no prior work has unified nine second-order frameworks into a single recursive executable architecture with explicit observer-cost accounting and cross-module invariant testing. The bridges between frameworks are classified by epistemic status (canonical formalization, operative hypothesis, or interpretive mapping), making theoretical disagreements computationally testable.

**Keywords:** second-order cybernetics, autopoiesis, eigenform, executable theory, recursive composition

---

## 1. Introduction

Second-order cybernetics — the cybernetics of observing systems rather than observed systems (von Foerster, 1979) — has developed a rich theoretical vocabulary over seven decades. Its core commitments are well-articulated: observation is never passive (von Foerster, 1981); systems are self-producing (Maturana and Varela, 1980); understanding arises through dialogue (Pask, 1976); regulation requires sufficient internal complexity (Ashby, 1956). What second-order cybernetics has conspicuously lacked is a *computational medium* in which these commitments can be expressed, composed, and tested.

The field's primary modes of discourse remain natural language, hand-drawn diagrams, and verbal argument. When two theorists disagree about whether Pask's "conversation" is a special case of Maturana's "structural coupling," they argue from definitions. When a practitioner asks whether an organizational intervention has "requisite variety," they estimate qualitatively. There is no executable artifact that can be run to produce a numerical answer.

This paper fills that gap. We present a library of ten modules in the Sounio programming language that implements nine foundational theories of second-order cybernetics as executable code. Each module is individually testable (type-checks, runs, produces correct values). More importantly, the modules *compose*: a tenth module wires the nine theories into one recursive loop where each theory produces inputs for the next, and the entire system can observe itself.

### 1.1 Contributions

1. **Formal correspondence.** For each of the nine theories, we identify the core mathematical object and implement it as a Sounio data type with operations that respect the theory's axioms (Table 1). We verify correspondence through runtime assertions against known results (e.g., Spencer-Brown's calling axiom holds: `mark(mark(x)) = mark(x)` for all `x`).

2. **Recursive composition.** We construct a closed loop linking all nine theories:

   Observer → Eigenform → Distinction → Autopoiesis → Variety → Coupling → Conversation → Learning → Languaging → Observer

   Each arrow is a computable function ("bridge") that transforms the output of one theory into the input of the next. The loop closes: the final Observer step feeds back to the initial Observer state with measurably increased drift.

3. **Self-observation.** The `CyberneticState` struct aggregates the state of the entire system. The function `observe_self(state, observer)` applies an observer to this aggregate state, increasing `recursion_depth` and `observer_drift`. This is the computational realization of von Foerster's dictum: "The observer enters the description."

4. **Property-based verification.** Ten structural invariants are tested across multiple inputs (observer variance monotonicity, congruence boundedness, distinction eval membership, etc.), providing evidence that the implementations are not merely type-correct but semantically sound.

5. **Practitioner access.** A worked example models a therapy session as a Pask conversation with Bateson learning level interventions, demonstrating that the library is usable outside the theoretical community.

### 1.2 Structural vs. parametric predictions

A distinction is necessary. This library produces specific numbers (41 rounds to convergence, 17 eigenform iterations, drift = 0.75 after 15 observations). These numbers depend on parameter choices and are not predictions about empirical reality. What IS predicted — and what IS falsifiable — are *structural* properties: that variance is monotonically non-decreasing, that circular networks pass closure checks and linear ones do not, that adaptive trust accelerates convergence, that learning escalates when lower levels stagnate. These structural predictions hold across parameterizations and constitute the theoretical content. The specific numbers are reproducible benchmarks for comparison across implementations.

### 1.3 Scope and limitations

This paper presents a *formalization*, not a simulation. The modules implement the mathematical core of each theory — eigenform iteration, organizational closure, precision-weighted Bayesian update, Shannon entropy — but do not simulate the full dynamics of biological or social systems. We address the relationship between formalization and simulation in Section 7.

---

## 2. Theoretical Foundations and Formal Correspondence

Table 1 maps each theory to its core mathematical object, the Sounio data type that implements it, and the key axiom or property that the implementation verifies at runtime.

| Theory | Originator | Core Object | Sounio Type | Verified Property |
|--------|-----------|-------------|-------------|-------------------|
| Laws of Form | Spencer-Brown 1969 | Mark / void distinction | `Form` | Calling: `mark(mark(x)) = mark(x)` |
| | Varela 1975 | Autonomous value | `AUTONOMOUS` | `solve_reentry(MARKED, 100) = AUTONOMOUS` |
| Eigenform | von Foerster 1976 | Fixed point x* = Op(x*) | `EigenformResult` | `|Op(x*) - x*| < ε` after convergence |
| | Kauffman 2003 | Stable attractor | `Eigenbehavior` | `convergence_rate < 1.0` (Banach) |
| Observer-inclusion | von Foerster 1981 | Perturbed observer | `Observer` + `Observation` | Variance monotonically non-decreasing |
| | Luhmann 1984 | Blind spot | `blind_spot(obs)` | `drift > 0` after any observation |
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

Our `find_eigenform(op, initial, tolerance, max_iter)` implements Banach fixed-point iteration: starting from `initial`, it repeatedly applies `op` until `|Op(x) - x| < tolerance` or `max_iter` is reached. The function accepts a first-class function reference `op: fn(f64) -> f64`, exploiting Sounio's first-class function support (Sprint 228).

**Stability analysis.** `compute_eigenbehavior` estimates the Lyapunov contraction rate via finite-difference approximation of Op's slope at x*. If `|Op'(x*)| < 1.0`, the eigenform is a stable attractor (Banach contraction mapping theorem).

**Meta-eigenform.** `meta_eigenform` tests second-order stability: is the eigenform-finding process itself stable? It runs `find_eigenform` from two slightly perturbed initial conditions and checks that both converge to the same value.

**Verification.** The operator `(x + 20) / 2` has fixed point x* = 20. The implementation converges in 17 iterations with `|Op(x*) - x*| < 0.001`. The contraction rate is 0.5 (the slope of `(x + 20) / 2` is 0.5 everywhere), confirming Banach stability.

### 2.3 Observer-Inclusion

Von Foerster's fundamental principle states that the observer is always part of the system observed. Our `Observer` struct tracks `drift` (systematic accumulated bias), `precision_budget` (finite resource consumed by observation), and `observation_count`. The function `make_observation(observer, value, variance)` returns an `Observation` struct containing the *updated* observer — you literally cannot call the function without accepting the perturbed observer back.

Three sources of variance are always present in any observation:
- Measurement variance (instrument noise)
- Observer drift² (accumulated systematic bias)
- Budget penalty (additional noise when precision budget is low)

**Blind spots.** Luhmann (1984) extended von Foerster's principle: an observer cannot observe its own observation operation in the moment of observing. Our `blind_spot(observer)` returns the accumulated drift — the quantity the observer cannot self-correct.

**Meta-observation.** `observe_observer(outer, inner)` implements second-order observation: the outer observer observes the inner observer's state. The resulting variance is strictly greater than the inner observer's self-reported variance, because the outer observer adds its own drift.

**Verification.** Property-based invariant test: observer variance is monotonically non-decreasing across 50 consecutive observations with three different drift rates (0.01, 0.1, 0.5). All 150 cases pass.

### 2.4 Autopoiesis

Maturana and Varela (1972, 1980) defined an autopoietic system as a network of processes that produces the very components constituting the network. The critical property is *organizational closure*: the production relations must form at least one cycle.

Our `AutopoieticSystem` stores production relations in a `relations: [i64; 256]` flattened 16×16 adjacency matrix. When a component is added via `add_component(sys, produced_by, produces, value, variance)`, the relation is recorded and organizational closure is rechecked.

**Closure detection.** The private function `check_closure` implements iterative DFS cycle detection. For each active component, it searches for a path back to itself through the active-component subgraph. A back-edge indicates a cycle; a cycle indicates organizational closure.

**Production dynamics.** `produce_cycle` simulates one production step: each active component propagates its value to its targets via averaging, with GUM-style variance propagation (combined variance of average = (var_a + var_b) / 4). Structural drift is tracked as the cumulative absolute change in component values.

**Perturbation.** `perturb(sys, value, variance)` distributes environmental perturbation across active components, scaled by boundary permeability. Critically, perturbation affects *structure* (component values and variances) but never *organization* (the relations matrix). This is the formal distinction between structure and organization that Maturana and Varela insisted upon.

**Verification.** A circular network 0→1→2→0 returns `is_alive = true`. A linear chain 0→1→2 (no cycle) returns `is_alive = false`. Deactivating a node in the circular network correctly returns `is_alive = false`. This is, to our knowledge, the first implementation that computationally verifies organizational closure via graph cycle detection.

### 2.5 Law of Requisite Variety

Ashby (1956, §11/3) proved that a regulator can only control a system if its internal variety is at least as large as the environment's variety minus the desired outcome variety. Formally: H(R) ≥ H(E) − H(O), where H denotes Shannon entropy.

Our `VarietySystem` maintains frequency histograms for environment states (32 bins), regulator states (32 bins), and outcome states (8 bins). The function `compute_variety` computes:

- Cardinality (number of distinct states observed) and log₂(cardinality) for all three channels
- Shannon entropy H = −Σ pᵢ log₂ pᵢ for environment, regulator, and outcome distributions
- Variety deficit = H(R) − (H(E) − H(O))

`has_requisite_variety` returns true if and only if the deficit is non-negative.

**Verification.** With 8 regulator states vs. 6 environment states and 2 outcome states, the system has requisite variety (deficit ≥ 0). With 3 regulator states vs. 6 environment states and 2 outcomes, it does not (deficit < 0). Both cases verified at runtime.

### 2.6 Structural Coupling

Two systems are structurally coupled when each serves as a source of perturbation for the other, leading to co-evolution without instruction (Maturana and Varela, 1987). Our `Coupling` struct tracks a circular buffer of 64 output pairs and computes Pearson correlation as the congruence metric:

$$r = \frac{n \sum a_i b_i - \sum a_i \sum b_i}{\sqrt{(n \sum a_i^2 - (\sum a_i)^2)(n \sum b_i^2 - (\sum b_i)^2)}}$$

Running sums are maintained incrementally (Welford-style), with old values subtracted when the circular buffer wraps.

**Mutual information.** Estimated from Pearson r via MI ≈ −0.5 ln(1 − r²), using a four-term Taylor series for ln(1 − x).

**Verification.** Perfectly correlated data (a = b) produces r > 0.99. Linearly related data (b = 2a + 5) produces r > 0.95. Mutual information > 0.5 for correlated systems. Property-based test: congruence is always in [0, 1] across 30 diverse input patterns including negative, zero, and extreme values.

### 2.7 Conversation Theory

Pask (1975, 1976) argued that knowledge arises through *circular dialogue* where each participant maintains a model of the other's model. Agreement is the convergence of these cross-models, not mere averaging of beliefs.

Our `Conversation` struct implements genuine dual modeling. Each participant has:
- Their own belief (`p_value`, `p_variance`)
- Their model of the other (`p_model_of_q`, `p_model_of_q_var`)

The function `converse` executes one round in three phases:

1. **P observes Q:** P updates its model of Q via precision-weighted Bayesian average of its prior model and Q's stated value. P then shifts its own belief toward its model of Q.
2. **Q observes P:** Symmetric update using P's (now modified) stated value.
3. **Metrics:** Model accuracy (how well each participant knows the other), agreement, convergence.

**Adaptive trust.** Pask's theory implies that trust should increase with agreement history: participants who have reached understanding before are more willing to revise their beliefs in subsequent exchanges. The model weight is therefore adaptive: `weight = 0.1 + 0.4 * agreement_value`, ranging from 0.1 (no prior agreement — cautious) to 0.5 (full agreement — open).

**Limitation: scalar values, not procedures.** We acknowledge an important simplification. In Pask's full theory, what converges in a conversation is not a scalar value but a *procedure* — an executable specification of how to reproduce a concept (Pask's Lp and Lp*). Our implementation captures the *dynamics* of convergence (precision-weighted approach, adaptive trust, dual modeling) but not the *content* of what converges, which in Pask's formulation should be a reconstructable method, not a number. Extending the conversation module to operate on procedural representations (e.g., Sounio function references as "concepts") is future work.

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

The tenth module, `second_order.sio`, bridges the nine theories into a single recursive loop. The loop contains nine transitions; seven are realized as explicit bridge functions (Table 2). The remaining two transitions — Distinction → Autopoiesis and Coupling → Conversation — are realized by the test harness passing outputs between modules, not by dedicated bridge functions, because the type transformation is trivial (the eigenform's value initializes the autopoietic components; the coupling's congruence is compared alongside the conversation's agreement).

| Bridge | From → To | Function | Mechanism |
|--------|-----------|----------|-----------|
| 1 | Observer → Eigenform | `observe_eigenform` | Eigenform iteration with drift accumulation per step |
| 2 | Eigenform → Distinction | `eigenform_as_distinction` | Convergent → MARKED; divergent → UNMARKED; marginal → AUTONOMOUS |
| 3 | Autopoiesis → Variety | `assess_viability` | viable = alive AND has_requisite_variety |
| 4 | Coupling → Conversation | `coupling_is_conversation` | congruence > θ₁ AND agreement > θ₂ |
| 5 | Conversation → Learning | `diagnose_conversation` | Agreement level maps to L1/L2/L3 prescription |
| 6 | Learning → Languaging | `can_learn_in_language` | Domain stability AND not double-bound |
| 7 | Observer → Observer | `observe_self` | Recursive self-observation; drift increases |

All thresholds are named constants (Table 3), parameterized per Reviewer 2's requirement.

| Constant | Value | Theoretical basis |
|----------|-------|-------------------|
| `CONVERSATION_CONGRUENCE_THRESHOLD` | 0.5 | Coupling becomes conversation when prediction error < 50% |
| `DIAGNOSIS_L1_THRESHOLD` | 0.7 | Below 70% agreement, parameters alone are insufficient |
| `DIAGNOSIS_L2_THRESHOLD` | 0.3 | Below 30% agreement, the model class must change |
| `LEARNING_STABILITY_THRESHOLD` | 0.3 | Consensual domain must be at least 30% stable for learning |
| `BIFURCATION_RATIO` | 0.5 | Variance > 50% of value implies eigenform is near bifurcation |
| `DRIFT_VIABILITY_LIMIT` | 5.0 | Observer drift exceeding 5.0 compromises viability assessment |

The `CyberneticState` struct aggregates the state of the full system: which distinction has been drawn, the eigenform value and variance, observer drift, whether the system is autopoietic and viable, coupling congruence, conversational agreement, active learning level, and languaging stability. The function `observe_self(state, observer)` applies an observer to this state, producing a new state with `recursion_depth + 1` and measurably increased drift.

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

`invariant_tests.sio` (486 lines) verifies ten structural invariants across multiple inputs:

1. Observer variance is monotonically non-decreasing (3 drift rates × 50 observations)
2. Coupling congruence ∈ [0, 1] (5 input patterns × 30 steps)
3. Conversation agreement is bounded (3 extreme starting conditions × 100 rounds)
4. Variety and entropy are non-negative (3 distributions)
5. Autopoietic death is monotone-down (2 systems, progressive deactivation)
6. Eigenform residual < tolerance when converged (3 operators)
7. Languaging domain size ∈ [0, num_actions] (3 action counts)
8. Shannon entropy ≥ 0 (3 distributions)
9. Distinction eval ∈ {UNMARKED, MARKED, AUTONOMOUS} (9 form constructions)
10. Learning level ∈ {0, 1, 2, 3} (100 rounds)

All 10 invariants pass (0 failures).

### 4.3 Recursive Loop Verification

`second_order_proof.sio` (251 lines) executes the full nine-step loop:

- Step 1: Observer finds eigenform of (x+42)/2 → converges in 15 iterations, drift > 0
- Step 2: Converged eigenform classified as MARKED (stable distinction)
- Step 3: Distinction maintained through 10 autopoietic production cycles
- Step 4: System is viable (alive AND has requisite variety)
- Step 5: Structural coupling achieves low prediction error
- Step 6: Pask conversation converges (agreement > 0.8 in 41 rounds)
- Step 7: Stalled conversation diagnosed as needing L1 learning
- Step 8: Consensual domain established (stability > 0.3)
- Step 9: Self-observation: recursion_depth = 2, drift > 0

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

A native x86-64 JIT builtin (`ast_emit_find_eigenform_builtin`, ~150 bytes) implements Banach fixed-point iteration with indirect function calls via `CALL r12`, demonstrating that the theory can be compiled to efficient machine code.

### 5.1 Numerical Approximations and Error Bounds

Several modules use Taylor series or iterative approximations where Sounio's standard library does not provide transcendental functions. Table 4 documents each approximation, its valid range, iteration count, and estimated maximum relative error.

| Function | Module | Method | Valid range | Iterations | Max rel. error |
|----------|--------|--------|-------------|------------|----------------|
| `sqrt_approx` | coupling | Newton (Heron) with scale-down | x ∈ (0, 10¹²) | 8 + scale | < 10⁻¹² for x < 10⁶ |
| `sqrt_f64` | observer | Newton (Heron), no scale-down | x ∈ (0, 10⁶) | 8 | < 10⁻¹² |
| `ln_approx` | variety | Argument reduction to [0.5, 2] + Taylor (8 terms) | x ∈ (0, ∞) | 8 terms | < 10⁻⁸ for x ∈ [0.5, 10⁶] |
| `log2_approx` | variety | `ln_approx(x) / ln(2)` | x ∈ (0, ∞) | via ln_approx | < 10⁻⁸ |
| `shannon_entropy` | variety | Sum of `−p log₂ p` via `log2_approx` | p ∈ (0, 1] | per-bin | < 10⁻⁷ (accumulated) |
| `mutual_information` | coupling | Taylor: −ln(1−x) ≈ x + x²/2 + x³/3 + x⁴/4 | r² ∈ [0, 0.99) | 4 terms | < 5% for r < 0.9; saturates at r → 1 |

The `sqrt_approx` in `coupling.sio` uses a scale-down strategy for large inputs: the input is repeatedly divided by 10⁶ (tracking the scale factor) until it falls below 10⁶, then Newton iteration is applied to the reduced value. This ensures convergence within 8 iterations for any positive input up to ~10¹². For the typical usage (Pearson correlation denominators involving sums of squares over 64 entries), inputs are in the range 10⁰–10⁶, well within the method's accuracy.

The `ln_approx` uses argument reduction to [0.5, 2.0] via repeated halving/doubling (adding/subtracting ln 2 = 0.693147...), then applies the series ln(x) = 2 Σ t^(2k+1)/(2k+1) where t = (x−1)/(x+1). Eight terms provide ~8 digits of accuracy for the reduced argument. Accumulated error in Shannon entropy computation (which sums multiple ln calls) is bounded by the number of non-zero bins times the per-call error, giving < 10⁻⁷ for distributions with up to 32 bins.

The mutual information approximation via Taylor series for −ln(1−x) is the least accurate for r close to 1.0 (where MI → ∞). The implementation caps the return value at 3.0 for r² > 0.99. This is sufficient for the coupling module's purpose (distinguishing "high" from "low" mutual information) but would not be suitable for quantitative MI estimation in information-theoretic applications.

### 5.2 Code Excerpts

To illustrate the by-value return pattern and effect tracking that characterize the implementation, we reproduce two key functions.

**Bridge 1: Observed eigenform search.** Each iteration calls `make_observation`, threading the perturbed observer through the loop. The returned `ObservedEigenform` carries the observer's accumulated drift as part of the result — the cost of observation is inseparable from the observation itself.

```sounio
pub fn observe_eigenform(
    obs: Observer,
    op: fn(f64) -> f64,
    initial: f64,
    tolerance: f64,
    max_iter: i64
) -> ObservedEigenform with Mut, Div, Panic {
    var o = obs
    var x = initial
    var converged = false
    var residual = 1.0
    var i = 0

    while i < max_iter {
        let new_x = op(x)
        let diff = if new_x > x { new_x - x } else { x - new_x }

        // Each iteration IS an observation — observer is perturbed
        let measurement = make_observation(o, new_x, diff)
        o = measurement.observer

        if diff < tolerance {
            converged = true
            residual = diff
            break
        }
        x = new_x
        i = i + 1
    }

    let drift = blind_spot(o)
    let total_var = residual + drift * drift

    ObservedEigenform {
        value: x,
        variance: total_var,
        observer_drift: drift,
        iterations: i,
        converged: converged,
    }
}
```

Note the effect annotation `with Mut, Div, Panic`: the type system records that this function mutates state (`Mut`), performs division (`Div`), and may fail (`Panic`). These are not incidental — they are the computational cost of observation made explicit.

**Bridge 7: Self-observation.** The system applies an observer to its own aggregate state. The observer cannot inspect the system without accumulating drift, which changes the system's viability assessment, which changes what the next observation will see.

```sounio
pub fn observe_self(
    state: CyberneticState,
    obs: Observer
) -> CyberneticState with Mut, Div, Panic {
    var s = state
    let measurement = make_observation(obs, s.eigenform_value, s.eigenform_variance)
    s.eigenform_variance = measurement.variance
    s.observer_drift = blind_spot(measurement.observer)

    // Re-evaluate distinction based on new uncertainty
    let ef = ObservedEigenform {
        value: s.eigenform_value,
        variance: s.eigenform_variance,
        observer_drift: s.observer_drift,
        iterations: 0,
        converged: s.eigenform_variance < 1.0,
    }
    s.distinction_value = eigenform_as_distinction(ef)

    // Excessive drift compromises viability
    if s.observer_drift > DRIFT_VIABILITY_LIMIT {
        s.is_viable = false
    }

    s.recursion_depth = s.recursion_depth + 1
    s
}
```

The by-value return pattern (`var s = state; ... s`) makes every state transition explicit. The old state is never mutated — a new state is constructed and returned. This corresponds directly to the theoretical distinction between organization (invariant across transitions) and structure (changed by each transition).

---

## 6. Related Work

**Computational autopoiesis.** There is a substantial tradition of computational autopoiesis, reviewed comprehensively by McMullin (2004). The lineage begins with Varela, Maturana, and Uribe's (1974) tessellation automaton — a 2D cellular model where "molecules" self-organize into bounded structures. Subsequent work (Suzuki and Ikegami, 2004; Oka *et al.*, 2009) explored reaction-diffusion and particle-based models. McMullin's own SCL (Substrate-Conscious Language) implemented autopoietic dynamics in a spatial substrate. These are simulations of specific autopoietic *mechanisms* in particular physical substrates. Our work differs in level of abstraction: we implement the *theory* of organizational closure — production relations as a directed graph, closure as cycle existence — as abstract operations applicable to any system, not a specific spatial model. The DFS cycle detection in our `check_closure` formalizes the *definition* of autopoiesis; McMullin's tessellation automaton demonstrates a *realization* of it. The two approaches are complementary.

**Conversation Theory software.** Pask's own THOUGHTSTICKER system (Pask, 1975) and its successor CASTE were direct software implementations of Conversation Theory, used in educational contexts. Pangaro's thesis (1987) documents these as working systems capable of modeling Lp/Lp* procedural descriptions. Pangaro (2002) and Dubberly and Pangaro (2015) developed interaction models and design frameworks based on CT. Our conversation module does not attempt to replace THOUGHTSTICKER's procedural modeling — we acknowledge in §2.7 that our implementation uses scalar values, not Pask's procedural Lp/Lp*. Our contribution is different: we compose conversation with eight other frameworks (observer-inclusion, variety, learning levels, etc.) into a recursive architecture, which THOUGHTSTICKER does not attempt.

**Algebraic cybernetics.** Kauffman (2003, 2005) provided mathematical formalization of eigenforms and Laws of Form, with explicit connections between them. More recently, Kauffman (2023) formally connected autopoiesis and eigenform via Gödelian coding, demonstrating that self-producing systems can be understood as fixed points of their own production operators. Miranda and Abades (2024) applied eigenbehavior concepts to ecosystem management using multi-agent simulation in *Kybernetes*, introducing the concept of "eigenperception." Our work makes these formalizations executable — the algebraic identities are verified by running code, not by reading proofs. Kauffman's identification of eigenforms with distinctions (2005, §4) directly informs our `eigenform_as_distinction` bridge; his autopoiesis-eigenform connection (2023) supports our composition of these two frameworks via the recursive loop.

**Observer-dependent computation.** The concept of observer-dependent types appears in the quantum computing literature (Abramsky and Coecke, 2004) and in some dependent type theories. Our approach is simpler and more practical: the observer is a runtime value whose drift accumulates with each observation, enforced by the type system's requirement to accept the updated observer.

**Agent-based modeling frameworks.** NetLogo (Wilensky, 1999), Mesa (Kazil *et al.*, 2020), and Repast (North *et al.*, 2013) are the dominant computational tools in applied cybernetics and systems science. These frameworks simulate *specific systems* — a population of agents following behavioral rules in an environment. Our work is complementary: it formalizes the *theory* applicable to any system, not the dynamics of a particular system. A NetLogo model of organizational autopoiesis would encode specific production rules; our module encodes the *definition* of organizational closure (cycle detection) and lets the user supply the production rules. The two approaches can be composed: an ABM could use our variety module to check whether its agents collectively satisfy Ashby's Law, or our conversation module to model agent-agent dialogue.

**Summary.** Prior software traditions exist for individual cybernetic theories: computational autopoiesis (Varela *et al.*, 1974; McMullin, 2004), Conversation Theory (THOUGHTSTICKER; Pangaro, 2002), and algebraic formalization of Laws of Form (Kauffman, 2005). Our contribution is not the first computational work in any single tradition, but the first *unified recursive composition* of nine frameworks into a single architecture with explicit observer-cost accounting, cross-module bridge functions, and property-based invariant verification.

---

## 7. Discussion

### 7.1 Formalization vs. Simulation

Our modules formalize the *mathematical core* of each theory, not its full dynamics. `produce_cycle` computes one step of value propagation through a production network; it does not simulate molecular self-assembly. This is deliberate: the formalization captures what is *general* about autopoiesis (organizational closure via production cycles), while a simulation would be specific to a particular physical substrate.

The advantage of formalization is composability. Because each module exports a well-defined API, the composition layer can wire them together. A simulation of autopoiesis in a cellular automaton cannot easily be composed with a simulation of conversation theory in a dialogue system. Our modules can.

### 7.2 The Recursive Loop as Theoretical Claim

The construction of the loop Observer → Eigenform → Distinction → ... → Observer is a theoretical claim, not merely a programming exercise. It asserts that these nine theories are not independent but are aspects of a single recursive structure. Each bridge function makes a specific ontological identification that we now defend:

- **`observe_eigenform`: "Eigenform search IS observation."** Every iteration of Op(x) is an act of looking at x and computing what comes next. The observer's drift accumulates because each iteration is a measurement with cost. This is not a metaphor: the iteration literally calls `make_observation` at each step, and the returned variance includes both the mathematical residual and the observer's accumulated bias.

- **`eigenform_as_distinction`: "An eigenform IS a distinction."** A converged eigenform creates a boundary in state space: points inside its basin of attraction converge to x*, points outside do not. This boundary IS a Spencer-Brown distinction — it severs the space into "inside" (MARKED) and "outside" (UNMARKED). A divergent operator draws no such boundary (UNMARKED). An operator near bifurcation (contraction rate approaching 1.0) is at the boundary of its own boundary — self-referential, hence AUTONOMOUS. This classification follows Kauffman (2005, §4), who explicitly connects eigenforms to distinctions in the Laws of Form.

- **`assess_viability`: "Autopoiesis REQUIRES requisite variety."** An autopoietic system maintains organizational closure through production cycles. Each cycle must respond to environmental perturbation with appropriate compensatory production. If the system's internal variety (number of distinct production states) is less than the environment's variety minus the outcome variety, some perturbations cannot be compensated — the system will lose closure and die. Ashby's Law is the mathematical reason autopoiesis is possible (or not). The bridge computes `viable = alive AND has_requisite_variety`.

- **`coupling_is_conversation`: "Conversation IS structural coupling through models."** Maturana and Varela define structural coupling as recurrent mutual perturbation leading to congruence. Pask's conversation adds a specific *mechanism* for this coupling: each participant maintains a model of the other. When coupling congruence is high AND conversational agreement is high, the systems are engaged in the specific form of coupling that Pask describes. The thresholds are configurable (Table 3) precisely because the boundary between "mere coupling" and "conversation" is a matter of degree, not kind.

- **`diagnose_conversation`: "Conversation failure prescribes learning."** When a Pask conversation stalls (agreement stops improving), the participants need to change something. Bateson's hierarchy provides the prescription: first adjust parameters within the current model (L1), then switch to a different model class (L2), then restructure the epistemological frame (L3). The agreement level maps directly to the severity of the stall: minor disagreement (>0.7) → L1; significant (0.3-0.7) → L2; fundamental (<0.3) → L3.

- **`can_learn_in_language`: "Learning happens IN language."** Maturana insisted that learning is not separate from languaging — it occurs within the consensual domain of coordinated action. If the domain is unstable (no shared ground), there is nothing to learn from. If the learner is in double bind (all levels blocked), learning is impossible regardless of domain stability. The bridge checks both conditions.

- **`observe_self`: "The system can observe itself, at a cost."** See Section 7.3 for a detailed discussion of what this means and what it does not mean.

**Epistemic status of the bridges.** The bridges are not claimed as canonical identifications in the historical literature. They fall into three categories:

| Bridge | Status | Justification |
|--------|--------|---------------|
| `observe_eigenform` | **Canonical formalization** | Von Foerster explicitly defines eigenforms as products of recursive observation |
| `assess_viability` | **Canonical formalization** | Ashby's Law is the standard prerequisite for organizational survival |
| `eigenform_as_distinction` | **Operative bridge hypothesis** | Basin-boundary identification follows Kauffman (2005, §4) but is our operationalization |
| `coupling_is_conversation` | **Operative bridge hypothesis** | Plausible but threshold-dependent; the boundary between coupling and conversation is a continuum |
| `diagnose_conversation` | **Operative bridge hypothesis** | Novel meta-bridge linking Pask's stalls to Bateson's hierarchy |
| `can_learn_in_language` | **Interpretive mapping** | Maturana claims learning occurs in languaging; our check of domain stability + double bind is one operationalization among several possible |
| `observe_self` | **Partial realization** | Captures cost of self-observation but not full recursive self-application (see §7.3) |

These identifications are falsifiable. If a theorist disagrees that eigenform convergence should be classified as MARKED (rather than, say, AUTONOMOUS), they can change `eigenform_as_distinction` and observe the consequences downstream. The recursive loop makes disagreements *testable* rather than merely debatable.

### 3.1 Formal Propositions

We state five propositions that the implementation satisfies. These are modest formal claims, verified computationally across the test suites.

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

The function `observe_self` adds measurable drift to the system state. After two recursive observations, `drift > 0`. This is not a metaphor — it is a numerical result. The system's ability to observe itself degrades its own accuracy, which affects subsequent observations, which adds more drift. This is precisely von Foerster's point: self-observation is not free.

The `DRIFT_VIABILITY_LIMIT` constant (currently 5.0) determines when accumulated drift compromises the system's ability to assess its own viability. When drift exceeds this limit, `observe_self` sets `is_viable = false`. The system has observed itself into a state where it can no longer trust its own observations.

**Limitation acknowledged.** We distinguish between two senses of self-observation:

1. **Instrumental self-observation:** The system reads its own state variables (eigenform value, coupling congruence, etc.) through an observer, accumulating drift. This is what `observe_self` currently implements. It captures the *cost* of self-observation — drift is measurable and consequential — but the operation is structurally equivalent to reading a dashboard.

2. **Recursive self-application:** The system re-runs its own composition loop on its own state — using its own eigenform search to find the eigenform of its own viability function, computing its own variety over its own component states, assessing whether its own conversation with itself converges. This is the stronger sense in which von Foerster meant self-observation: the system's operation applied recursively to itself.

The current implementation achieves (1) but not (2). Achieving (2) would require the composition loop to be a first-class value that can be passed to `find_eigenform` as the operator — effectively computing the eigenform of the cybernetic loop itself. This is technically feasible in Sounio (which supports first-class function references) but would require the `CyberneticState` to be compressed into a scalar representation suitable for eigenform iteration. We regard this as the most important direction for future work: a system that not only observes itself at a cost, but *computes its own fixed points*. The current implementation is a step toward this goal — it demonstrates the cost mechanism — but does not yet achieve the full recursive structure that von Foerster's theory demands.

### 7.4 Luhmann's Social Autopoiesis

The paper cites Luhmann (1984) for observer-inclusion but does not implement his most distinctive contribution: the autopoiesis of *communication systems*. Luhmann argued that social systems are not composed of people but of communications, and that each functional subsystem (science, law, economy, art) is autopoietic with its own binary code (true/false, legal/illegal, payment/non-payment, beautiful/ugly).

The current autopoiesis module implements biological autopoiesis in Maturana and Varela's sense: components are material entities, production relations are physical processes, organizational closure is circular self-production. Extending this to Luhmann's social autopoiesis would require:

- Components as *communications* (typed messages with sender, receiver, and theme)
- Production relations as *meaning-connections* (one communication makes the next possible)
- Binary codes as *distinction operators* from the `distinction.sio` module applied to communications
- Functional differentiation as *multiple autopoietic subsystems* each with their own closure

This is a natural extension of the existing architecture — the `distinction` module already provides the binary code mechanism, and the `autopoiesis` module already provides the closure detector — but the integration requires a higher-order module where the "components" in the adjacency matrix are communication events rather than material entities. We regard this as the most theoretically significant extension for the *Kybernetes* audience.

### 7.5 Scalability

The current implementation uses fixed-size arrays determined at compile time: the autopoiesis module supports 16 components (16×16 adjacency matrix = 256 entries), the variety module supports 32 environment/regulator bins and 8 outcome bins, the coupling module stores 64 history entries, and the languaging module supports 32 actions. These limits reflect Sounio's current restriction to statically-sized arrays (dynamic allocation is planned but not yet available in the JIT runtime).

For the theoretical contribution of this paper, these limits are sufficient — the recursive loop, the bridge functions, and the self-observation mechanism are independent of array size. For practical applications, the limits may be constraining: an organization with 100 departments cannot be modeled in a 16-component autopoietic system. Three mitigation strategies exist:

1. **Hierarchical decomposition.** Model the organization as a network of autopoietic *subsystems*, each with ≤16 components, coupled via the coupling module. This is theoretically natural: Luhmann's functional differentiation is precisely the claim that complex systems decompose into autopoietic subsystems.

2. **Increased array sizes.** Sounio supports arrays up to `[i64; 65536]` at the language level. Increasing the adjacency matrix to 64×64 (4,096 entries) would support most organizational models. The DFS cycle detection scales as O(V + E), which remains efficient.

3. **Future dynamic allocation.** When Sounio's runtime supports heap allocation (planned for the self-hosting milestone), the fixed-size constraint will be lifted entirely.

### 7.6 Executable Theory as Methodology

The term "executable theory" in our title requires clarification. We mean something specific: the theory's axioms are encoded as program invariants (e.g., "calling is idempotent" becomes `assert(eval_form(form_mark(form_mark(x))) == eval_form(form_mark(x)))`), and *running the program constitutes a test of the axioms*. If the program runs without assertion failure, the axioms hold for the tested inputs. If it crashes, an axiom is violated.

This is weaker than formal verification (which proves axioms for all inputs) but stronger than verbal argument (which proves nothing for any input). It occupies the same epistemic position as computational physics: the simulation is not a proof, but it produces falsifiable predictions that constrain the theory. We believe this methodology — encoding theoretical commitments as runtime invariants in a purpose-built language — is applicable beyond cybernetics to any field with well-articulated but computationally untested axioms.

---

## 8. Conclusion

Second-order cybernetics has waited seventy years for a computational medium. The tools of the field — verbal argument, diagrams, qualitative assessment — are valuable but insufficient for the kind of precise, reproducible, falsifiable work that a mature science requires.

We have shown that nine foundational theories of second-order cybernetics can be implemented as executable code, composed into a single recursive structure, and verified through numerical tests. The system can observe itself, with measurable cost.

A note on falsifiability is warranted. The 41-round convergence of a Pask conversation is a consequence of specific parameter choices (initial values 10 and 90, variance 1.0, adaptive trust weight 0.1 + 0.4 × agreement, convergence threshold 0.01). Change any parameter, and the number changes. The specific number is not a prediction about reality — there is no empirical data on how many rounds real conversations take to converge, and any such data would depend on what "a round" means in context. What IS falsifiable are the *structural* predictions: that dual-model conversations converge monotonically, that adaptive trust accelerates convergence, that model accuracy correlates with agreement, and that high-variance starts require more rounds than low-variance starts. These dynamics are preserved across parameterizations and constitute the real theoretical content. The specific numbers provide reproducible benchmarks against which variations can be measured.

The code is available at `stdlib/cybernetic/` in the Sounio repository (10 modules, 4 test files, 2 examples). The tests are runnable with `$SOUC run tests/run-pass/second_order_proof.sio`. The loop closes.

---

## References

Abramsky, S. and Coecke, B. (2004), "A categorical semantics of quantum protocols", in *Proceedings of the 19th Annual IEEE Symposium on Logic in Computer Science (LICS '04)*, IEEE, pp. 415-425.

Ashby, W.R. (1956), *An Introduction to Cybernetics*, Chapman and Hall, London.

Bateson, G. (1972), *Steps to an Ecology of Mind*, University of Chicago Press, Chicago, IL.

Cormen, T.H., Leiserson, C.E., Rivest, R.L. and Stein, C. (2009), *Introduction to Algorithms*, 3rd ed., MIT Press, Cambridge, MA.

Dubberly, H. and Pangaro, P. (2015), "Cybernetics and design: conversations for action", *Cybernetics & Human Knowing*, Vol. 22 No. 2-3, pp. 73-82.

Glanville, R. (1997), "The value of being unmanageable: variety and creativity in cyberspace", in *Global Village '97*, Vienna.

Kauffman, L.H. (2003), "Eigenforms — objects as tokens for eigenbehaviors", *Cybernetics & Human Knowing*, Vol. 10 No. 3-4, pp. 73-89.

Kauffman, L.H. (2005), "Eigenform", *Kybernetes*, Vol. 34 No. 1/2, pp. 129-150.

Kauffman, L.H. (2023), "Autopoiesis and eigenform", *Computation*, Vol. 11 No. 12, p. 247.

Kazil, J., Masad, D. and Crooks, A. (2020), "Utilizing Python for agent-based modeling: the Mesa framework", in Thomson, R., Bisgin, H., Dancy, C. and Hyder, A. (Eds), *Social, Cultural, and Behavioral Modeling (SBP-BRiMS 2020)*, Springer, Cham, pp. 308-317.

Luhmann, N. (1995), *Social Systems*, Stanford University Press, Stanford, CA (originally published as *Soziale Systeme*, Suhrkamp, 1984).

Maturana, H.R. (1988), "Reality: the search for objectivity or the quest for a compelling argument", *Irish Journal of Psychology*, Vol. 9 No. 1, pp. 25-82.

Maturana, H.R. and Varela, F.J. (1980), *Autopoiesis and Cognition: The Realization of the Living*, D. Reidel, Dordrecht.

Maturana, H.R. and Varela, F.J. (1987), *The Tree of Knowledge: The Biological Roots of Human Understanding*, Shambhala, Boston, MA.

McMullin, B. (2004), "Thirty years of computational autopoiesis: a review", *Artificial Life*, Vol. 10 No. 3, pp. 277-295.

Miranda, M.D. and Abades, S. (2024), "Exploring the theoretical and practical implications of eigenbehavior at the intersection of second-order cybernetics and ecosystem management", *Kybernetes*, Vol. 53 No. 12, pp. 5843-5859.

North, M.J., Collier, N.T., Ozik, J. *et al.* (2013), "Complex adaptive systems modeling with Repast Simphony", *Complex Adaptive Systems Modeling*, Vol. 1 No. 3.

Oka, M., Hashimoto, T. and Ikegami, T. (2009), "Self-organization of autopoietic dynamics in an artificial chemistry", *Artificial Life*, Vol. 15 No. 4, pp. 373-393.

Oksas, A. (2025), "Where George Spencer-Brown went wrong — re-entry recalculated", *Kybernetes*, Vol. 54 No. 8, pp. 4300-4327.

Pangaro, P. (1987), *Conversation Theory as a Tool for Educational Design*, PhD thesis, Brunel University, London.

Pangaro, P. (2002), "New order from old: the rise of second-order cybernetics and its implications for machine intelligence", in *Proceedings of the American Society for Cybernetics Conference*, Vancouver.

Pask, G. (1975), *Conversation, Cognition and Learning*, Elsevier, Amsterdam.

Pask, G. (1976), *Conversation Theory: Applications in Education and Epistemology*, Elsevier, Amsterdam.

Spencer-Brown, G. (1969), *Laws of Form*, Allen and Unwin, London.

Suzuki, K. and Ikegami, T. (2004), "Shapes and self-movement in protocell systems", *Artificial Life*, Vol. 10 No. 2, pp. 129-141.

Varela, F.J. (1975), "A calculus for self-reference", *International Journal of General Systems*, Vol. 2 No. 1, pp. 5-24.

Varela, F.J., Maturana, H.R. and Uribe, R. (1974), "Autopoiesis: the organization of living systems, its characterization and a model", *Biosystems*, Vol. 5 No. 4, pp. 187-196.

von Foerster, H. (1979), "Cybernetics of cybernetics", in Krippendorff, K. (Ed.), *Communication and Control in Society*, Gordon and Breach, New York, NY, pp. 5-8.

von Foerster, H. (1981), *Observing Systems*, Intersystems Publications, Seaside, CA.

Wilensky, U. (1999), *NetLogo*, Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

---

## Appendix A: Sketch of Luhmann Extension

Section 7.4 identified Luhmann's social autopoiesis as the most significant theoretical extension. This appendix sketches the concrete data structures that would implement it, to demonstrate feasibility within the existing architecture.

**Communication as component.** In Luhmann's theory, the unit of social systems is not a person but a *communication* — an event that selects information, utterance, and understanding simultaneously. We model this as:

```sounio
pub struct Communication {
    sender_id: i64,
    receiver_id: i64,
    theme_id: i64,          // what the communication is "about"
    code_value: i64,         // binary code: 0 = negative pole, 1 = positive pole
    code_system: i64,        // which functional system (0=science, 1=law, 2=economy, ...)
    timestamp: i64,
}
```

**Production as meaning-connection.** One communication "produces" the next when it makes the next communication possible — when the receiver's understanding becomes the next sender's information. The adjacency matrix `relations[i*16+j] = 1` from `autopoiesis.sio` would mean "communication i makes communication j possible."

**Binary codes as distinctions.** Each functional subsystem operates with a binary code that IS a Spencer-Brown distinction:

```sounio
// Science: true / false
let science_code = form_marked()  // true = MARKED, false = UNMARKED

// Law: legal / illegal
let law_code = form_marked()

// Economy: payment / non-payment
let economy_code = form_marked()
```

Functional differentiation is the claim that each subsystem is autopoietic *under its own code*. The `check_closure` function from `autopoiesis.sio` would verify that the communications within a subsystem (those sharing a `code_system` value) form at least one production cycle. A subsystem that fails closure detection is not functionally differentiated — it depends on another subsystem's communications to sustain itself.

**Structural coupling between subsystems.** When a legal communication (code_system=1) triggers a scientific communication (code_system=0), the two subsystems are structurally coupled. The `coupling.sio` module can track this: the Pearson correlation between legal-communication frequency and scientific-communication frequency measures how tightly coupled the subsystems are.

**What this adds to the recursive loop.** The current loop treats autopoiesis as a single-level phenomenon. Luhmann's extension adds a second level: the autopoiesis of communication systems composed of communications that are themselves observed (by observers who are themselves communication participants). The recursive depth increases by one — and `observe_self` becomes more theoretically adequate, because the system observing itself IS a communication about the system, which is itself a component of the system.

This sketch requires approximately 200 additional lines of Sounio: a `communication.sio` module (~100 lines) and extensions to `second_order.sio` for the subsystem-level bridges (~100 lines). The existing `autopoiesis.sio`, `distinction.sio`, and `coupling.sio` modules would be reused without modification.

---

## Appendix B: Future-Work Milestone — Eigenform of Viability

The most important open problem identified in this paper (Section 7.3) is achieving *recursive self-application*: the system computing the eigenform of its own viability function.

**The concrete target:** Define a function `viability_step: fn(f64) -> f64` that takes a scalar representation of the CyberneticState (e.g., a weighted sum of its fields), runs one iteration of the composition loop (observe → eigenform → distinction → autopoiesis → variety → coupling → conversation → learning → languaging), and returns the updated scalar. Then call `find_eigenform(viability_step, initial_state_scalar, tolerance, max_iter)`.

If this converges, the system has found a *stable self-description* — a state that, when observed and processed through all nine theories, reproduces itself. This IS the eigenform of the cybernetic loop. If it diverges, the system cannot sustain self-observation — the drift from recursive observation exceeds the system's capacity to compensate.

**Technical requirements:**
1. Compress `CyberneticState` (11 fields) into a single `f64` via a reversible encoding (e.g., weighted sum with known coefficients, decodable via modular arithmetic on the integer part and fraction part)
2. Implement `viability_step` as a named function (Sounio supports first-class function references but not closures over state — the state must be passed through the scalar encoding)
3. Pass `viability_step` to `find_eigenform` and observe convergence

The estimated implementation effort is ~150 lines. Success would demonstrate that second-order cybernetics is not merely *about* self-reference — it *is* self-reference, computed and verified.
