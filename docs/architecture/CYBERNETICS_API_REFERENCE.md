<!-- docs:meta
topic_id: repo.docs.architecture.cybernetics-api-reference
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.cybernetics-api-reference
-->

# Cybernetics API Reference

Complete API reference for `stdlib/cybernetic/`. All signatures, structs, and constants
taken directly from the source `.sio` files.

Nine modules spanning the full spectrum of second-order cybernetic theory:

| Module | Theory | Source |
|--------|--------|--------|
| `distinction` | Spencer-Brown's Laws of Form + Varela's self-reference | Spencer-Brown 1969, Varela 1975 |
| `eigenform` | Fixed-point objects through recursive observation | von Foerster 1976, Kauffman 2003 |
| `observer` | Observer-inclusion and epistemic drift | von Foerster, Luhmann 1984 |
| `autopoiesis` | Self-producing organizationally closed systems | Maturana/Varela 1972 |
| `variety` | Requisite variety and Shannon entropy | Ashby 1956 |
| `bateson` | Hierarchical learning levels and double binds | Bateson 1972 |
| `languaging` | Consensual coordination of coordinated action | Maturana 1988 |
| `conversation` | Knowledge through dialogue with dual modeling | Pask 1975 |
| `coupling` | Structural coupling via recurrent mutual perturbation | Maturana/Varela |

---

## cybernetic::distinction

Spencer-Brown's Laws of Form + Varela's Calculus for Self-Reference.

The primitive act of cognition is drawing a distinction. Spencer-Brown (1969) formalized
this with two axioms (Calling and Crossing). Varela (1975) extended it with a third value
(AUTONOMOUS) that resolves self-reference without paradox.

### Constants

```sio
pub const UNMARKED: i64 = 0
```
Unmarked state -- the void, outside all distinctions.

```sio
pub const MARKED: i64 = 1
```
Marked state -- inside the distinction.

```sio
pub const AUTONOMOUS: i64 = 2
```
Autonomous state -- Varela's self-indicating form J, neither marked nor unmarked.
The fixed point of crossing. Structurally identical to `mu X. cross(X)` in type theory.

### Structs

**`Form`**

A node in the calculus of indications. Forms compose by nesting (children inside a mark).

| Field | Type | Description |
|-------|------|-------------|
| `value` | `i64` | Current value: UNMARKED, MARKED, or AUTONOMOUS |
| `children` | `[i64; 8]` | Nested sub-forms |
| `child_count` | `i64` | Number of active children |
| `depth` | `i64` | Nesting depth (number of marks crossed to reach this form) |
| `is_reentrant` | `bool` | True if this form contains itself (re-entry) |
| `period` | `i64` | Period of oscillation in imaginary time (0 = convergent, 2 = standard crossing) |

### Functions

```sio
pub fn form_void() -> Form
```
The void -- unmarked, empty, no distinction drawn.

```sio
pub fn form_marked() -> Form
```
A mark -- the simplest distinction, the MARKED state.

```sio
pub fn form_reenter() -> Form
```
The self-indicating form J where J = cross(J). Varela's autonomous value.

```sio
pub fn form_mark(f: Form) -> Form with Mut
```
Axiom 1 -- Calling (condensation). Marking something already marked is idempotent.
Autonomous forms absorb marks.

```sio
pub fn form_cross(f: Form) -> Form with Mut
```
Axiom 2 -- Crossing (cancellation). Toggles marked/unmarked. Autonomous forms
are invariant under crossing (J = cross(J)).

```sio
pub fn eval_form(f: Form) -> i64
```
Evaluate a form to its canonical value. Re-entrant forms yield AUTONOMOUS; otherwise
depth parity determines MARKED (odd) vs UNMARKED (even).

```sio
pub fn forms_equal(a: Form, b: Form) -> bool
```
Check if two forms are equivalent under the calculus (by comparing evaluated values).

```sio
pub fn form_juxtapose(a: Form, b: Form) -> i64
```
Juxtaposition (Boolean OR). AUTONOMOUS propagates. Marked if either is marked.

```sio
pub fn form_nest(outer: Form, inner: Form) -> i64 with Mut
```
Superposition -- nesting one form inside another's mark. In Boolean interpretation:
NOT-outer OR inner (implication). AUTONOMOUS propagates.

```sio
pub fn solve_reentry(initial: i64, max_iter: i64) -> i64 with Mut, Div
```
Solve f = cross(f) iteratively. If oscillation is detected, returns AUTONOMOUS.
Implements Varela's insight that self-reference produces a third value.

```sio
pub fn solve_reentry_timed(initial: i64, max_iter: i64) -> Form with Mut, Div
```
Solve f = cross(f) and compute the oscillation period. Returns a Form with the
period field populated. Standard crossing oscillation has period 2.
Implements Spencer-Brown's imaginary-time analysis (Ch. 11).

```sio
pub fn form_period(f: Form) -> i64
```
Get the period of oscillation for a form (Spencer-Brown imaginary time).

```sio
pub fn value_name(v: i64) -> i64
```
Human-readable name for a form value. Returns the value itself (caller matches on constant).

---

## cybernetic::eigenform

Eigenform Theory (von Foerster 1976, Kauffman 2003).

An eigenform is a fixed point of an operator: if Op(x*) = x*, then x* is an eigenform
of Op. This is the mathematical definition of an object as a stable pattern in observation.
Eigenforms are isomorphic to Y combinators and Scott reflexive domains in type theory.

### Structs

**`EigenformResult`**

Result of eigenform search via fixed-point iteration.

| Field | Type | Description |
|-------|------|-------------|
| `value` | `f64` | The fixed point x* where Op(x*) = x* |
| `variance` | `f64` | Variance (uncertainty) in the eigenform value |
| `iterations` | `i64` | Number of iterations until convergence |
| `converged` | `bool` | Did iteration converge within tolerance? |
| `residual` | `f64` | Final residual |Op(x) - x| |

**`Eigenbehavior`**

Stability analysis of an eigenform (Kauffman's extension).

| Field | Type | Description |
|-------|------|-------------|
| `eigenform` | `EigenformResult` | The eigenform itself |
| `convergence_rate` | `f64` | |Op'(x*)| -- slope of Op near x* |
| `stability_radius` | `f64` | How far you can perturb before losing attraction |
| `is_stable` | `bool` | Is this eigenform globally stable? (|slope| < 1) |

### Functions

```sio
pub fn find_eigenform(
    op: fn(f64) -> f64,
    initial: f64,
    tolerance: f64,
    max_iter: i64
) -> EigenformResult with Mut, Div, Panic
```
Find an eigenform (fixed point) of an operation via Banach iteration.
Repeatedly applies `op` until |Op(x) - x| < tolerance or max_iter is reached.

```sio
pub fn is_eigenform(op: fn(f64) -> f64, candidate: f64, tolerance: f64) -> bool with Div
```
Check if a value is an eigenform of the given operation (|Op(candidate) - candidate| < tolerance).

```sio
pub fn compute_eigenbehavior(
    op: fn(f64) -> f64,
    initial: f64,
    tolerance: f64,
    max_iter: i64,
    delta: i64
) -> Eigenbehavior with Mut, Div, Panic
```
Compute full eigenbehavior: eigenform + Lyapunov-style stability analysis.
Measures the slope of Op near the eigenform via finite difference.

```sio
pub fn meta_eigenform(
    op: fn(f64) -> f64,
    initial: f64,
    tolerance: f64,
    max_iter: i64
) -> EigenformResult with Mut, Div, Panic
```
Meta-eigenform: is the eigenform-finding process itself stable?
Computes eigenforms from two slightly perturbed initial values. If results converge
to the same value, the process is second-order stable.

---

## cybernetic::observer

Observer-Inclusion -- the core principle of second-order cybernetics (von Foerster, Luhmann 1984).

The observer is always part of the system being observed. Observation is never passive:
it perturbs the observer, accumulates drift, and is subject to blind spots. Every observation
returns Knowledge (never bare values).

### Structs

**`Observer`**

An observer embedded in the system it observes.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `i64` | Unique identifier |
| `state_value` | `f64` | Observer's own epistemic state value |
| `state_variance` | `f64` | Observer's own epistemic state variance |
| `observation_count` | `i64` | Number of observations performed |
| `drift` | `f64` | Accumulated observer drift (systematic bias) |
| `drift_rate` | `f64` | How much drift accumulates per observation |
| `precision_budget` | `f64` | Total precision available (graded effect) |
| `initial_budget` | `f64` | Initial precision budget (for computing exhaustion ratio) |

**`Observation`**

Result of an observation: updated observer + measurement.

| Field | Type | Description |
|-------|------|-------------|
| `observer` | `Observer` | Updated observer state (drift increased, budget consumed) |
| `value` | `f64` | Observed value |
| `variance` | `f64` | Total variance (measurement + drift + budget penalty) |

### Functions

```sio
pub fn observer_new(id: i64, drift_rate: f64) -> Observer
```
Create a new observer with given drift rate. Default precision budget: 1000.0.

```sio
pub fn observer_with_budget(id: i64, drift_rate: f64, budget: f64) -> Observer
```
Create an observer with a specific precision budget.

```sio
pub fn make_observation(
    obs: Observer,
    system_value: f64,
    measurement_variance: f64
) -> Observation with Mut, Div, Panic
```
Observe a system value. The returned variance always includes measurement noise,
observer drift, and budget penalty. Enforces von Foerster's principle: observation
is not passive.

```sio
pub fn observe_observer(outer: Observer, observed_obs: Observer) -> Observation with Mut, Div, Panic
```
Second-order observation: an observer observing another observer. Compounds
uncertainty (uncertainty about uncertainty).

```sio
pub fn blind_spot(obs: Observer) -> f64
```
The observer's blind spot: accumulated drift it cannot self-correct (Luhmann).

```sio
pub fn relative_blind_spot(obs: Observer) -> f64 with Div, Panic
```
Relative blind spot: drift squared as fraction of total uncertainty.

```sio
pub fn precision_remaining(obs: Observer) -> f64 with Div, Panic
```
Budget exhaustion ratio: how much precision remains (0.0 to 1.0).

```sio
pub fn observer_agreement(a: Observer, b: Observer) -> f64 with Div, Panic
```
Agreement between two observers: do they see the same thing?
Returns a value between 0.0 (complete disagreement) and 1.0 (perfect agreement).

```sio
pub fn informative_disagreement(a: Observer, b: Observer) -> f64 with Div, Panic
```
Informative disagreement: how much we learn from observer divergence.
Normalized by average uncertainty.

---

## cybernetic::autopoiesis

Autopoiesis -- self-producing systems (Maturana/Varela 1972).

An autopoietic system is a network of production processes that produces the very
components constituting the network. Organizational closure (circular production) is
the defining property. Production relations are stored in a 16x16 adjacency matrix.
Closure is verified via DFS cycle detection.

### Structs

**`AutopoieticSystem`**

| Field | Type | Description |
|-------|------|-------------|
| `component_count` | `i64` | Number of components in the system |
| `relations` | `[i64; 256]` | 16x16 flattened adjacency matrix: relations[i*16+j]=1 means i produces j |
| `component_values` | `[f64; 16]` | Current value of each component |
| `component_variances` | `[f64; 16]` | Variance (uncertainty) of each component |
| `component_active` | `[i64; 16]` | 1=active, 0=inactive |
| `boundary_permeability` | `f64` | How permeable the system boundary is to perturbation |
| `alive` | `bool` | Whether the system has organizational closure (is autopoietic) |
| `generation` | `i64` | Number of production cycles completed |
| `total_perturbation` | `f64` | Accumulated perturbation from environment |
| `structural_drift_val` | `f64` | Accumulated structural drift from production cycles |

### Functions

```sio
pub fn system_new(permeability: f64) -> AutopoieticSystem
```
Create a new empty autopoietic system with given boundary permeability.

```sio
pub fn add_component(
    sys: AutopoieticSystem,
    produced_by: i64,
    produces: i64,
    value: f64,
    variance: f64
) -> AutopoieticSystem with Mut, Div, Panic
```
Add a component with production relations: `produced_by -> this -> produces`.
Automatically rechecks organizational closure after adding. Max 16 components.

```sio
pub fn produce_cycle(sys: AutopoieticSystem) -> AutopoieticSystem with Mut, Div, Panic
```
Run one production cycle. Each active producer propagates values to targets.
Values average with GUM variance propagation. Tracks structural drift.

```sio
pub fn perturb(
    sys: AutopoieticSystem,
    value: f64,
    variance: f64
) -> AutopoieticSystem with Mut, Div, Panic
```
Perturb from environment. Affects structure (values) NOT organization (relations).
Perturbation is scaled by boundary permeability and distributed across active components.

```sio
pub fn deactivate_component(
    sys: AutopoieticSystem,
    id: i64
) -> AutopoieticSystem with Mut, Div, Panic
```
Deactivate a component and recheck organizational closure.

```sio
pub fn is_alive(sys: AutopoieticSystem) -> bool
```
Returns true if the system has organizational closure (is autopoietic).

```sio
pub fn generation(sys: AutopoieticSystem) -> i64
```
Returns the number of production cycles completed.

```sio
pub fn structural_drift(sys: AutopoieticSystem) -> f64
```
Returns total structural drift (production drift + perturbation accumulation).

---

## cybernetic::variety

Ashby's Law of Requisite Variety (1956).

Variety = log2(cardinality of distinct states). Shannon entropy H = -Sum p*log2(p)
for frequency distributions. Ashby's Law: reg_variety >= env_variety - outcome_variety.

### Structs

**`VarietySystem`**

| Field | Type | Description |
|-------|------|-------------|
| `env_states` | `[i64; 32]` | Environment state frequency histogram |
| `env_total` | `i64` | Total environment observations |
| `reg_states` | `[i64; 32]` | Regulator state frequency histogram |
| `reg_total` | `i64` | Total regulator observations |
| `outcome_states` | `[i64; 8]` | Outcome state frequency histogram |
| `outcome_total` | `i64` | Total outcome observations |
| `env_variety` | `f64` | log2(environment cardinality) |
| `reg_variety` | `f64` | log2(regulator cardinality) |
| `outcome_variety` | `f64` | log2(outcome cardinality) |
| `env_entropy` | `f64` | Shannon entropy of environment distribution |
| `reg_entropy` | `f64` | Shannon entropy of regulator distribution |
| `variety_deficit` | `f64` | reg_variety - (env_variety - outcome_variety); positive = sufficient |

### Functions

```sio
pub fn variety_new() -> VarietySystem
```
Create a new empty variety system with all counters zeroed.

```sio
pub fn record_env_state(vs: VarietySystem, state: i64) -> VarietySystem with Mut, Panic
```
Record an environment state observation. State is hashed into the 32-slot histogram.

```sio
pub fn record_reg_state(vs: VarietySystem, state: i64) -> VarietySystem with Mut, Panic
```
Record a regulator state observation. State is hashed into the 32-slot histogram.

```sio
pub fn record_outcome(vs: VarietySystem, state: i64) -> VarietySystem with Mut, Panic
```
Record an outcome observation. State is hashed into the 8-slot histogram.

```sio
pub fn compute_variety(vs: VarietySystem) -> VarietySystem with Mut, Div, Panic
```
Compute variety (log2 cardinality), Shannon entropy, and Ashby deficit for all
three subsystems (environment, regulator, outcome).

```sio
pub fn has_requisite_variety(vs: VarietySystem) -> bool
```
Ashby's Law: returns true iff reg_variety >= env_variety - outcome_variety.

```sio
pub fn variety_deficit(vs: VarietySystem) -> f64
```
Return the variety deficit. Positive means sufficient variety; negative means insufficient.

---

## cybernetic::bateson

Bateson's Hierarchical Learning Levels (1972).

L0: Fixed stimulus-response (zero learning). L1: Parameter adjustment via EMA (learning
within a model). L2: Model switching (learning to learn). L3: Epistemological restructuring
(learning about learning to learn). Hierarchy is enforced: L1 failure triggers L2, L2
failure triggers L3. Double bind: L1 stagnating + L2 blocked + L3 frozen.

### Structs

**`LearningContext`**

| Field | Type | Description |
|-------|------|-------------|
| `level` | `i64` | Current learning level (0-3) |
| `response_table` | `[i64; 16]` | L0 stimulus-response table |
| `active_param_set` | `i64` | Index of currently active parameter set (L2) |
| `params` | `[f64; 64]` | 8 parameter sets of 8 params each (64 total) |
| `param_counts` | `[i64; 8]` | Observation count per parameter set |
| `available_sets` | `i64` | Bitmask of available parameter sets |
| `meta_frame` | `i64` | Current meta-frame index (L3) |
| `frame_count` | `i64` | Total number of L3 restructures |
| `double_bind_count` | `i64` | Number of double bind detections |
| `l1_fail_count` | `i64` | Consecutive L1 stagnation count |
| `l2_fail_count` | `i64` | Consecutive L2 failure count |
| `l1_threshold` | `f64` | Minimum improvement to count as L1 success |
| `last_improvement` | `f64` | Most recent L1 improvement magnitude |

**`LearningEvent`**

| Field | Type | Description |
|-------|------|-------------|
| `changed` | `bool` | Whether any state change occurred |
| `level_engaged` | `i64` | Which learning level was engaged |
| `model_switched` | `bool` | Did L2 model switching occur? |
| `frame_restructured` | `bool` | Did L3 epistemological restructuring occur? |
| `double_bind` | `bool` | Was a double bind detected? |
| `context` | `LearningContext` | Updated learning context |

### Functions

```sio
pub fn bateson_new() -> LearningContext
```
Create a new learning context at L0 with default parameters. All 8 parameter sets
available, threshold 0.01.

```sio
pub fn level_one_update(
    ctx: LearningContext,
    stimulus: i64,
    outcome: f64
) -> LearningEvent with Mut, Div, Panic
```
L1 parameter adjustment. Updates the active parameter set via EMA (alpha=0.1).
Tracks consecutive stagnation for L2 trigger.

```sio
pub fn level_two_switch(
    ctx: LearningContext,
    threshold: f64
) -> LearningEvent with Mut, Div, Panic
```
L2 model switching. Scans available parameter sets (bitmask) for an alternative.
Resets L1/L2 failure counters on success.

```sio
pub fn level_three_restructure(ctx: LearningContext) -> LearningEvent with Mut, Div, Panic
```
L3 epistemological restructuring. Clears all parameter sets and counts,
restores all 8 sets as available, increments meta-frame counter.

```sio
pub fn detect_double_bind(ctx: LearningContext) -> bool with Div, Panic
```
Double bind detection. Returns true when L1 is stagnating (3+ failures),
L2 is blocked (no available sets), and L3 is frozen (3+ restructures).

```sio
pub fn learn(
    ctx: LearningContext,
    stimulus: i64,
    outcome: f64
) -> LearningEvent with Mut, Div, Panic
```
Main hierarchical learning state machine. Applies L1; if L1 stagnates (3+ consecutive
failures), triggers L2; if L2 fails, triggers L3; if L3 is frozen, detects double bind.

---

## cybernetic::languaging

Maturana's Languaging (1988): coordination of consensual coordination of action.

Languaging is NOT symbolic representation. It is a feedback loop where agents adjust actions
based on the OTHER's history, a consensual domain that grows/shrinks based on coordination
success, and distinction-making in that domain.

### Structs

**`LanguagingPair`**

| Field | Type | Description |
|-------|------|-------------|
| `a_action_prefs` | `[f64; 32]` | Agent A's action preferences |
| `b_action_prefs` | `[f64; 32]` | Agent B's action preferences |
| `action_freq` | `[i64; 32]` | Frequency of consensual actions |
| `num_actions` | `i64` | Number of possible actions |
| `rounds` | `i64` | Number of languaging rounds completed |
| `consensus_count` | `i64` | Total consensus events |
| `a_last_action` | `i64` | Agent A's last action (for feedback loop) |
| `b_last_action` | `i64` | Agent B's last action (for feedback loop) |
| `coordination_streak` | `i64` | Current consecutive consensus streak |
| `max_streak` | `i64` | Maximum streak achieved |
| `domain_stability` | `f64` | Stability metric: streak / rounds |
| `distinction_count` | `i64` | Number of linguistic distinctions drawn |
| `distinction_pairs` | `[i64; 32]` | Recorded distinction pairs (16 pairs, 2 slots each) |

### Functions

```sio
pub fn languaging_new(num_actions: i64) -> LanguagingPair with Mut, Div, Panic
```
Create a new languaging pair with uniform action preferences over `num_actions` actions.

```sio
pub fn languaging_step(
    pair: LanguagingPair,
    a_action: i64,
    b_action: i64
) -> LanguagingPair with Mut, Div, Panic
```
One round of languaging. Checks consensus, boosts/decays preferences via EMA,
applies Maturana's mutual feedback (each agent adjusts toward the OTHER's last action),
updates domain stability, and draws linguistic distinctions when streak > 3.

```sio
pub fn linguistic_mark(
    pair: LanguagingPair,
    action_a: i64,
    action_b: i64
) -> LanguagingPair with Mut, Panic
```
Draw a distinction (linguistic mark) in the consensual domain. Records
(action_a, action_b) as a distinction pair. Max 16 pairs.

```sio
pub fn has_consensual_domain(pair: LanguagingPair, threshold: i64) -> bool
```
Returns true if a consensual domain has formed (consensus_count >= threshold).

```sio
pub fn domain_size(pair: LanguagingPair) -> i64
```
Size of the consensual domain: count of actions with nonzero frequency.

```sio
pub fn domain_stability(pair: LanguagingPair) -> f64
```
Return the domain stability metric (coordination_streak / rounds).

---

## cybernetic::conversation

Pask's Conversation Theory (1975) -- Knowledge through dialogue.

Core principle: each participant maintains TWO models: their own belief and their model
of the other's belief. These update independently via observation. Agreement = convergence
of cross-models, not averaging of values.

### Structs

**`Conversation`**

| Field | Type | Description |
|-------|------|-------------|
| `p_id` | `i64` | Participant P's identifier |
| `q_id` | `i64` | Participant Q's identifier |
| `topic_id` | `i64` | Topic identifier |
| `p_value` | `f64` | P's actual belief |
| `p_variance` | `f64` | P's confidence in own belief |
| `q_value` | `f64` | Q's actual belief |
| `q_variance` | `f64` | Q's confidence in own belief |
| `p_model_of_q` | `f64` | P's estimate of what Q believes |
| `p_model_of_q_var` | `f64` | P's confidence in that estimate |
| `q_model_of_p` | `f64` | Q's estimate of what P believes |
| `q_model_of_p_var` | `f64` | Q's confidence in that estimate |
| `agreement_value` | `f64` | Current agreement metric |
| `model_accuracy_p` | `f64` | |p_model_of_q - q_value| |
| `model_accuracy_q` | `f64` | |q_model_of_p - p_value| |
| `rounds` | `i64` | Number of conversation rounds |
| `converged` | `bool` | Whether beliefs have converged |
| `convergence_threshold` | `f64` | Threshold for convergence detection |

### Functions

```sio
pub fn conversation_new(p_id: i64, q_id: i64, topic_id: i64) -> Conversation
```
Create a new conversation between participants P and Q on a topic.
Cross-models start with high variance (100.0) representing initial uncertainty about the other.

```sio
pub fn set_models(
    conv: Conversation,
    p_val: f64,
    p_var: f64,
    q_val: f64,
    q_var: f64
) -> Conversation with Mut
```
Set initial beliefs and cross-models. Cross-models are initialized to the other's
stated value with extra variance (+10.0).

```sio
pub fn converse(conv: Conversation) -> Conversation with Mut, Div, Panic
```
One round of Pask conversation with proper dual modeling. Phase 1: P observes Q's
stated value and updates its cross-model, then shifts own belief. Phase 2: Q does
the same with P's updated value. Phase 3: compute agreement, model accuracy, convergence.

```sio
pub fn agreement(conv: Conversation) -> f64
```
Return the current agreement metric (0.0 to 1.0).

```sio
pub fn has_converged(conv: Conversation) -> bool
```
Whether the conversation has converged (belief difference < threshold).

```sio
pub fn shared_understanding(conv: Conversation) -> f64 with Div, Panic
```
Shared understanding: midpoint of converged beliefs.

```sio
pub fn model_accuracy(conv: Conversation) -> f64 with Div, Panic
```
Average model accuracy across both participants. Lower = better.

```sio
pub fn conversation_asymmetry(conv: Conversation) -> f64
```
Asymmetry: how differently do participants model each other? |accuracy_p - accuracy_q|.

---

## cybernetic::coupling

Structural Coupling (Maturana/Varela).

Two systems coupled through recurrent mutual perturbation. Congruence is measured via
Pearson correlation on output histories (64-element circular buffer). Mutual information
is estimated from the correlation coefficient.

### Structs

**`Coupling`**

| Field | Type | Description |
|-------|------|-------------|
| `system_a_id` | `i64` | System A identifier |
| `system_b_id` | `i64` | System B identifier |
| `history_len` | `i64` | Total steps recorded (may exceed buffer size) |
| `congruence_value` | `f64` | Current Pearson correlation (0.0 to 1.0) |
| `a_history` | `[f64; 64]` | Circular buffer of system A outputs |
| `b_history` | `[f64; 64]` | Circular buffer of system B outputs |
| `mean_error` | `f64` | EMA of |a - b| prediction error |
| `error_variance` | `f64` | Error variance |
| `sum_a` | `f64` | Running sum of A values (for Pearson) |
| `sum_b` | `f64` | Running sum of B values (for Pearson) |
| `sum_a2` | `f64` | Running sum of A^2 values (for Pearson) |
| `sum_b2` | `f64` | Running sum of B^2 values (for Pearson) |
| `sum_ab` | `f64` | Running sum of A*B values (for Pearson) |

### Functions

```sio
pub fn coupling_new(a_id: i64, b_id: i64) -> Coupling
```
Create a new coupling between two systems.

```sio
pub fn couple_step(c: Coupling, a_val: f64, b_val: f64) -> Coupling with Mut, Div, Panic
```
Record one coupling step. Updates 64-element circular buffers, running sums for
Pearson correlation, and EMA prediction error. Congruence is clamped to [0, 1].

```sio
pub fn congruence(c: Coupling) -> f64
```
Return the current congruence (Pearson correlation, clamped to [0, 1]).

```sio
pub fn prediction_error(c: Coupling) -> f64
```
Return the EMA prediction error |a - b|.

```sio
pub fn mutual_information(c: Coupling) -> f64 with Div, Panic
```
Mutual information estimated from Pearson r: MI = -0.5 * ln(1 - r^2).
Uses Taylor series approximation. Capped at 3.0 for near-perfect correlation.

---

## cybernetic::mod

Module declaration file. Re-exports all nine cybernetic modules.
