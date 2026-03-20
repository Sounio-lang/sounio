# Cybernetics API Reference

Complete API reference for `stdlib/cybernetic/`. All signatures taken directly
from the source `.sio` files.

---

## `cybernetic::distinction`

Spencer-Brown's Laws of Form + Varela's Calculus for Self-Reference.

### Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `UNMARKED` | `i64` | `0` | The void, outside all distinctions |
| `MARKED` | `i64` | `1` | Inside the distinction |
| `AUTONOMOUS` | `i64` | `2` | Self-indicating form (Varela's J) |

### Structs

**`Form`** -- A node in the calculus of indications.

| Field | Type | Description |
|-------|------|-------------|
| `value` | `i64` | Current value: UNMARKED, MARKED, or AUTONOMOUS |
| `children` | `[i64; 8]` | Nested sub-forms |
| `child_count` | `i64` | Number of active children |
| `depth` | `i64` | Nesting depth (number of marks crossed) |
| `is_reentrant` | `bool` | True if this form contains itself (re-entry) |
| `period` | `i64` | Oscillation period in imaginary time (0 = convergent, 2 = standard) |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `form_void() -> Form` | -- | Unmarked empty form (the void) |
| `form_marked() -> Form` | -- | Simplest distinction (MARKED state) |
| `form_reenter() -> Form` | -- | Self-indicating form J where J = cross(J) |
| `form_mark(f: Form) -> Form` | `Mut` | Axiom 1 (Calling): idempotent marking |
| `form_cross(f: Form) -> Form` | `Mut` | Axiom 2 (Crossing): toggle marked/unmarked |
| `eval_form(f: Form) -> i64` | -- | Evaluate form to canonical value via depth parity |
| `forms_equal(a: Form, b: Form) -> bool` | -- | Check equivalence under the calculus |
| `form_juxtapose(a: Form, b: Form) -> i64` | -- | Juxtaposition (Boolean OR) |
| `form_nest(outer: Form, inner: Form) -> i64` | `Mut` | Nesting (Boolean implication) |
| `solve_reentry(initial: i64, max_iter: i64) -> i64` | `Mut, Div` | Iterative solution of f = cross(f) |
| `solve_reentry_timed(initial: i64, max_iter: i64) -> Form` | `Mut, Div` | Solve re-entry with oscillation period tracking |
| `form_period(f: Form) -> i64` | -- | Get oscillation period of a form |
| `value_name(v: i64) -> i64` | -- | Returns the value itself (for pattern matching) |

---

## `cybernetic::eigenform`

Eigenform theory: fixed points of operators (von Foerster 1976, Kauffman 2003).

### Structs

**`EigenformResult`** -- Result of eigenform search via fixed-point iteration.

| Field | Type | Description |
|-------|------|-------------|
| `value` | `f64` | The fixed point x* where Op(x*) = x* |
| `variance` | `f64` | Variance (uncertainty) in the eigenform value |
| `iterations` | `i64` | Number of iterations until convergence |
| `converged` | `bool` | Did iteration converge within tolerance? |
| `residual` | `f64` | Final residual |Op(x) - x| |

**`Eigenbehavior`** -- Stability analysis of an eigenform.

| Field | Type | Description |
|-------|------|-------------|
| `eigenform` | `EigenformResult` | The eigenform itself |
| `convergence_rate` | `f64` | |Op'(x*)| -- slope of Op near x* |
| `stability_radius` | `f64` | How far you can perturb before losing attraction |
| `is_stable` | `bool` | True if |slope| < 1 (attractive fixed point) |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `find_eigenform(op: fn(f64) -> f64, initial: f64, tolerance: f64, max_iter: i64) -> EigenformResult` | `Mut, Div, Panic` | Banach iteration to find fixed point of `op` |
| `is_eigenform(op: fn(f64) -> f64, candidate: f64, tolerance: f64) -> bool` | `Div` | Check if `candidate` is a fixed point of `op` |
| `compute_eigenbehavior(op: fn(f64) -> f64, initial: f64, tolerance: f64, max_iter: i64, delta: i64) -> Eigenbehavior` | `Mut, Div, Panic` | Full eigenform search + Lyapunov stability analysis |
| `meta_eigenform(op: fn(f64) -> f64, initial: f64, tolerance: f64, max_iter: i64) -> EigenformResult` | `Mut, Div, Panic` | Second-order stability: do perturbed starts converge to same eigenform? |

---

## `cybernetic::observer`

Observer-inclusion principle (von Foerster) + blind-spot theorem (Luhmann).

### Structs

**`Observer`** -- An observer embedded in the system it observes.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `i64` | Unique identifier |
| `state_value` | `f64` | Observer's current epistemic state |
| `state_variance` | `f64` | Uncertainty in that state |
| `observation_count` | `i64` | Number of observations performed |
| `drift` | `f64` | Accumulated systematic bias |
| `drift_rate` | `f64` | Drift increment per observation |
| `precision_budget` | `f64` | Remaining precision (graded effect) |
| `initial_budget` | `f64` | Starting precision (for exhaustion ratio) |

**`Observation`** -- Result of an observation: updated observer + measurement.

| Field | Type | Description |
|-------|------|-------------|
| `observer` | `Observer` | Updated observer state (drift increased, budget consumed) |
| `value` | `f64` | Observed value |
| `variance` | `f64` | Total variance (measurement + drift + budget penalty) |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `observer_new(id: i64, drift_rate: f64) -> Observer` | -- | Create observer with default budget (1000.0) |
| `observer_with_budget(id: i64, drift_rate: f64, budget: f64) -> Observer` | -- | Create observer with specific precision budget |
| `make_observation(obs: Observer, system_value: f64, measurement_variance: f64) -> Observation` | `Mut, Div, Panic` | Core operation: observe a value, updating drift and budget |
| `observe_observer(outer: Observer, observed_obs: Observer) -> Observation` | `Mut, Div, Panic` | Meta-observation: observe another observer with compounded uncertainty |
| `blind_spot(obs: Observer) -> f64` | -- | Accumulated drift the observer cannot self-correct |
| `relative_blind_spot(obs: Observer) -> f64` | `Div, Panic` | Drift as fraction of total uncertainty |
| `precision_remaining(obs: Observer) -> f64` | `Div, Panic` | Budget exhaustion ratio (1.0 = full, 0.0 = depleted) |
| `observer_agreement(a: Observer, b: Observer) -> f64` | `Div, Panic` | Agreement between two observers (0.0 to 1.0) |
| `informative_disagreement(a: Observer, b: Observer) -> f64` | `Div, Panic` | How much we learn from observer divergence |

---

## `cybernetic::autopoiesis`

Self-producing systems (Maturana and Varela 1972).

### Structs

**`AutopoieticSystem`** -- A self-producing organizational closure.

| Field | Type | Description |
|-------|------|-------------|
| `component_count` | `i64` | Number of components in the network |
| `alive` | `bool` | Whether organizational closure is maintained |
| `generation` | `i64` | Production cycle count |
| `total_perturbation` | `f64` | Accumulated perturbation (structural drift) |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `system_new(permeability: f64) -> AutopoieticSystem` | -- | Create a new autopoietic system |
| `add_component(sys: AutopoieticSystem, produced_by: i64, produces: i64, value: f64, variance: f64) -> AutopoieticSystem` | `Mut, Panic` | Add a component to the production network |
| `produce_cycle(sys: AutopoieticSystem) -> AutopoieticSystem` | `Mut` | Execute one production cycle (increment generation) |
| `perturb(sys: AutopoieticSystem, value: f64, variance: f64) -> AutopoieticSystem` | `Mut` | Apply external perturbation (accumulates drift) |
| `is_alive(sys: AutopoieticSystem) -> bool` | -- | Check if organizational closure is maintained |
| `generation(sys: AutopoieticSystem) -> i64` | -- | Get current generation count |
| `structural_drift(sys: AutopoieticSystem) -> f64` | -- | Get total accumulated perturbation |

---

## `cybernetic::coupling`

Structural coupling between two systems (Maturana and Varela).

### Structs

**`Coupling`** -- Tracks mutual perturbation history between two systems.

| Field | Type | Description |
|-------|------|-------------|
| `system_a_id` | `i64` | Identifier for system A |
| `system_b_id` | `i64` | Identifier for system B |
| `history_len` | `i64` | Number of coupling steps recorded |
| `congruence_value` | `f64` | Current congruence (0.0 to 1.0) |
| `a_history` | `[f64; 64]` | Circular buffer of system A values |
| `b_history` | `[f64; 64]` | Circular buffer of system B values |
| `mean_error` | `f64` | Exponentially smoothed prediction error |
| `error_variance` | `f64` | Variance of the prediction error |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `coupling_new(a_id: i64, b_id: i64) -> Coupling` | -- | Create a new coupling between two systems |
| `couple_step(c: Coupling, a_val: f64, b_val: f64) -> Coupling` | `Mut, Div, Panic` | Record one interaction step, updating congruence |
| `congruence(c: Coupling) -> f64` | -- | Get current congruence (1.0 = perfectly coupled) |
| `prediction_error(c: Coupling) -> f64` | -- | Get smoothed prediction error |

---

## `cybernetic::conversation`

Pask's Conversation Theory: dual modeling with Bayesian updates.

### Structs

**`Conversation`** -- Two participants (P, Q) modeling each other on a topic.

| Field | Type | Description |
|-------|------|-------------|
| `p_id` | `i64` | Participant P identifier |
| `q_id` | `i64` | Participant Q identifier |
| `topic_id` | `i64` | Topic identifier |
| `p_value` | `f64` | P's current belief value |
| `p_variance` | `f64` | P's uncertainty |
| `q_value` | `f64` | Q's current belief value |
| `q_variance` | `f64` | Q's uncertainty |
| `p_model_of_q` | `f64` | P's model of what Q believes |
| `q_model_of_p` | `f64` | Q's model of what P believes |
| `agreement_value` | `f64` | Current agreement level (0.0 to 1.0) |
| `rounds` | `i64` | Number of conversation rounds |
| `converged` | `bool` | Whether agreement threshold has been reached |
| `convergence_threshold` | `f64` | Threshold for convergence detection |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `conversation_new(p_id: i64, q_id: i64, topic_id: i64) -> Conversation` | -- | Create a new conversation between P and Q on a topic |
| `set_models(conv: Conversation, p_val: f64, p_var: f64, q_val: f64, q_var: f64) -> Conversation` | `Mut` | Initialize participant beliefs and cross-models |
| `converse(conv: Conversation) -> Conversation` | `Mut, Div, Panic` | Execute one round of precision-weighted Bayesian exchange |
| `agreement(conv: Conversation) -> f64` | -- | Get current agreement level |
| `has_converged(conv: Conversation) -> bool` | -- | Check if convergence threshold has been reached |
| `shared_understanding(conv: Conversation) -> f64` | `Div, Panic` | Average of P and Q beliefs (the shared meaning) |

---

## `cybernetic::variety`

Ashby's Law of Requisite Variety (1956).

### Structs

**`VarietySystem`** -- Tracks environment, regulator, and outcome variety.

| Field | Type | Description |
|-------|------|-------------|
| `env_states` | `[i64; 32]` | Histogram of environment state occurrences |
| `env_total` | `i64` | Total environment observations |
| `reg_states` | `[i64; 32]` | Histogram of regulator state occurrences |
| `reg_total` | `i64` | Total regulator observations |
| `outcome_states` | `[i64; 8]` | Histogram of outcome occurrences |
| `outcome_total` | `i64` | Total outcome observations |
| `env_variety` | `f64` | Computed environment variety (distinct states) |
| `reg_variety` | `f64` | Computed regulator variety (distinct states) |
| `outcome_variety` | `f64` | Computed outcome variety (distinct states) |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `variety_new() -> VarietySystem` | -- | Create a new variety tracking system |
| `record_env_state(vs: VarietySystem, state: i64) -> VarietySystem` | `Mut, Panic` | Record an observed environment state |
| `record_reg_state(vs: VarietySystem, state: i64) -> VarietySystem` | `Mut, Panic` | Record a regulator response state |
| `compute_variety(vs: VarietySystem) -> VarietySystem` | `Mut, Div, Panic` | Compute variety counts from histograms |
| `has_requisite_variety(vs: VarietySystem) -> bool` | `Div, Panic` | Check Ashby's law: reg_variety >= env_variety |

---

## `cybernetic::bateson`

Bateson's Learning Levels (L0-L3).

### Structs

**`LearningContext`** -- State for multi-level learning.

| Field | Type | Description |
|-------|------|-------------|
| `level` | `i64` | Current learning level (0-3) |
| `response_table` | `[i64; 16]` | Fixed response lookup (L0) |
| `active_param_set` | `i64` | Index of currently active parameter set (L1) |
| `params` | `[f64; 64]` | Parameter storage (8 sets of 8 parameters) |
| `param_counts` | `[i64; 8]` | Usage counts per parameter set |
| `available_sets` | `i64` | Bitmask of available parameter sets (L2) |
| `meta_frame` | `i64` | Meta-frame identifier (L3) |
| `double_bind_count` | `i64` | Count of detected double binds |

**`LearningEvent`** -- Result of a learning step.

| Field | Type | Description |
|-------|------|-------------|
| `changed` | `bool` | Whether the learning step caused a change |
| `level_engaged` | `i64` | Which learning level was active |
| `context` | `LearningContext` | Updated learning context |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `bateson_new() -> LearningContext` | -- | Create a new learning context at L0 |
| `level_one_update(ctx: LearningContext, stimulus: i64, outcome: f64) -> LearningEvent` | `Mut, Div, Panic` | L1: exponential parameter update (alpha = 0.1) |
| `level_two_switch(ctx: LearningContext, threshold: f64) -> LearningEvent` | `Mut, Div, Panic` | L2: switch to next available parameter set |
| `detect_double_bind(ctx: LearningContext) -> bool` | `Div, Panic` | Check if all parameter sets are exhausted (double bind) |

---

## `cybernetic::languaging`

Maturana's Languaging: consensual domain coordination.

### Structs

**`LanguagingPair`** -- Two agents coordinating actions.

| Field | Type | Description |
|-------|------|-------------|
| `a_action_prefs` | `[f64; 32]` | Agent A's action preference weights |
| `b_action_prefs` | `[f64; 32]` | Agent B's action preference weights |
| `action_freq` | `[i64; 32]` | Frequency of coordinated (matching) actions |
| `num_actions` | `i64` | Size of the action space |
| `rounds` | `i64` | Total interaction rounds |
| `consensus_count` | `i64` | Number of rounds where both agents chose the same action |

### Functions

| Signature | Effects | Description |
|-----------|---------|-------------|
| `languaging_new(num_actions: i64) -> LanguagingPair` | `Mut, Div, Panic` | Create a pair with uniform action preferences |
| `languaging_step(pair: LanguagingPair, a_action: i64, b_action: i64) -> LanguagingPair` | `Mut, Panic` | One round: reinforce matching actions, decay preferences |
| `has_consensual_domain(pair: LanguagingPair, threshold: i64) -> bool` | -- | Check if consensus count meets threshold |
| `domain_size(pair: LanguagingPair) -> i64` | -- | Count distinct actions that have been jointly performed |
