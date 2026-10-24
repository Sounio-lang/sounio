<!-- docs:meta
topic_id: repo.docs.spec.2026-10.04-knowledge-uncertainty-and-provenance
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.04-knowledge-uncertainty-and-provenance
-->

# 04. Knowledge, uncertainty, and provenance

Status: source-grounded specification chapter for review, 2026-10-16. Research
baseline: `f5b2c5850be9f3d3995719ffcade44a6d851658a`. Companion catalog:
[04-evidence.tsv](04-evidence.tsv). Paths below are repository-relative;
identifiers identify implementation symbols, not guessed source line numbers.

This chapter describes the checked-in source and the contracts expressed by
fixtures. It does not certify a compiler executable. No compiler build, fixture
execution, numerical verification, or Lean build was performed for this chapter.
In particular, an expected-output directive is an expectation, not a receipt.
Changes made concurrently outside this chapter do not enlarge these claims.

## 4.1 Scope, standing, and semantic quantities

`EPI-001` — **specified-contract**. Preserve the following quantities as distinct
semantic facts. A representation may omit one, but omission does not identify it
with another quantity or establish that its value is zero.

| Quantity | Meaning in this chapter | What it does not establish |
|---|---|---|
| Value | A point estimate or nominal program result in a specified numeric representation | An exact real value, an uncertainty distribution, or a causal effect |
| Arithmetic error/correction | Information about numerical evaluation relative to a stated reference arithmetic or model | Measurement uncertainty or missing scientific knowledge |
| Measurement uncertainty | Dispersion attributed to measured inputs and propagated under an explicit model | The probability that an arbitrary claim is true |
| Confidence | A score, threshold, or modeled probability with an API-specific interpretation | A standard deviation, variance, or universal calibration guarantee |
| Ignorance | Missing information, unspecified uncertainty, distributional imprecision, or an unknown dependency | A deterministic zero-uncertainty result |
| Computational provenance | Source identifiers, derivation records, assumptions, policies, or audit lineage | A physical mechanism or a validated causal graph |
| Physical causality | A claim about interventions or mechanisms under a scientific model | A consequence of two values sharing a compiler source ID |

This separation follows `FOUNDER_INTENT.md` and is a reporting and interpretation
contract. The current compiler does not provide seven independent runtime lanes
for every `Knowledge<T>` value. In particular, the legacy field name `epsilon`
does not override the quantity stored in the field it aliases.

`EPI-002` — **source-observed**. Three implementation surfaces must be named when
describing epistemic behavior:

1. Compiler `Knowledge<T>`: parser `TypeKnowledge`, checker `TyKnowledge`, static
   metadata, synthetic runtime layout, and scalar sensitivity shadows.
2. Ordinary stdlib values, including `epistemic::knowledge::Epistemic`,
   `epistemic::correlation::CorrelatedValue`, and covariance-matrix APIs. Their
   function contracts apply independently of compiler `Knowledge<T>` rules.
3. ENIR/EISA profiles: explicit numeric/error/uncertainty/status/provenance
   interfaces and dedicated drivers, interpreters, and translation experiments.

Neither a matching name nor a matching mathematical formula supplies an implicit
conversion or an end-to-end preservation theorem between these surfaces.
`scripts/epistemic_knowledge_madaros_e2e_gate.sh` explicitly excludes a claim about
generic language `Knowledge<T>` from its stdlib API receipt. The source-level
rules below describe Madaros unless an engine or target is named explicitly.

Standing vocabulary used by both this chapter and the catalog:

| Standing | Meaning |
|---|---|
| `source-observed` | Descriptive rule read in current source; execution remains unverified here |
| `specified-contract` | Semantic or mathematical contract stated with its domain; not a claim that every backend implements it |
| `library-contract` | Behavior of a named stdlib representation/API, with its preconditions and limits |
| `target-contract` | Descriptive behavior restricted to an engine, driver, profile, or target |
| `open-decision` | An unresolved choice or mismatch; candidate behavior is not adopted as language design |

## 4.2 Compiler representation and syntax

### 4.2.1 AST and annotations

`EPI-003` — **source-observed**. `Parser.parse_knowledge_type` in
`self-hosted/parser/types.sio` constructs a `KnowledgeTypeInfo` with:

- `inner_type`, the payload type expression;
- optional `epsilon: EpsilonBound`, retaining comparison operator and numeric
  value;
- optional `validity: ValidityCondition`;
- optional `provenance: AstProvenanceKind`;
- optional `proof_constraints: KnowledgeConstraintList`.

Both `Knowledge<T>` and `Knowledge[T, ...]` delimiters are recognized. Despite a
comment saying that components occur only in bracket form, the actual
comma-processing loop is shared by both forms. For example,
`tests/run-pass/knowledge_provenance_measured.sio` contains both
`Knowledge[f64, Measured]` and `Knowledge<f64, Measured>` declarations. This is a
description of that parser, not an adoption of every parser recovery behavior as
valid syntax.

`EPI-004` — **source-observed**. The directly dispatched provenance annotation
tokens are `Derived`, `Computed`, `Measured`, and `Input`; validity tokens are
`Valid(...)`, `ValidUntil(...)`, and `ValidWhile(...)`. Unknown components have an
explicit E241 reporting path, `report_unknown_knowledge_component`. The AST enum
also contains `AstProvSource` and `AstProvLiterature`, and metadata conversion
knows those categories; that alone does not make `Source` or `Literature`
accepted annotation spellings. The parser's identifier-comparison and lexer
recovery branches consume some nonliteral epsilon forms without resolving their
meaning. Those branches are not a contract allowing arbitrary symbolic bounds.
Duplicate components overwrite an optional slot in the current parser; duplicate
annotation policy remains an unresolved syntax choice.

### 4.2.2 Static epsilon and runtime confidence

`EPI-005` — **source-observed**. The directional relation implemented by
`epsilon_mode_from_compare_op` and `epsilon_subsumes` in
`self-hosted/check/epistemic.sio` is:

| Required annotation mode | Descriptive compatibility condition |
|---|---|
| Unspecified mode-0 bound, negative sentinel | No epsilon restriction from that requirement |
| Mode 0, error/variance bound | Provided known bound must be at most the required bound; a confidence-floor or mode-3 value does not satisfy it |
| Mode 1, equality | Known provided and required values must be equal |
| Mode 2, confidence floor | Known provided value must be at least the required value |
| Mode 3 | Special unspecified-state handling; not a general numeric confidence |

For numeric comparisons, valid known epsilon values lie in `[0,1]`; negative
sentinels mean unspecified/unknown. `<` and `<=` collapse to mode 0, while `>`
and `>=` collapse to mode 2. The implementation consequently does not retain
strictness in the relation. Do not infer a strict mathematical inequality merely
from a source annotation containing `<` or `>`.

Recent source explicitly treats a literal `measure` uncertainty as a variance
bound: `checker_measure_literal_variance` and
`checker_check_measure_expr_inplace` can populate `knowledge_epsilon` with the
squared literal uncertainty. `knowledge_binary_result` adds known mode-0 bounds
for addition when the sum remains representable in `[0,1]`; otherwise its bound
stays unknown. `knowledge_same_slot_add_variance` uses the same-source variance
scaling for the supported repeated-identifier addition. The older
`epsilon_combine_independent` helper and its approximate square root are not the
authority for this newer addition path.

`EPI-006` — **open-decision**. A unified meaning for epsilon has not been
established across these APIs. In particular:

- Static mode-0 `knowledge_epsilon` can be a variance bound, which is
  dimensionful for a dimensionful payload, despite the numeric `[0,1]` limit.
- `Knowledge(v, e=...)` is a confidence constructor, not a sigma constructor.
- Runtime `.epsilon` aliases `.confidence`, not `.variance`.
- Annotation polarity distinguishes an upper bound from a confidence floor,
  but `TypeEntry` reuses one numeric payload for these interpretations.

A future design must decide distinct names/types, units, strictness, unknown
states, and conversion rules before claiming these are interchangeable.
Normalizing all of them to either a standard deviation or a probability would
change existing behavior. The catalog records this as a decision, not a new
normative interpretation.

### 4.2.3 Construction and payloads

`EPI-007` — **target-contract**. Madaros
`Lowerer.lower_measure_call_ref` in `self-hosted/ir/lower.sio` stores the nominal
argument, squares the uncertainty argument into the variance slot, stores
confidence `1.0`, binds a variance shadow, and seeds first-order sensitivities.
The corresponding checker preserves the nominal argument's type, records
`PROVENANCE_KIND_MEASURED`, and issues a fresh noise source.

The mathematical input domain for a measurement standard uncertainty is finite
and nonnegative. Squaring a negative input, or obtaining overflow/NaN from a
malformed input, is not validation of that domain. This chapter does not claim a
complete runtime domain check in this constructor. In this Madaros lowering the
confidence is a constant; a third `confidence:` argument in a historic fixture
does not demonstrate that the lowering consumes it. Constructors must be
audited per engine rather than merged into one inferred signature.

`EPI-008` — **target-contract**. Madaros
`Lowerer.lower_knowledge_ctor_call_ref` implements the call form
`Knowledge(v, e=<confidence>, ...)` by storing zero variance and using the second
argument for confidence. The checker
`checker_check_knowledge_ctor_expr_inplace` starts with an empty noise-source
set, in contrast to `measure`. This is a declared point value with confidence,
not evidence that a real measurement has zero uncertainty. Struct-literal
construction is another route: `lower_knowledge_literal_variance` can recover a
variance shadow from its value expression. It must not be conflated with the
zero-variance call constructor.

The synthetic layout registered by `ir_register_knowledge_layout` has value at
index 0, variance at index 1, confidence at index 2, `epsilon` as another name
for index 2, and a provenance-write slot at index 3. The checker exposes
`.value`, `.variance`, `.confidence`, and `.epsilon`; it does not expose a
general readable `.provenance` field. A provenance string written to a literal
is not a verified lineage graph. This is an internal layout description, not a
stable foreign ABI.

`EPI-009` — **source-observed**. `Knowledge<T>` retains a payload type rather
than universally substituting `f64`. The integer payload fixtures for issue
2504 distinguish measurement, field access, function return, and unknown-return
refusal. The intrinsic-type gate pairs an actual `Knowledge<T>` argument with a
scalar impostor refused by E009. These witnesses do not establish general
uncertainty arithmetic for every payload. In particular, `Knowledge<f128>` and
its GUM propagation remain reserved/unsupported surfaces in the current
limitations record; `f128` arithmetic elsewhere does not close that gap.

## 4.3 Extraction, confidence gates, and observation

`EPI-010` — **source-observed**. Reading `.value` from compiler `Knowledge<T>`
requires the `Epistemic` effect in the checker, with E170 for its absence.
Reading `.variance`, `.confidence`, or `.epsilon` does not require the same
extraction permission. Arithmetic on an extracted scalar may still carry
compiler sensitivity shadows; a scalar source type alone does not imply that
the compiler has forgotten its measurement dependencies.

`EPI-011` — **source-observed**. Explicit discharge operations have their own
contracts. `acknowledge(k, reason)` returns the payload and is deliberately
available without requiring `Epistemic` in the inplace checker.
`Knowledge.unwrap(reason)` has fixtures requiring exactly one string reason.
The inplace `acknowledge` path checks the existence of a second argument and
visits its type, but its displayed body does not itself enforce a string
reason. The method's reason-type fixture must not be presented as coverage of
the free function's entire argument-validation path. A reason string records an
explicit decision; no inspected rule proves its truth, scientific adequacy, or
durable storage in an audit log.

`EPI-012` — **target-contract**. `require_confidence(k, minimum)` has an explicit
bootstrap x86 implementation, `compile_require_confidence_call_x86` in
`self-hosted/compiler/lean_single.sio`: compare the stored floating confidence
with the supplied threshold, fail via `emit_assert_fail` when the comparison
does not pass, then extract the payload. There is an ARM64 counterpart.
`tests/run-pass/knowledge_require_confidence.sio` and the arity refusal fixture
express expectations. Madaros builtin registration by itself does not prove
equivalent runtime lowering; no equivalent dedicated method was found in the
inspected `ir/lower.sio`. Threshold range, NaN behavior, and any missing
Madaros implementation need dedicated engine-specific evidence. A successful
threshold comparison is not statistical calibration.

`EPI-013` — **source-observed**. Observation comprises separate surfaces:

| Surface | Current evidence boundary |
|---|---|
| `Unobserved<T>` and `with Observe` | Checker permissions for comparisons, extraction/casts, and selected IO/boundary uses; positive declaration and negative boundary fixtures |
| An ordinary function named `observe` | For example, `observe_pattern_ok.sio` defines its own function; its name does not introduce a universal observation intrinsic |
| `epistemic::observe` helpers | `observe_bare`, `observe_with_prior`, and `observed_variance` operate on explicit scalar moments; these are not compiler `Knowledge<T>` method signatures |
| Historic `observe(prior, observation, uncertainty: ...)` fixtures | `observe_contraction.sio` expresses a normal-normal posterior expectation; the modular checker explicitly does not register a generic `observe` builtin, and no dedicated lowering was found in the inspected modular lowering |
| `observe_transition` | A separate acquisition/monitoring-plan checker surface, with `observe_transition_requires_plan.sio` as a refusal witness; not a synonym for Bayesian moment updating |

For a scalar normal prior and an observation with independent normal error,
strictly positive prior variance `v_p` and observation variance `v_o` give the
model-specific posterior expressions

\[
 m_{post}=\frac{v_o m_p+v_p y}{v_p+v_o},\qquad
 v_{post}=\frac{v_pv_o}{v_p+v_o}.
\]

These are a mathematical model description, not a claim that observing always
contracts arbitrary epistemic uncertainty. The parent-reported symbolic review
dispositions are recorded in section 4.10.4; they do not establish executable
observation support. Zero
variance, inconsistent exact observations, dependent errors, biased models,
and non-normal likelihoods require separate rules. The helper's zero-total
fallback does not settle those scientific cases. The information-gain helper
uses a rational approximation to a logarithm; it is not an exact information
theorem.

## 4.4 First-order uncertainty and dependency propagation

### 4.4.1 Mathematical model and domain

`EPI-014` — **specified-contract**. The scalar first-order covariance model in
this chapter has a finite-dimensional input random vector `X`, finite mean
`mu`, and finite covariance matrix `Sigma`, with

\[
 \Sigma_{ij}=\mathbb E[(X_i-\mu_i)(X_j-\mu_j)].
\]

`Sigma` is symmetric positive semidefinite (PSD). Zero variances and singular
covariance matrices are allowed; strict positive definiteness is not required.
For a differentiable scalar function near `mu`, let `g = grad f(mu)`. Its affine
linearization `L(X) = f(mu) + g^T (X-mu)` has

\[
 \operatorname{Var}(L)=g^T\Sigma g
 =\sum_i g_i^2\Sigma_{ii}
   +2\sum_{i<j}g_i g_j\Sigma_{ij}.
\]

For affine `f`, this is the variance of `f(X)` exactly in real arithmetic.
For nonlinear `f`, it is a local first-order approximation to that variance.
Differentiability and a finite input covariance alone do not bound its remainder
or guarantee finite variance of the transformed random variable. Accuracy needs
control of nonlinearity, input spread, and tails on the model's domain. For
vector output, the corresponding linearized covariance is `J Sigma J^T`.

With `u_i = sqrt(Sigma_ii)` and nonzero `u_i u_j`, a correlation coefficient is
`rho_ij = Sigma_ij/(u_i u_j)`. The zero-variance case requires zero covariance
with every other input. Pairwise `|rho| <= 1` is necessary but does not ensure
PSD of a matrix of three or more variables. Unknown covariance is not zero
covariance. Uncorrelatedness suffices to drop first-order cross terms;
independence is sufficient but not necessary, given these finite moments, and
is not inferred from that calculation. Neither condition makes a nonlinear
first-order approximation an exact transformed-moment identity.

The local derivatives for product and quotient use the nominal input values.
For a quotient, a nonzero nominal denominator permits evaluation of the local
derivatives but does not establish finite ratio moments or useful propagation
accuracy. A defensible local approximation needs a controlled region separated
from denominator zero and control of any probability mass outside that region.
A ratio distribution approaching the singularity can have divergent moments
despite finite input moments and a nonzero nominal denominator. Exact ratio
variance additionally requires square integrability of the ratio itself. A
denominator bounded away from zero together with a finite numerator second
moment is sufficient for that integrability, but does not make the first-order
quotient variance formula exact.
`log` requires its real domain, `sqrt` has a singular derivative at zero, and
general real `pow(x,y)` with varying `y` needs the domain of `log(x)`.
Piecewise derivatives along an executed branch do not integrate uncertainty in
which branch is taken. At a discontinuity or nondifferentiable point there is no
automatic first-order variance contract for the whole distribution.

This affine model does not itself license a distribution-specific moment
replacement. The compiler's exponential/lognormal override described under
`EPI-015` is an additional implemented branch with additional model assumptions;
it is not derived from measurement variance alone.

The formulas in this section and section 4.5 have parent-reported symbolic
review dispositions in section 4.10.4. They are not asserted to have been proved
in Lean, numerically verified, or validated over the whole numeric range by
this documentation task.

### 4.4.2 Compiler channels and scalar intrinsics

`EPI-015` — **target-contract**. Madaros first-order lowering associates
measurement sources with sensitivity channels and source variances. In
`self-hosted/ir/lower.sio`, `fo_seed_from_variance`, the `fo_expr_sens` arrays,
and `fo_emit_var_from_sens` are the operational connection between measurement
and scalar variance. The observed scalar intrinsic interface is:

| Expression | Meaning on the tracked, supported path |
|---|---|
| `variance_of(expr)` | First-order propagated variance, stored variance for recognized direct `Knowledge` returns, or the qualifying exponential/lognormal override described below |
| `uncertainty_of(expr)` | Square root of the variance returned by the lowering path |
| `sensitivity_of(expr, channel)` | Partial derivative with respect to a tracked source channel |
| `covariance_of(left, right)` | Bilinear covariance from the two sensitivity vectors and the source covariance state; it does not generally share the exponential/lognormal variance override |

**Implemented distribution-specific exception.** In `fo_apply_transfer_kind`,
kind 6 records `exp_affine` when the derivative name is `exp` and
`fo_hess_is_clear()` finds no live Hessian registers. When its nominal and
variance registers are available, the branch retains the ordinary exponential
derivative sensitivities, sets `FO_LOGN_ACTIVE`, and returns
`fo_emit_lognormal_var` instead of `fo_emit_var_from_sens`. With exponent
nominal value `m_z` and variance `v_z`, the emitted expression is

\[
 \exp(2m_z+v_z)\bigl(\exp(v_z)-1\bigr).
\]

This is the lognormal variance expression for a Gaussian exponent, rather than
the tangent expression `exp(2*m_z) * v_z`. `fo_logn_note` and
`fo_logn_activate` preserve the recorded state for recognized local bindings,
so the exception is not limited to an immediately nested exponential call.
The corresponding mean override is described under `EPI-021`.

This is descriptive current-source behavior, not an approved inference that
every measured exponent is Gaussian or affine. `measure` supplies moments and
source identity without a distribution certificate. `fo_hess_is_clear` checks
absent registers, which can reflect disabled or incomplete Hessian tracking
rather than an affine exponent. Because tracking is activated syntactically,
a nonlinear exponent with no second-order metric in its function needs a
separate discriminating witness. `knowledge_exp_lognormal.sio` contains a
second-order mean request and compares qualifying exponential and curved
exponent cases, but does not cover that tracking-disabled distinction.

`covariance_of` still uses the retained derivative sensitivities through
`fo_emit_cov_from_sens`. Its self-covariance formula can therefore differ from
the lognormal `variance_of` formula. A common-moment interpretation of those
metrics is not established on this mixed path. Missing acceptance evidence
includes identical-expression variance/self-covariance comparison, a nonlinear
exponent with and without second-order activation, a non-Gaussian model with
the same first two moments, and direct versus locally bound exponential
expressions. All such witnesses remain unrun here.

Sensitivity storage has 32 channels in this Madaros implementation. The inplace
checker reports E242 for recognized literal indices outside `[0,32)`;
Hessians have a different bound, described below. The checker permits some
nonliteral index expressions, while the lowerer only selects slots using
literal indices and otherwise returns zero. Thus runtime indexing is a known
gap, not an accepted rule that an unrecognized index means zero dependence.
Arity checks and a successfully returned `f64` do not establish complete
argument-domain validation.

Channel capacity is also a semantic boundary: `fo_seed_from_variance` clears
the sensitivity vector when the 32-channel table is full and relies on scalar
variance fallback. That can preserve an individual variance without preserving
the shared-source identity needed by later covariance calculations. The
checker's distinct E178/E179 capacity diagnostics must not be described as
covering every lowering-channel overflow.

`EPI-016` — **target-contract**. `correlate(a, b, rho)` records a coefficient
between channels identified by the two tracked expressions and evaluates to
the coefficient. `lower_correlate_call_ref` stores literal coefficients in the
rho table and nonliteral coefficients in register-backed entries. Negative
literal expressions and computed coefficients have dedicated fixtures. Rho
state is reset per function by the first-order setup.

The rho table and off-diagonal variance loop cover only source indices in
`[0,8)`, even though diagonal first-order sensitivity storage covers 32.
`fo_cov_rho_set` and its register counterpart ignore pairs outside that
eight-source domain. Consequently a valid `sensitivity_of` index does not
guarantee that an explicitly declared correlation involving that channel is
represented. No diagnostic for this narrower correlation capacity is
established by the inspected setter path.

This implementation uses `fo_primary_channel` to choose the channels; it is not
an API for installing an arbitrary covariance matrix between two composite
multisource expressions. Its scan considers live sensitivity-register entries
only in the first eight slots and rejects multiple candidates in that scan;
the presence of a register does not establish a nonzero runtime derivative.
Mixed low/high-channel expressions and cancelled gradients need additional
selection witnesses. Nor does the shown checker prove finite rho, its range,
or global PSD. The mathematical preconditions of `EPI-014` remain the caller's
model obligation. A source fixture named
`correlate_rho_must_be_literal.sio` still exists, but current checker comments,
register-backed lowering, and `knowledge_runtime_rho.sio` show that literal-only
wording is stale for this path. That negative fixture is not evidence for
rejecting every computed coefficient.

`EPI-017` — **target-contract**. First-order propagation across user functions
is implemented for a bounded, shape-dependent set of bodies. Current
`fo_inline_call_variance_ex` re-lowers a registered helper body using the
caller's values, sensitivities, variances, and available Hessians. It checks a
nesting cap of eight, a budget of 96 inlines per outer call, at most sixteen
ordinary arguments, a nested-statement budget of 24, supported return shape,
and safe-to-replay arguments. Struct arguments must resolve to recognized caller locals or their
references. Scalar returns and projected struct returns take distinct paths.

The source now also has witnesses for runtime `if`, `while`, `for`,
`loop`/`break`, selected `match` arms, local closures, borrowed fields, and
three-argument calls. These are positive fixtures, not reasons to promise all
calls preserve all derivatives. Unsupported replay, effectful conditions,
unrecognized calls, and resource caps still matter. Truncation handling uses
`FO_INLINE_TRUNCATED`; paths returning absent variance registers can ultimately
be read as zero by `lower_variance_of_call_ref`. A missing shadow is not proof
of certainty. The current source distinguishes some truncation cases with NaN;
this does not establish a uniformly explicit unknown state for every failure.

The current `gum_fo_across_call.sio` and `fo_call_boundary_arity3.sio` do not carry
`known-failure` directives. The first compares to the value specified by
`tests/oracles/gum_fo_across_call.cpp`; the latter primarily checks liveness.
`CLAUDE.md` still says they are known failures. Consequently, the historical
description “channels do not cross user calls” is too broad, while “KL-11 is
closed” is also too broad. These findings are fresh source/fixture inspection,
not fresh runtime confirmation.

`EPI-018` — **target-contract**. Returned, nested, and branded `Knowledge`
payloads have additional preservation mechanisms. `fo_ctor_chain_apply` and
`lower_direct_knowledge_call_variance_ref` recover tracked constructor/field
chains or read the actual returned variance slot for recognized call layouts.
The latter explicitly avoids replaying an effectful factory in the general
variance walker. `knowledge_branded_return.sio`,
`knowledge_returned_expr.sio`, and
`knowledge_direct_call_variance_issue2827.sio` cover different shapes. A correct
nominal return alone is insufficient: its variance, source sensitivities,
correlation response, and static annotations are separate obligations. These
fixtures must not be replaced by an old report that only printed the nominal
amount.

`EPI-019` — **target-contract**. `variance_of` returns a value snapshot of a
tracked mutable scalar's variance. `lower_variance_of_call_ref` copies its
variance register so later reassignment of the scalar does not retroactively
alter an earlier read. This is distinct from the correlation table being
consulted when a new metric expression is lowered/evaluated. The implementation
does not promise that a previously materialized variance number is a live view
of later correlation declarations.

## 4.5 Hessians and higher-order scope

`EPI-020` — **target-contract**. Madaros `hessian_of(expr, j, k)` reads the
symmetric Hessian shadow for source indices in `[0,8)`, represented by 36
upper-triangular slots. `so_hess_idx` supplies the index mapping. This is smaller
than the 32-channel first-order domain. Literal arity/range refusals exist, but
nonliteral indices share the known zero-fallback gap described in `EPI-015`.
Second-order activation depends on syntactic use detected by
`so_expr_uses_hessian` and related scanning helpers, which recognize
`second_order_mean` as well as `hessian_of`; available derivative slots
are not a universal higher-order AD contract. In particular, absent Hessian
registers are not an affinity certificate, although the current exponential
override uses that test as described under `EPI-015`.

Products, quotients, and selected transcendentals have specific regression
fixtures. `fo_apply_transfer_kind` implements a `pow` transfer with base first
and second derivatives and an exponent first derivative. It does not establish
the complete mixed Hessian when the exponent is uncertain. Its formulas divide
by the base, so zero-base behavior is a separate domain boundary even when a
particular mathematical power function has a finite derivative there.
`madaros_kl11_pow_fo_gate.sh` includes a Hessian-chain sabotage control; a pass
of that gate would certify the named witness, not close arbitrary call or
power-domain coverage.

`EPI-021` — **target-contract**. `second_order_mean(expr)` is a separate
lowering, `lower_second_order_mean_call_ref`. Its ordinary branch forms the
nominal value plus the Hessian-covariance mean correction, including registered
off-diagonal covariances. When the expression walk activates `FO_LOGN_ACTIVE`,
however, it returns `fo_emit_lognormal_mean(FO_LOGN_U, FO_LOGN_V)` instead of
that computed correction. For the exponent quantities defined under `EPI-015`,
the emitted expression is `exp(m_z + v_z/2)`, not the quadratic-surrogate mean
`exp(m_z) * (1 + v_z/2)`. That Gaussian-exponent model assumption is not
certified by the constructor or by absent Hessian registers.

The ordinary Hessian-covariance branch has the eight-source domain. The
intrinsic's existence does not imply that ordinary expression evaluation
returns a corrected mean or that `variance_of` includes a general second-order
variance correction. Its exponential exception remains a distinct model path.
`knowledge_exp_lognormal.sio` is the relevant existing override fixture;
`madaros_gum_fo_second_order_mean.sio` covers another mean path. Neither is
reported as executed here. Correlation changes
in `knowledge_claim_meet.sio` exercise separate mean and variance readouts;
`claim_meet` itself is an interval-overlap utility, not a general probabilistic
fusion or truth procedure.

`EPI-022` — **library-contract**. Higher-order variance formulas must identify
the approximation model independently of Hessian availability. Fix a
deterministic expansion point `mu`, finite deterministic coefficients
`f(mu)` and `g`, and a finite deterministic symmetric matrix `H`. When they
come from derivatives, `g` and `H` are evaluated at that fixed point; they are
not random functions of the perturbation. Let `delta` be centered and jointly
Gaussian with covariance `Sigma`. The covariance may be singular and may have
nonzero off-diagonal entries. Define the quadratic surrogate

\[
 Q(\delta)=f(\mu)+g^T\delta+\tfrac12\delta^TH\delta.
\]

In real arithmetic its moments are

\[
 \mathbb E Q=f(\mu)+\tfrac12\operatorname{tr}(H\Sigma),\qquad
 \operatorname{Var}(Q)=g^T\Sigma g+
 \tfrac12\operatorname{tr}(H\Sigma H\Sigma).
\]

The mean formula for this surrogate needs finite second moments but not
Gaussianity. The displayed variance reduction explicitly assumes joint
Gaussianity, which supplies the required third/fourth moments; Gaussian
marginals alone do not establish the reduction. In the general expression
below, require finite fourth moments of every perturbation component and the
same deterministic finite coefficients and symmetric `H`. A random or
input-dependent Hessian cannot be pulled out of these moment contractions.
Without those Gaussian moment identities, non-Gaussian inputs can add skew and
fourth-moment contributions. These are
moments of the truncated surrogate. They do not prove that the true nonlinear
function has those moments, and do not supply a Taylor-remainder bound. At a
stationary point the first-order variance can vanish while the quadratic
variance does not; this is a model limitation, not a proof of deterministic
output.

More explicitly, for a centered perturbation with finite component fourth
moments and the fixed coefficients specified above, define
`M_ijk = E[delta_i delta_j delta_k]` and
`M_ijkl = E[delta_i delta_j delta_k delta_l]`. The general surrogate expression
is

\[
 \operatorname{Var}(Q)=g^T\Sigma g+
 \sum_{i,j,k}g_iH_{jk}M_{ijk}
 +\tfrac14\sum_{i,j,k,l}H_{ij}H_{kl}
       (M_{ijkl}-\Sigma_{ij}\Sigma_{kl}).
\]

Centering alone does not eliminate the linear/quadratic covariance contribution
involving `M_ijk`; Gaussian third central moments vanish, while Gaussian fourth
moments reduce the last term to the trace expression above.

The third-moment coefficient in the general expression is one. Its cross term
is `2 Cov(g^T delta, (1/2) delta^T H delta)`: the factor two in variance
expansion cancels the quadratic surrogate's half. Since the perturbation is
centered, the resulting contraction is `sum_ijk g_i H_jk M_ijk`. Inserting
another factor two would double-count this contribution. This symbolic
identity is the basis for rejecting the contrary reviewer suggestion; no
numerical oracle or formal proof build is implied.

`tests/run-pass/epistemic_second_order_variance_sanity.sio` contains an inlined
implementation and a linear zero-Hessian sanity case. That case cannot validate
nonzero Hessian corrections, global nonlinear coverage, or the imported
library's execution. `gum_so_analytic_hessian_two_points.sio` and the dedicated
Hessian fixtures address different obligations. Parent-reported symbolic
reviews and their accepted/rejected conclusions are recorded in section
4.10.4. They do not certify these fixtures, the nonlinear remainder, or an
implementation of general non-Gaussian moments; this chapter asserts no Lean
proof status for the formulas.

## 4.6 Compiler dependency guards and direct Knowledge arithmetic

`EPI-023` — **source-observed**. Checker noise-source sets and scalar numerical
channels are related purposes implemented by separate mechanisms. The checker
issues fresh source sets for `measure`, uses an empty set for deterministic
construction, propagates sets/sign information through expressions, and
conservatively uses unknown for unsupported call-result substitution. Proven
projection helpers may preserve the projected argument's source set;
arbitrary `Knowledge`-returning functions do not inherit “the last argument's
sources.” Overflow has E178/E179 diagnostic paths.

`checker_ns_should_refuse` is the E230 guard for independence-assuming direct
`Knowledge` operations. Addition and multiplication refuse non-disjoint
sources; division also refuses them. Subtraction has a narrower escape only
when shared coefficient products prove nonnegative covariance in its abstract
domain. Multiplication/division make resulting sign information opaque.
Unknown source/sign information is not a disjointness certificate. This is an
abstract dependency guard, not a full PSD solver or a physical independence
proof; distinct measurement constructors do not scientifically establish that
two instruments are uncorrelated.

`EPI-024` — **target-contract**. Madaros direct `Knowledge` arithmetic is
narrower than scalar arithmetic on `.value`. In the inspected checker,
E245 rejects ordinary two-`Knowledge` arithmetic lacking a supported lowering.
An implemented exception is addition of two bare `Knowledge` identifiers.
Repeated use of the identical identifier requires `Correlated` (E236 if
absent); for that addition the supported path uses same-source variance
scaling. Distinct-identifier addition also remains subject to source-set and
payload/unit checks. An inline `measure(...)` operand is deliberately outside
that identifier-only exception.

The checker contains a special hypercomplex-inner path, plus E251 for an
unsupported norm-multiplicativity shortcut at higher algebra levels. An
ordinary stdlib octonion struct and a checker `TyHyper` payload are not the
same classification. No blanket arithmetic or norm-preservation contract for
all `Knowledge<T>` follows from those exemptions. Bootstrap-engine arithmetic
must be described independently; E245 is a modular-checker refusal, not a
requirement that every engine reject a shape it implements correctly.

## 4.7 Standard library epistemic contracts

`EPI-025` — **library-contract**. `epistemic::knowledge::Epistemic` in
`stdlib/epistemic/knowledge.sio` is a flat struct with `val: f64`,
`variance: f64`, and `confidence: i64`. It is not compiler `Knowledge<T>` and
has no compiler source-set guard merely because of its module name. Its
confidence range is clamped to integer `[0,1000]`. `ep_new` accepts variance
without a general nonnegative/finite check; `ep_certain` uses zero variance and
confidence 1000; `ep_measured` squares a supplied standard deviation and uses
confidence 900. Free functions and method wrappers coexist. The current import
gate requires both method/free-function behavior and parity; the header's old
method-lowering warning is not sufficient evidence that methods still fail.

Other stdlib APIs use different structures again. `GUMComponent` in
`stdlib/epistemic/gum.sio` stores standard uncertainty, sensitivity, and degrees
of freedom. `gum_type_a`, `gum_type_b`, `gum_with_sensitivity`, and
`gum_combine2` construct and combine those components. `GUMResult` separates
nominal value, combined standard uncertainty, effective degrees of freedom,
coverage factors, and expanded uncertainties; `GUMUncertainty` is a retained
legacy component struct. These are not aliases of a compiler confidence slot.
Type-A standard-error estimation presupposes an appropriate repeated-sampling
model, while Type-B constructors encode supplied distributional information.
The API's sample-count/coverage-factor fallbacks do not validate those
scientific assumptions. A reported coverage factor is not a guarantee of
coverage for arbitrary distributions or biased models.

`gum_combine2`, `gum_combine3`, and their `gum_add`/`gum_sub` callers sum
diagonal component contributions without cross-covariance state. Their
first-order combination requires zero component covariances; they are not
substitutes for the full-matrix propagation surface. Effective degrees of
freedom and coverage calculations require their own statistical assumptions
beyond that covariance condition.

`epistemic::propagate` operates explicitly on the flat `Epistemic` struct:
its `exp`, `ln`, `pow`, `sin`, and related functions compute nominal values and
delta-method variance using their own scalar approximations and confidence
attenuation policies. The `exp_delta`/`cos_delta` names remain compatibility
aliases. Its fixed Monte Carlo kernels and generic function-pointer Monte
Carlo surface have different implementation and sampling assumptions. None is
an automatic compiler transfer rule, and no full-range accuracy or general
Monte Carlo convergence result is established by their module placement.

`EPI-026` — **library-contract**. The four plain binary functions
`ep_add`, `ep_sub`, `ep_mul`, and `ep_div` omit covariance. Their first-order
precondition is zero covariance, not merely different variable names.
Independence is sufficient under the finite-moment model but is not necessary
for that omission. Neither zero covariance nor independence makes the nonlinear
product/quotient approximation exact; quotient domain and ratio-moment
conditions remain those in `EPI-014`. Their
covariance-aware counterparts accept `Cov(X,Y)` explicitly:

| Function | Variance expression, with nominal values `x`, `y` |
|---|---|
| `ep_add_cov` | `v_x + v_y + 2 c` |
| `ep_sub_cov` | `v_x + v_y - 2 c` |
| `ep_mul_cov` | `y^2 v_x + x^2 v_y + 2 x y c` |
| `ep_div_cov` | `v_x/y^2 + x^2 v_y/y^4 - 2 x c/y^3` |

Addition/subtraction are affine variance identities on valid moments.
Multiplication/division are first-order approximations. `ep_scale` scales
variance by the square of its constant; `ep_shift` preserves variance;
`ep_square` uses the derivative of a single source rather than assuming two
independent copies. Passing `&x` twice to `ep_mul` does not trigger E230 and is
not equivalent to using `ep_square` or the known covariance. Confidence
attenuation factors in these functions are API policies, not Bayesian update
theorems. For example, the implementations scale the minimum confidence by
operation-specific integer factors, rather than generally multiplying two
probabilities as the module header suggests.

`EPI-027` — **library-contract**. The covariance-aware functions call
`ep_cov_input_verdict` and `ep_cov_output_verdict`. The input check requires
finite bounded variance/covariance values, nonnegative variances, zero
covariance when a variance is zero, a representable variance product, and the
two-by-two PSD bound subject to the coded floating tolerance. It rejects rather
than silently reporting certainty on inconsistent inputs. The implementation
uses the literal tolerance `1.8e-15` in the product test and permits a
rounding-sized negative result to be reported as zero using `7.2e-15` times a
term-magnitude scale. These are implementation constants read from source, not
newly verified error bounds.

This is not an exact PSD decision procedure: underflow, overflow, tolerances,
and nominal-value range have separate boundaries. The output check accepts a
nonnegative result without a general finite-output test, so it is not a
universal finite-result guarantee. `ep_gum_covariance_gate.sh` generates
negative runtime probes for non-PSD input, excess slack, negative variances,
NaN, infinity, product overflow, and nonzero covariance at zero variance. It
requires abnormal/nonzero termination without the escape sentinel; the chapter
does not claim those probes were run.

`EPI-028` — **library-contract**. `CorrelatedValue` in
`stdlib/epistemic/correlation.sio` tracks up to four source IDs with
sensitivities and source uncertainties, plus a residual variance. Its
`covariance` operation uses shared source identity, independently of compiler
`noise_set_id` and EISA origin IDs. The fifth-source overflow is absorbed into
the residual without an equivalent diagnostic channel. Retaining total
uncertainty at that step does not retain the identity needed to recognize
future shared-source covariance. This capacity loss is a known limitation,
not a safe general independence inference.

There is also loss before capacity overflow. `combine_correlated` retains a
merged source only when `abs(sensitivity * source_uncertainty) >= 1.0e-12`.
Smaller nonzero contributions are removed during compaction and are not added
to the later overflow residual. Calling those entries cancelled does not make
the threshold an exact-zero test: both uncertainty magnitude and dependency
identity can disappear even with one source. The absolute cutoff has no unit
parameter; it is not an approved unit-invariant numerical policy.

Source inspection therefore does not establish preservation of all nonzero
uncertainty under the four-slot limit. Adding a deterministic zero to a
single-source value below the cutoff is a discriminating, currently unrun
case. Below/equal/above-cutoff contributions, exact cancellation, subsequent
rescaling, and later covariance queries need separate witnesses. This pruning
is a known information-loss defect/domain boundary, distinct from fifth-source
absorption and floating underflow. Choosing exact-zero compaction, explicit
refusal, or a visible approximation with retained accounting requires a policy
decision; none is adopted by documenting the present branch.

`EPI-029` — **library-contract**. `CovarianceMatrix` in
`stdlib/epistemic/covariance.sio` stores a symmetric upper triangle in 300
`f64` entries with dimension at most 24. `cov_propagate_general` implements the
scalar quadratic form, and `cov_propagate_full` implements a multi-output
Jacobian propagation. Dimension, array extent, PSD, and valid input moments
are caller obligations unless the particular constructor checks them.

Current boundary behavior includes silent dimension clamping in `cov_new`,
absolute-value conversion of negative diagonal inputs in `cov_diagonal` and
`cov_set_variance`, zero on invalid indices in `cov_get`, and ignored invalid
indices in `cov_set`. These are descriptive API behaviors with information-loss
risks, not normative validation policy. `cov_is_positive_definite` is not a
substitute for a specification allowing singular PSD covariance. Pairwise
correlation checks cannot establish full PSD. The module's local fixed-iteration
square-root helper does not carry an independently established full-range
accuracy contract here.

## 4.8 Provenance, validity, and constrained payload boundaries

`EPI-030` — **source-observed**. Compiler provenance has static categories
`Derived`, `Source`, `Computed`, `Literature`, `Measured`, and `Input` in the
metadata implementation. `provenance_is_trusted` groups `Source`, `Literature`,
and `Measured`; `provenance_subsumes` requires a trusted provided category when
the requirement is trusted, and otherwise accepts the category. It does not
require exact category equality or authenticate an external source. The
parser's narrower spellings remain as described in `EPI-004`.

`check_knowledge_type` and its inplace lowering persist provenance and validity
in `TypeEntry`; `knowledge_meta_from_ty` reads them back. Absent provenance
falls back to `Derived`, and `measure` explicitly stamps `Measured`. This
present wiring supersedes an older audit description in which those fields
were dropped. Stdlib `ProvEntity`, `ProvActivity`, `ProvenanceRecord`, and
traceability chains in `stdlib/epistemic/prov.sio` are separate explicit data
structures; they are not automatically populated by every compiler expression.

`EPI-031` — **source-observed**. `Valid(duration)` metadata requires a
nonnegative numeric literal. `ValidUntil(date)` requires an accepted exact
`YYYY-MM-DD` date literal; `parse_iso_date_to_f64` validates date shape and month
length, including leap years, then stores an ordered decimal date encoding.
It is not Unix time, a monotonic clock, or elapsed seconds. `ValidWhile` is a
different predicate form. A nonliteral duration/deadline cannot be admitted by
letting a failed conversion silently become zero.

`validity_subsumes` compares like kinds: a provided duration/deadline must reach
at least the required duration/deadline; always-valid satisfies a finite
requirement; a requirement of always-valid needs an always-valid provided
value. `validity_intersect` takes the shorter like-kind window and uses an
unsatisfiable marker for incomparable kinds or nonidentical predicates. Such a
marker denotes the limits of this representation and must not be interpreted
as a proof that no physical conjunction could ever hold. There is no automatic
wall-clock expiry check established by these static comparisons.

`EPI-032` — **source-observed**. `ValidWhile(expr)` predicates are checked for
name resolution and boolean type, then compared through structural predicate
identity. The canonical token/interner machinery in
`self-hosted/check/epistemic.sio` resolves identity by structure, not hash
equality alone. A zero/unrecorded identity does not certify equality. This is
not arbitrary logical implication, SMT completeness, continuous monitoring,
or a theorem that the predicate remains true after mutation. Shadowed binding
and predicate-collision fixtures matter because identical textual appearances
need not justify identical semantic assumptions. In particular, the current
serializer refuses bare identifier references because it does not have binding
identity at that site; matching the spelling of a flag is not a supported proof
of predicate equality. Unrecognized subexpressions similarly abort comparison.
Some adjacent comments still
mention old span-based identity; the actual structural implementation outranks
them.

`EPI-033` — **source-observed**. A `Knowledge` boundary has several independent
axes: payload/ontology compatibility, epsilon compatibility, provenance, and
validity. `knowledge_call_boundary_compatible` checks exact inner-type equality
plus the metadata axes. The main checker also has ontology-aware subtype and
join paths, so that helper alone is not the full relation. The positive
`ontology_epistemic_covariance_pass.sio` is about **type covariance** under a
subclass relation, not numerical covariance. Its unrelated-subclass refusal
does not validate any uncertainty matrix.

Assignment, annotated fields, returns, branches, closures, arrays, and casts
need their own checks: a correct function-argument path does not prove them.
The named ontology/epistemic negative fixtures exercise those escape routes.
A metadata-bearing `Knowledge` literal is inspected by
`knowledge_struct_lit_apply_fields`; unsupported or missing metadata must not
be described as authenticated provenance or unconditional validity merely
because a runtime value exists.

`EPI-034` — **source-observed**. `Knowledge<T where {...}>` is parsed into
`proof_constraints`, but the ordinary modular checker explicitly calls
`checker_refuse_knowledge_where_mut` with E269 when such constraints are
present. Both the valid-value and invalid-value fixtures for issue 2753 are
intentional refusals: this prevents silently ignoring a predicate on that
path. Parsing and `knowledge_context` analysis do not establish backend
enforcement, and a frontend probe that extracts obligations is not successful
execution of a constrained program. This limitation is route-specific: a
separate native driver and explicit expanders exist below.

`EPI-035` — **target-contract**. The source-to-source guard bridge is an explicit
execution route. `scripts/ontology/expand_knowledge_runtime_guards.sh` rewrites
supported numeric constraints into helpers that check payload fields and then
lowers the proof-context annotation to plain `Knowledge<T>`. It supports a
bounded family of comparison/unit-labelled contexts, an optional deterministic
cache, and a diagnostic manifest. The Sounio expander in
`self-hosted/compiler/knowledge_runtime_guard_expander.sio` is another
implementation consuming a lowering plan; its advertised scope includes up to
two conjunctive numeric constraints and a bounded source buffer.

Successful expansion requires that every intended obligation be represented
and consumed, not just that output source compiles. The expansion gate includes
before/after obligation counts, positive and runtime-reject cases, diagnostics,
and cache identity. The gate named `knowledge_runtime_guard_native_lowering_gate.sh`
actually invokes the Sounio expander, then compiles the expanded source; its
name is not evidence that the ordinary checker inserts native guards. A
generated field predicate tests the payload value, not all possible values in
its uncertainty distribution, an ontology's truth in the world, or clinical
validity.

`EPI-036` — **target-contract**. A separate native-v2 compilation path now
consumes `KnowledgeRuntimeGuardLoweringPlan` directly.
`drv_precompute_knowledge_guard_step_walk` in
`self-hosted/compiler/native_compile_driver.sio` refuses a semantic error or an
unrepresentable outstanding threshold, and accepts a plan only when its
obligations are representable. Row-stream payload hydration/emission inserts
comparison/trap steps and reports emitted-guard counters. The plan's current
`native_lowering_ready` is computed over all obligations; an old “false” default
in a directive struct is not the final readiness verdict.

`knowledge_runtime_threshold_contract_gate.sh` requires an explicit driver,
source receipt, and driver digest. Its threshold matrix distinguishes runtime
acceptance/rejection, strict/inclusive comparisons, signed/zero/decimal/wide
integer thresholds, unit conversion, and dimension mismatch. This evidence is
for that bounded native-driver route. It does not repeal ordinary-checker
E269 or prove equivalent guards on every entrypoint, backend, or unsupported
constraint. The published route must name whether it used unexpanded source,
the shell expander, or the Sounio expander.

## 4.9 ENIR and EISA interfaces

### 4.9.1 ENIR data, verification, and lowering

`EPI-037` — **target-contract**. ENIR is a compiler-owned semantic artifact in
`self-hosted/enir/`, with independent schema, stage, and profile fields.
`EnirType` distinguishes value kind, arithmetic-error kind, uncertainty kind,
status tracking, provenance tracking, and EISA/general profile. Error kinds
include exact, dd64, qd128, and static-bound; uncertainty kinds include exact,
GUM1, interval, p-box, and unknown. A tag being representable is not evidence
that every operation implements that kind.

`EnirValue` separates value bits, floating class, four error words, uncertainty
bits, status/reason, and a provenance ID. `known_bits == 0` denotes unavailable
static payload information, not exact zero. `EnirProvenance` carries source-span,
input-origin, transform, assumption, and policy identifiers. `EnirObservation`
is an artifact receipt entry; it is not the language `Observe` effect or a
measurement update.

`enir_verify` checks structure and profile restrictions, including IDs,
references, counts, floating classifications, tracking fields, and permitted
branch uncertainty. CFG schemas add block arguments/edges and dominance checks.
`enir_print_canonical` and `enir_canonical_hash_l64` support deterministic
serialization/identity. A deterministic hash is not a cryptographic
attestation, an empirical calibration, or compiler-correctness proof.

`EPI-038` — **target-contract**. `enir_lower_source`,
`enir_lower_source_v1`, `enir_lower_source_v2`, and
`enir_lower_source_join_v2` lower their dedicated source slices, selected by
`self-hosted/enir/driver.sio`. They do not establish that arbitrary parsed
Sounio `Knowledge<T>` expressions traverse ENIR on the default Madaros path.
The original E1 artifact remains explicitly shadow data; later E2 code adds
source lowering, profiles, interpretation, fuel/control, memory/move, and
poison handling in separate bounded stages.

The interpreter uses its own numeric implementations rather than delegating
all results to the EISA reference implementation. That is useful for comparison
but does not by itself establish independent scientific validation. The finite
v2 profile, poison/trap/unsupported states, fuel stops, and frail branch counters
must remain visible when describing an execution. Representable interval/p-box
tags do not imply that branching on them is supported: the verifier restricts
branchability to its implemented exact/GUM1 cases.

`EPI-039` — **target-contract**. `enir_lower_to_mir`, `enir_mir_verify`,
and `enir_mir_validate_relation` implement an additional bounded ENIR-to-MIR
surface. Arithmetic E3A and memory/move E3B have separate gates; CFG/join MIR
have additional modules. Validation checks the stated relation and source
identity for supported profiles; it is not a proof that the whole Sounio
compiler factors through this representation. E3A explicitly refuses several
unsupported control/profile examples and has tamper/poison probes.

The E1/E2/E3 gate scripts include legacy Python verification/oracle components.
They are cataloged as existing repository evidence, not executed by this task
and not endorsed as ADR009-compliant numerical verification. In addition, E1
can return an explicit skip when protected compiler surfaces have drifted
outside its lane. Exit zero without reading that status is insufficient to
report ENIR validation as passed.

### 4.9.2 EISA numeric and uncertainty lanes

`EPI-040` — **library-contract**. EISA v0 `EReg` in `stdlib/eisa/core.sio`
contains `val: f64`, `err: Dd64`, `u: f64`, and `origin: i64`.
`ereg_true` reconstructs a corrected numeric quantity according to that API;
`u` is a standard-uncertainty lane, not its variance. `ereg_origin` admits
positive origin IDs and maps other IDs to the mixed/unknown sentinel zero.
`eisa_fuse_origin` preserves an origin only if both IDs match and are nonzero.

For equal nonzero origins, the v0 addition/subtraction uncertainty path uses a
linear envelope, `u_x + u_y`. In particular, `esub(x,x)` deliberately does not
claim covariance cancellation. This differs from the signed covariance identity
for a random variable minus itself. Multiplication and division retain their
uncorrelated first-order formulas even when origin IDs fuse; comments explicitly
leave their correlation corrections unimplemented. A mixed intermediate loses
shared-source identity. Avoiding false origin matches is not a guarantee
against understated uncertainty after that loss.

The origin gate includes equal-origin, independent, mixed-fusion, subtraction,
and force-equal controls, but also executes a legacy Python integer oracle and
temporarily edits `stdlib/eisa/core.sio`. It must not be run unmodified in a
shared read-only research lane. A fixture asserting one deliberate envelope is
not contradictory evidence against a different API asserting an exact signed
covariance formula; the quantities and policies differ.

`EPI-041` — **library-contract**. EISA v2 `EReg2` in
`stdlib/eisa/core_v2.sio` contains `val: f64`, `err: Qd128`, and `u: f64`.
On their main reconstruction branches its operations form a corrected
quad-double quantity and store the correction relative to the nominal binary64
result. This is a quad-double correction
representation, not an IEEE binary128 payload and not an automatic widening of
compiler `Knowledge<f64>`. Source comments about rounding/correctness depend
on the expansion algorithms and their assumptions; this chapter supplies no
new proof or full-range execution evidence for them.

`esqrt2` has an information-erasing early return outside that general
reconstruction path: when `x.val <= 0.0`, it returns `ereg2_exact(0.0)` before
examining `ereg2_true(x)` or the uncertainty lane. `ereg2_exact` clears both
the qd correction and `u`. A nominal zero carrying a positive finite correction
therefore does not take square root of that reconstructed input on this
branch; a negative-domain input is not reported as a visible domain failure.
Even with zero correction, nominal zero does not justify zero propagated
uncertainty by the first-order sqrt rule, whose derivative is singular there.

These are source-derived defects/domain boundaries, not adopted zero or sqrt
semantics. Direct-core admission must be distinguished from any EVM or bridge
poison policy. Missing unrun witnesses include nominal zero with positive
correction, nominal zero with positive uncertainty, a negative-domain case,
and an ordinary positive control, with the intended domain policy declared.
The existing v2 differential fixture does not independently certify those
direct-core cases.

The v2 struct has no v0 `origin` field, and its displayed addition uncertainty
formula is the uncorrelated root-sum-square form. Therefore “v2 uncertainty is
unchanged from v0” is too imprecise when current v0 origin-aware addition is
included. The bridge gates compare EVM and translated executable receipts for
named programs and include tamper/anti-vacuity checks. They are bounded
conformance experiments, not hardware EISA implementation or universal backend
parity. Zero-event flags derived from value/correction lanes do not infer a
physical reason for the zero.

## 4.10 Ignorance, causality, evidence, and remaining decisions

`EPI-042` — **library-contract**. The repository has explicit uncertainty-kind,
interval, p-box, Knightian, and model-contest representations, but these are
separate APIs with separate domains. Examples include `PBox` in
`stdlib/epistemic/pbox.sio`, another `PBox` in `knightian.sio`, and
`EpistemicValue` in `epistemic_kind.sio`. Their matching words do not establish
representation equality or sound implicit conversion to scalar variance.
Compiler `TyEpistemic` and `TyAleatoric` kind checks also do not identify either
with the flat public stdlib `Epistemic` struct.

An unknown covariance, missing derivative, unspecified epsilon, p-box width,
confidence score, and zero variance answer different questions. The existing
negative conversion fixtures do not prove that all omissions are rejected.
In particular, returning zero for an untracked scalar variance remains an
implementation limitation when the real reason is unavailable information.
Any future conversion from imprecise/distribution-free information to a single
moment must state its assumptions and information loss.

`EPI-043` — **specified-contract**. Computational provenance and source
dependence do not establish physical causality. `Intervention<T>`,
`Counterfactual<T>`, `Validated<T>`, conditional-independence types, and causal
library models require their own scientific assumptions and implementation
evidence. `cond_indep_structurally_holds` is not a universal d-separation proof:
its displayed fallback permits unresolved IDs and returns true after a small
set of structural checks. Neither a category named `Measured` nor an EISA
origin ID authenticates an intervention. No clinical validity or causal
identification claim follows from the compiler paths documented here.

`EPI-044` — **specified-contract**. Conformance reporting must preserve the
tested layer, engine, source/artifact identity, route, target, and outcome.
Successful parsing, checking, compilation, execution, numeric agreement, a
bounded gate, a theorem build, and physical validation are distinct results.
All fixtures and gates in this chapter are **unrun in this task**. No historic
receipt is promoted to a current pass. A positive parser-only declaration or
liveness test cannot replace an analytic-value assertion; an arbitrary failure
cannot replace the expected negative diagnostic; a skipped gate is not a pass.

### 4.10.1 Detailed source and fixture map

The companion TSV provides one row for every requirement. This table records
additional evidence and exactly what each family discriminates. A negative
entry naming a script refers to probes generated by that script; it is not a
claim that every probe exists as a checked-in standalone file.

| Area | Current source / symbols | Positive witness or gate | Negative / limitation witness | Interpretation and coverage gap |
|---|---|---|---|---|
| Parser/metadata | `self-hosted/parser/types.sio`: `Parser.parse_knowledge_type`; `self-hosted/parser/ast.sio`: `KnowledgeTypeInfo` | `tests/run-pass/knowledge_provenance_measured.sio` | `tests/compile-fail/knowledge_unknown_component_ident.sio`; `scripts/ci/knowledge_annotation_silent_drop_gate.sh` | Accepted tokens differ from enum inventory; no comprehensive recovery/duplicates contract |
| Epsilon relation | `self-hosted/check/epistemic.sio`: `epsilon_subsumes`, `knowledge_binary_result` | `tests/run-pass/epistemic_subsumption.sio`; `tests/run-pass/knowledge_epsilon_alias_witness.sio` | `tests/compile-fail/epsilon_bound_violation.sio` | Bound polarity and runtime alias are distinct; strictness collapsed |
| Payload correspondence | `self-hosted/check/check.sio`: `checker_check_measure_expr_inplace` | `tests/run-pass/knowledge_i64_measure_value_2504.sio`; `scripts/ci/madaros_intrinsic_knowledge_type_gate.sh` | `scripts/ci/fixtures/madaros_intrinsic_knowledge_type/reject_nonknowledge.sio`; `tests/compile-fail/knowledge_return_unknown_inferred_2504.sio` | Payload type preservation does not prove generic numerical propagation |
| Extraction | `self-hosted/check/check.sio`: `Checker.check_field_access`, `checker_check_acknowledge_expr_inplace` | `tests/run-pass/knowledge_value_with_epistemic.sio`; `tests/run-pass/knowledge_acknowledge.sio` | `tests/compile-fail/knowledge_value_requires_epistemic.sio`; `tests/compile-fail/knowledge_no_silent_unwrap.sio` | Field permission and explicit discharge differ; free-function reason typing needs separate evidence |
| Observation | `self-hosted/check/check.sio`: `checker_unobserved_cast_allowed`; `stdlib/epistemic/observe.sio`: `observe_with_prior` | `tests/run-pass/observe_pattern_ok.sio` | `tests/compile-fail/unobserved_no_observe.sio`; `tests/compile-fail/observe_print_boundary.sio`; `tests/compile-fail/observe_return_boundary.sio` | Declaration success does not execute Bayesian update; `observe_contraction.sio` is a separate unresolved route expectation |
| First-order covariance | `self-hosted/ir/lower.sio`: `fo_emit_var_from_sens`, `lower_covariance_of_call_ref` | `tests/run-pass/gum_fo_correlation_and_call_variance_2776.sio`; `tests/run-pass/knowledge_runtime_rho.sio` | `tests/compile-fail/epistemic_index_must_refuse.sio` | Runtime index, rho-domain, full PSD, and multisource registration coverage remain open |
| Exponential model override | `self-hosted/ir/lower.sio`: `fo_apply_transfer_kind`, `fo_emit_lognormal_var`, `fo_emit_lognormal_mean`, `fo_hess_is_clear` | `tests/run-pass/knowledge_exp_lognormal.sio` | No dedicated self-covariance, non-Gaussian-model, or tracking-toggle negative identified | Gaussian-exponent expressions replace tangent variance/quadratic mean on a register-state test; distribution and affinity are not certified |
| Calls and controls | `self-hosted/ir/lower.sio`: `fo_inline_call_variance_ex` | `tests/run-pass/gum_fo_across_call.sio`; `tests/run-pass/gum_fo_runtime_if_1706.sio`; `tests/run-pass/gum_fo_while_call.sio`; `tests/run-pass/gum_fo_for_loop_call.sio`; `tests/run-pass/gum_fo_match_call.sio`; `tests/run-pass/gum_fo_closure_call.sio` | Inline depth/body/budget and effectful-condition gaps in source; no single complete refusal fixture | Current fixtures supersede obsolete blanket known-failure wording; none executed here |
| Returned Knowledge | `self-hosted/ir/lower.sio`: `fo_ctor_chain_apply`, `lower_direct_knowledge_call_variance_ref` | `tests/run-pass/knowledge_branded_return.sio`; `tests/run-pass/knowledge_returned_expr.sio`; `tests/run-pass/knowledge_direct_call_variance_issue2827.sio` | Unknown-return fixtures for issue 2504 | Stored variance, re-lowered sensitivities, and static annotations have different mechanisms |
| Hessians | `self-hosted/ir/lower.sio`: `lower_hessian_of_call_ref`, `fo_apply_transfer_kind` | `tests/run-pass/epistemic_hessian_8inputs.sio`; `tests/run-pass/gum_so_analytic_hessian_two_points.sio`; `tests/run-pass/kl11_pow_fo_hessian.sio` | `scripts/ci/madaros_kl11_pow_fo_gate.sh` sabotage; `tests/compile-fail/epistemic_arity_must_refuse.sio` | Eight-source Hessian domain; no complete uncertain-exponent Hessian contract |
| Second-order moments | `self-hosted/ir/lower.sio`: `lower_second_order_mean_call_ref` | `tests/run-pass/madaros_gum_fo_second_order_mean.sio`; `tests/run-pass/epistemic_second_order_variance_sanity.sio` | Nonlinear remainder, non-Gaussian moments, and zero-gradient cases need additional coverage | Linear sanity does not prove quadratic correction; `variance_of` remains a separate path |
| Noise guards | `self-hosted/check/check.sio`: `checker_ns_should_refuse`, `checker_propagate_call_result_ns_inplace` | `tests/run-pass/correlated_independent_measures_ok.sio`; `tests/run-pass/correlated_slot_identity_with_effect.sio` | `tests/compile-fail/correlated_slot_identity_requires_effect.sio`; `tests/compile-fail/knowledge_binary_arith_has_no_lowering.sio` | E230, E236, and E245 discriminate independent obligations |
| Stdlib arithmetic | `stdlib/epistemic/knowledge.sio`: `ep_add_cov`, `ep_mul_cov`, `ep_cov_input_verdict` | `tests/run-pass/ep_gum_covariance.sio`; `scripts/epistemic_knowledge_madaros_e2e_gate.sh` | `scripts/ci/ep_gum_covariance_gate.sh` generated runtime refusals | Confidence policy and tolerance constants need their own validation; E230 does not reach ordinary structs |
| Source-tracked library | `stdlib/epistemic/correlation.sio`: `CorrelatedValue`, `covariance`, `combine_correlated` | `tests/stdlib/epistemic/test_correlation_stdlib.sio` | Nonzero contributions below the absolute compaction cutoff disappear without residual accounting; fifth-source identity loss is separate | Bootstrap-labeled fixture is unrun and does not discriminate below/equal/above-cutoff preservation, rescaling, or later covariance |
| Matrix library | `stdlib/epistemic/covariance.sio`: `cov_new`, `cov_propagate_full` | `tests/stdlib/epistemic/test_covariance_stdlib.sio` | Silent dimension clamp/invalid-index fallback in source | Full PSD and dimension-domain validation not guaranteed by constructors |
| Provenance | `self-hosted/check/epistemic.sio`: `provenance_subsumes`, `knowledge_meta_from_ty` | `tests/run-pass/knowledge_provenance_measured.sio` | `tests/compile-fail/ontology_epistemic_provenance_downgrade_reject.sio`; `scripts/ci/knowledge_provenance_boundary_gate.sh` | Gate has historical assumptions and rebuild/stale classification; categories are not authenticated evidence |
| Validity | `self-hosted/check/epistemic.sio`: `validity_subsumes`, `validity_intersect`, `parse_iso_date_to_f64` | `tests/run-pass/ontology_epistemic_validity_positive_pass.sio` | `tests/compile-fail/ontology_epistemic_validwhile_shadowed_binding_reject.sio`; `tests/compile-fail/ontology_epistemic_validuntil_kind_mismatch_reject.sio` | Static relation, not a clock or continuous predicate monitor |
| Type covariance | `self-hosted/check/check.sio`: `checker_check_call_args_inplace` | `tests/run-pass/ontology_epistemic_covariance_pass.sio` | `tests/compile-fail/ontology_epistemic_covariance_reject.sio` | Ontology substitution is distinct from statistical covariance |
| Ordinary refinement route | `self-hosted/check/check.sio`: `checker_refuse_knowledge_where_mut` | `tests/frontend/knowledge_static_numeric_constraint_positive.sio` is an analysis fixture, not ordinary-backend success | `tests/compile-fail/knowledge_where_refinement_refused_valid_2753.sio`; `tests/compile-fail/knowledge_where_refinement_refused_invalid_2753.sio` | E269 refuses even satisfiable predicates on the ordinary checker route |
| Expansion route | `scripts/ontology/expand_knowledge_runtime_guards.sh`; `self-hosted/compiler/knowledge_runtime_guard_expander.sio` | `tests/frontend/knowledge_runtime_guard_positive.sio`; `scripts/ci/knowledge_runtime_guard_expansion_gate.sh` | `tests/frontend/knowledge_runtime_guard_reject.sio` | Source transformation plus obligation drainage; not implicit compiler enforcement |
| Native threshold route | `self-hosted/compiler/native_compile_driver.sio`: `drv_precompute_knowledge_guard_step_walk` | `tests/frontend/threshold-matrix/mg_g_positive.sio`; `scripts/ci/knowledge_runtime_threshold_contract_gate.sh` | `tests/frontend/threshold-matrix/dimension_mismatch.sio`; runtime-reject matrix entries | Driver digest/source receipt and emitted-guard evidence required |
| ENIR artifact/driver | `self-hosted/enir/ir.sio`: `EnirType`, `EnirValue`; `self-hosted/enir/verify.sio`: `enir_verify` | `self-hosted/enir/shadow_fixture.sio`; `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh` | `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh` generated malformed artifacts | Shadow and lowered stages differ; scripts contain legacy Python and skip cases |
| ENIR/MIR | `self-hosted/enir/mir.sio`: `enir_mir_validate_relation` | `tools/eisa/eisa_enir_v2_add.eisa`; `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh` | Same gate generates unsupported-profile, poison, and tamper probes | Bounded interpretation/relation evidence, not integrated native compiler proof |
| EISA origins | `stdlib/eisa/core.sio`: `eisa_fuse_origin`, `eisa_u_add_sub` | `tests/run-pass/eisa_origin_correlated_add.sio` | `tests/run-pass/eisa_origin_independent_add.sio`; `tests/run-pass/eisa_origin_fusion_mixed.sio` | Negative is a no-fire control, not compile failure; v0 subtraction is an envelope |
| EISA v2/bridge | `stdlib/eisa/core_v2.sio`: `EReg2`, `eadd2`, `esqrt2`, `ereg2_exact`; `stdlib/eisa/bridge_x86.sio` | `tests/stdlib/eisa/test_eisa_evm_v2.sio`; `scripts/ci/eisa_bridge_conformance_gate_madaros.sh` | Nominal-nonpositive sqrt early return erases correction/u; dedicated direct-core controls missing; existing zero-flags known-failure probe addresses another route | Core-v2 fixture declares the bootstrap lane; direct-core admission is separate from interpreter/bridge policy and all witnesses are unrun |

### 4.10.2 Decisions and acceptance work still required

The following are explicit unresolved choices or implementation gaps, not
requirements silently added to the language:

| Question | Present boundary | Evidence needed before expanding the contract |
|---|---|---|
| What is epsilon? | Static variance bounds, confidence floors, and runtime aliases coexist | Approved distinct semantic quantities, unit/range policy, strictness, conversion rules, and discriminating fixtures |
| What is unknown uncertainty? | Several paths use negative sentinels or missing shadows, with some zero fallbacks | An explicit failure/unknown design and witnesses separating unknown from exact zero across every metric/call path |
| What does a source ID mean? | Checker sets, scalar channels, stdlib source IDs, EISA origins, and ENIR provenance IDs differ | Approved scopes, lifetime/alias/conversion rules, and source-identity preservation witnesses |
| How are correlations validated? | Runtime rho storage exists; full PSD is not enforced there | Invalid-rho and non-PSD-matrix refusals, zero-variance behavior, composite-expression semantics, and branch/state scope |
| What calls preserve epistemic information? | Multiple bounded re-lowering/field/return mechanisms | Source-built differential receipts at caps, effectful branches, recursion, aliases, imports, and arbitrary unsupported calls |
| What is higher-order uncertainty? | Hessian extraction and corrected means exist separately from first-order variance | Approved order/distribution model, non-Gaussian policy, remainder/validity boundaries, and independently reviewed nonzero-Hessian cases |
| When may lognormal moments replace local propagation? | Exponential transfer uses absent Hessian registers and saved `FO_LOGN_*` state without a Gaussian certificate | Explicit model/affinity contract, variance/self-covariance coherence, nonlinear tracking-toggle and locally bound-expression witnesses |
| May source compaction erase small uncertainty? | `combine_correlated` drops contributions below an absolute cutoff before residual accounting | Exact-zero/refusal/visible-approximation decision, unit-scale policy, and cutoff/rescaling/later-covariance witnesses |
| What can expire? | Static date/duration/predicate subsumption exists | Clock origin, timestamp units, scope/mutation rules, monitoring semantics, and explicit runtime enforcement evidence |
| Where are refinements enforced? | Ordinary E269, source expanders, and native-v2 row guards are different routes | Route-specific complete-obligation and exact-threshold receipts, no silent fallback, and a decision on unsupported shapes |
| What is trusted provenance? | Category grouping and explicit library records | Source authentication, persistence/serialization policy, derivation links, and independently checked claim-to-record binding |
| Which EISA uncertainty policy is intended? | v0 source-aware envelopes and v2 uncorrelated formulas differ | Approved correlation/envelope policy and non-vacuous cross-profile witnesses; no change inferred from version number |
| What is the EISA sqrt admission domain? | v2 nominal-nonpositive early return clears both correction and uncertainty before reconstruction | Explicit corrected-input/domain policy and direct-core/interpreter/bridge controls for zero, positive correction, uncertainty, and negative inputs |
| What formal result applies? | Mathematical models and theorem-like source comments are present | Exact theorem statement, imports/axioms/status audit, successful Lean build, and an explicit relation to executable/compiler semantics |

Review of the mathematical statements is an independent checkpoint from file
traceability. A `darwin-llm` request to `qwen38-27b` failed without a review;
a request to `coder-next` returned an advisory review of the first-order and
Gaussian quadratic-surrogate statements. It supported the principal formulas,
but incorrectly suggested that centering alone kills third-moment terms; that
suggestion was rejected. This was a formula review, not repository inspection,
numerical verification, or review of the whole chapter. The parent subsequently
reported completed Kimi K3 and xAI fallback (requested xai-fast alias; response
model grok-4.3) symbolic reviews; their supplied conclusions are adjudicated in
section 4.10.4. Provider completion is recorded
from that parent report, without claiming direct inspection of the provider
receipts or unanimous approval of the whole chapter. No proved theorem or
numeric-oracle result is claimed. Parent-lane evidence custody and governance
registration remain separate checkpoints; shared review logs and registries
are outside this chapter's write set.

### 4.10.3 Independent source cross-review adjudication

[13-epistemic-cross-review.md](13-epistemic-cross-review.md) was adjudicated
against the matching source at this chapter's baseline. All three MAJOR
findings were accepted; none was rejected. Existing requirement IDs and their
standing were preserved. The changes document exceptions and known defects;
they do not adopt the defects as language design or claim an implementation
repair.

| Review finding | Disposition and updated requirements | Remaining acceptance boundary |
|---|---|---|
| 1: omitted lognormal override | Accepted. `EPI-014`, `EPI-015`, `EPI-020`, and `EPI-021` distinguish affine/tangent propagation, syntactically activated Hessians, and distribution-specific variance/mean overrides. Catalog rows identify the override and `knowledge_exp_lognormal.sio`. | Gaussian/affinity certification, metric coherence, non-Gaussian and tracking-toggle/local-binding witnesses remain missing or unrun. |
| 2: below-cap source pruning | Accepted. `EPI-028` documents absolute-threshold removal without residual accounting, separately from capacity overflow; the catalog names `combine_correlated`. | Magnitude/identity preservation below/equal/above cutoff and after rescaling/covariance queries need a policy and unrun regressions. |
| 3: sqrt2 exact-zero early return | Accepted. `EPI-041` limits reconstruction wording to the applicable branches and names correction/uncertainty erasure on the nominal-nonpositive branch; the catalog names `esqrt2`. | Direct-core domain cases and interpreter/bridge comparisons remain unrun; no semantic repair is claimed. |

The review's additional GUM-component qualification was also accepted under
`EPI-025`: diagonal component sums need zero cross-covariances and do not carry
the full-matrix contract. Its primary-channel observation was made explicit
under `EPI-016`. These source adjudications do not replace the parent-owned
symbolic math-provider review dispositions below. No compiler, numerical
oracle, semantic gate, or proof build was executed for this revision.

### 4.10.4 Parent-reported symbolic math review adjudication

The parent reports completed independent symbolic reviews by Kimi K3 (response
model `aws-bedrock/kimi-k3:global`) and xAI fallback (requested xai-fast alias; response model grok-4.3).
Both completed responses reportedly have `finish_reason=stop`. The Grok4.7
attempts exhausted their length limits and did not produce a complete review.
This chapter records the supplied conclusions and their adjudication; the
parent retains provider receipts and external review logs.
This is symbolic/model review, not numerical execution, formal proof, runtime
conformance, or probabilistic calibration.

| Supplied reviewer conclusion | Disposition | Exact chapter treatment |
|---|---|---|
| Kimi: covariance omission needs zero covariance; independence is sufficient but not necessary | Accepted. | `EPI-014` and `EPI-026` state this explicitly, retaining finite-moment conditions and the distinction between affine identities and nonlinear approximations. |
| Kimi: a nonzero nominal denominator alone is inadequate for exact ratio moments or reliable local propagation | Accepted. | `EPI-014` distinguishes local derivative evaluation, a controlled region separated from zero, tail/outside-region control, and square integrability of the ratio. A sufficient moment-existence condition is not described as exactness of the delta-method formula. |
| Both providers: the Gaussian reduction needs joint Gaussianity and the general quadratic moments need finite fourth moments with a fixed symmetric Hessian | Accepted as a clarification of the stated model. | `EPI-022` fixes deterministic finite coefficients, a deterministic symmetric `H`, and a centered perturbation; distinguishes joint Gaussianity from Gaussian marginals; and states finite component fourth moments for the general variance expression. |
| xAI fallback: the general third-moment cross term is missing a factor two | Rejected. | `EPI-022` retains coefficient one and explains cancellation in `2 Cov(g^T delta, (1/2) delta^T H delta)`. An additional two would double-count the existing contraction. No compiler/source repair follows from the rejected suggestion. |

The lognormal override's uncertified distribution/affinity assumption, source
pruning, EISA sqrt domain loss, missing numerical regressions, and nonlinear
remainder obligations remain open. Symbolic review does not close these
implementation or scientific-accuracy boundaries. All existing EPI IDs and
their standings are preserved, and metadata remains `unvalidated` because no
runtime semantic gate was executed for the chapter prose.
