<!-- docs:meta
topic_id: repo.docs.spec.2026-10.13-epistemic-cross-review
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.13-epistemic-cross-review
-->

# 13. Independent numerical cross-review of chapter 04

Review date: 2026-10-16. Source baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.

Reviewed chapter: `docs/spec/2026-10/04-knowledge-uncertainty-and-provenance.md`.
Reviewed catalog: `docs/spec/2026-10/04-evidence.tsv`.

Chapter SHA-256 at review:
`e406433f0b86ee38bfc5ca6f7b04f43ecc665537b93e3853f7fb60b97b9a8711`.
Catalog SHA-256 at review:
`7c1387434caf33e3b701fb07a0b1e8411a5b0fb6f0d091b6ac5e5e3de35379a1`.

This is an independent source and mathematical-assumption review. Chapter 04,
its catalog, and matching implementation files were read without alteration.
No compiler, fixture, numerical oracle, or Lean build was run. No numerical
calculation was delegated to Python, Rust, Julia, or another numerical tool.
Mathematical expressions below are an analytical audit of the stated models and
transcriptions of source formulas, not newly executed numerical results or
formal proofs. Parent-owned canonical external math review remains separate.

## Disposition

No algebraic BLOCKER was identified in the printed first-order,
Gaussian-quadratic, or general third/fourth-moment expressions under their
stated assumptions. The chapter correctly avoids equating a quadratic
surrogate's moments with moments of an arbitrary nonlinear function.

There are three MAJOR source/specification gaps. They prevent an unqualified
signoff on the current numerical contract until the chapter describes the
exceptions or the implementation is repaired and independently validated.
These findings do not require changing the approved research direction or
silently turning implementation defects into normative language design.

| Finding | Severity | Chapter rules | Principal source and symbol |
|---|---|---|---|
| Undocumented lognormal override changes both the variance and mean contracts | MAJOR | `EPI-015`, `EPI-020`, `EPI-021`; model boundary in `EPI-014` | `self-hosted/ir/lower.sio::fo_apply_transfer_kind`, `fo_emit_lognormal_var`, `lower_second_order_mean_call_ref` |
| Source compaction discards nonzero uncertainty before the four-source cap | MAJOR | `EPI-028` | `stdlib/epistemic/correlation.sio::combine_correlated` |
| EISA v2 square-root early return erases finite correction and uncertainty | MAJOR | `EPI-041`, distinction required by `EPI-001` | `stdlib/eisa/core_v2.sio::esqrt2`, `ereg2_exact` |

All behavioral consequences below are source-derived and unrun. A source
counterexample is not represented as a fresh executable failure receipt.

## Finding 1 — MAJOR: the compiler has an omitted distribution-specific override

### Contract discrepancy

The `EPI-015` metric table describes `variance_of` as first-order propagated
variance, apart from a recognized stored-variance return case. `EPI-021`
describes `second_order_mean` as nominal evaluation plus Hessian-covariance
correction. Actual source implements a further lognormal branch in both paths.
This is not merely an inaccurate older comment: current executable source and
an existing fixture explicitly select it.

### Exact source evidence

In `self-hosted/ir/lower.sio`:

- `fo_apply_transfer_kind`, kind 6, sets `exp_affine` when the derivative name
  is `exp` and `fo_hess_is_clear()` returns true. On the qualifying branch it
  leaves the ordinary exponential derivative sensitivities in place, sets
  `FO_LOGN_ACTIVE`, and returns `fo_emit_lognormal_var(argv, v0)`.
- `fo_emit_lognormal_var` emits `exp(2*u + v) * (exp(v) - 1)`.
- `lower_second_order_mean_call_ref` returns
  `fo_emit_lognormal_mean(FO_LOGN_U, FO_LOGN_V)` when that flag is active,
  replacing the otherwise computed Hessian correction.
- `fo_emit_lognormal_mean` emits `exp(u + v/2)`.
- `fo_logn_note`/`fo_logn_activate` preserve the state for recognized local
  bindings; the exception is not restricted to an immediately nested call.

These are Gaussian-exponent/lognormal moment formulas, not the tangent
variance `exp(2*u)*v` or the quadratic-surrogate mean
`exp(u)*(1+v/2)`. This distinction is symbolic; none of these expressions was
numerically evaluated in this review.

`tests/run-pass/knowledge_exp_lognormal.sio` explicitly expects the lognormal
variance and mean for a measured scalar exponent, and separately expects the
tangent variance for a curved exponent. The chapter and its principal metric
evidence row omit that fixture and exception.

### Assumption and internal-consistency risks

`measure` supplies a nominal value and standard uncertainty. The inspected
constructor and this transfer branch do not carry or verify a joint Gaussian
distribution certificate. A finite variance, even finite fourth moments, does
not determine exponential moments. Thus the lognormal branch needs an explicit
model assumption; ordinary source-channel identity does not provide it.

`fo_hess_is_clear` tests absence of live Hessian registers, not an independent
certificate that the exponent is affine. `so_expr_uses_hessian`,
`so_block_uses_hessian`, and function setup activate second-order tracking
syntactically. Without tracking, cleared Hessian slots cannot distinguish a
linear exponent from an untracked nonlinear one. The current lognormal fixture
contains `second_order_mean`, so it does not independently test this
classification in a function with no second-order metric calls.

There is also a separate-metric consistency obligation. The exponential branch
retains tangent sensitivities, whereas `fo_emit_cov_from_sens` computes a
bilinear covariance from those sensitivities. Consequently
`covariance_of(exp(x), exp(x))` follows a different source formula from the
lognormal `variance_of(exp(x))` branch. They must not be described as moments of
one common propagated random variable without an explicit profile distinction.
This is a source-derived discriminating test obligation, not a measured mismatch.

### Required adjudication and acceptance evidence

Describe the distribution-specific branch under `EPI-015` and `EPI-021` and
separate it from the general first-order and quadratic-surrogate contracts.
Record the missing Gaussian/affinity certificate as an implementation/model
boundary. Do not adopt inferred Gaussianity or absent Hessians as normative
design merely because a fixture expects the branch.

Required new or extended, presently unrun witnesses:

1. `variance_of` and `covariance_of` applied to the identical exponential
   expression, with the intended profile made explicit.
2. A curved exponent in a function without second-order metric calls, compared
   with the same function after an unrelated Hessian/mean request is added.
3. A non-Gaussian input model with the same first two moments, demonstrating
   that lognormal moments require more information than measurement variance.
4. Literal versus locally bound exponential expressions, to exercise saved
   `FO_LOGN_*` state.

The existing `knowledge_exp_lognormal.sio` is the nearest positive fixture;
`madaros_gum_fo_second_order_mean.sio` covers a different mean path. Neither was
run. No canonical external math approval is implied.

## Finding 2 — MAJOR: nonzero small sources disappear before capacity overflow

### Contract discrepancy

`EPI-028` correctly identifies loss of dependency identity at the fifth source,
but omits another source-loss route that operates even with a single tracked
source and can remove its uncertainty magnitude as well as its identity.
Describing the limitation solely through residual absorption at capacity is
insufficient for this API's numerical domain.

### Exact source evidence

`stdlib/epistemic/correlation.sio::combine_correlated`, step 2, retains a source
only if `abs_f64(sens[rp] * us[rp]) >= 1.0e-12`. An entry below that threshold
is discarded. The discarded squared contribution is not added to
`residual_u2`; the later overflow loop sees only the already compacted entries.
The final `total_u` is reconstructed from the surviving entries and residual.

The header calls these entries cancelled. The branch actually includes every
nonzero contribution below an absolute threshold, independently of exact
cancellation. The threshold also has no unit parameter, so a claim of
unit-independent source preservation would require additional design.

### Discriminating unrun case

Construct one `correlated_from_source` value with positive standard uncertainty
`eta` strictly below the coded cutoff and add a deterministic zero produced by
`correlated_independent` with zero uncertainty. `add_correlated` supplies unit
combination coefficients to `combine_correlated`. Source inspection shows the
only contribution is dropped before residual accounting, leaving an empty
source representation with zero reconstructed uncertainty. This is not a
fifth-source overflow example and is not exact algebraic cancellation.

`tests/stdlib/epistemic/test_correlation_stdlib.sio` exercises moderate-sized
source uncertainties, same/different source IDs, and self-covariance. It does
not discriminate this below-cutoff erasure. Its numerical tolerances must not
be reused for a new small-source test in a way that treats zero as an adequate
approximation to every tested uncertainty.

### Required adjudication and acceptance evidence

Add the absolute-pruning rule as a known information-loss defect/unsupported
domain in `EPI-028`; qualify any magnitude-preservation claim. Distinguish it
from overflow absorption, exact cancellation, and floating underflow.

Future source repair needs a policy decision between exact-zero compaction,
explicit refusal, or a reported approximation with retained error accounting.
Merely moving the contribution to an independent residual would retain its
magnitude but still lose dependency identity, which must remain visible.

Required tests are nonzero contributions below/equal/above the cutoff, exact
cancellation as a separate control, and a later covariance query or rescaling.
No such tests were added or run by this review.

## Finding 3 — MAJOR: EISA v2 sqrt can replace corrected information by exact zero

### Contract discrepancy

`EPI-041` describes v2 operations as forming a corrected quad-double quantity
and storing its difference from the nominal result. `esqrt2` has an explicit
early return outside that general reconstruction path. The exception needs to
be stated as source behavior and a defect/domain boundary.

### Exact source evidence

In `stdlib/eisa/core_v2.sio`:

- `ereg2_true(x)` reconstructs `qd(x.val) + x.err`.
- `esqrt2(x)` immediately returns `ereg2_exact(0.0)` when `x.val <= 0.0`,
  before inspecting that reconstructed quantity or its uncertainty.
- `ereg2_exact` sets both the qd correction and uncertainty to zero.

The early return can therefore discard an existing nonzero correction and
nonzero uncertainty. For a finite representation with nominal zero and a
positive reconstructed correction, this branch does not apply square root to
the corrected input. For a negative corrected input it reports a finite
exact-labelled zero rather than making a domain failure visible. Even when the
correction is zero, returning zero uncertainty at nominal zero is not justified
by a first-order sqrt propagation formula, whose derivative is singular there.

This is a direct-core observation. A separate EVM poison policy must not be
assumed to repair every direct `esqrt2` call. Likewise, it is not a statement
that an arbitrary physical quantity has become exactly zero.

### Required adjudication and acceptance evidence

Qualify the operation-wide reconstruction wording in `EPI-041`. Specify the
currently implemented early-return branch as a known loss of correction and
uncertainty, without adopting it as the numerical design. The existing qd
renormalization caveat does not explain away a branch that does not invoke qd
reconstruction at all.

`tests/stdlib/eisa/test_eisa_evm_v2.sio` is a bounded differential witness, not
independent coverage of this direct-core boundary. Required additional unrun
cases include nominal zero with a finite positive correction, nominal zero
with positive uncertainty, a negative-domain case, and an ordinary positive
input control. Direct core and EVM/bridge admission must be tested separately.

## Mathematical assumptions checked without numerical execution

| Rules | Review conclusion | Scope that must remain explicit |
|---|---|---|
| `EPI-013` | Normal-prior/independent-normal-error posterior expressions have the appropriate mean weights and variance expression. | Positive variances, specified likelihood, and independence; no general observation-contraction claim. |
| `EPI-014` | The scalar covariance expansion and vector `J Sigma J^T` relation are consistent with the affine model. Singular PSD matrices are correctly allowed. | Nonlinear transformed moments require additional tail/domain/remainder control; pairwise rho bounds do not establish global PSD. |
| `EPI-020`, `EPI-021` | Hessian contraction uses half-weighted diagonal contributions and full unordered off-diagonal contributions in the ordinary mean branch. | Eight-source domain; syntactic activation and missing derivative information; Finding 1's additional lognormal override. |
| `EPI-022` | The centered Gaussian quadratic-surrogate variance has the stated trace term. The general expression retains the correct linear/quadratic third-moment contraction and centered fourth-moment contraction. | Centering is not vanishing skewness. A symmetric distribution can have zero third moments while its fourth moments still differ from Gaussian moments. |
| `EPI-022` | The surrogate mean depends on second moments, not Gaussianity. | The surrogate is not the full nonlinear transform; the text must not be promoted to a complete higher-order expansion of its variance without additional derivative terms and remainder analysis. |
| `EPI-026`, `EPI-027` | The signs and nominal sensitivities in the four covariance-aware binary formulas agree with their affine/delta-method interpretations. | Floating tolerance checks are not exact PSD proofs or verified full-range error bounds; underflow and output finiteness remain open as already stated. |
| `EPI-040`, `EPI-041` | The distinction between v0 origin envelopes and v2 uncorrelated formulas is correctly stated. `u` and compiler variance are correctly separated; `Dd64`/`Qd128` corrections are not IEEE binary128. | Missing origins, dependency loss, direct-core versus interpreter admission, and Finding 3's domain exception. |

For the general quadratic expression, finite fourth moments of the required
components and finite coefficients justify the moment contractions under
review. Zero Gaussian third moments eliminate the mixed linear/quadratic term;
the two non-cancelled Gaussian fourth-moment pairings supply the trace term.
These are the assumption checks behind the review conclusion, not a claim that
an executable or Lean model was verified.

The linear sanity program
`tests/run-pass/epistemic_second_order_variance_sanity.sio` has an inlined
`gum_second_order_variance`. The current `stdlib/epistemic/gum.sio` does not
define that symbol despite the fixture's mirror comment. The chapter is right
not to present the linear sanity case as imported nonlinear-library coverage.
No implemented non-Gaussian higher-moment propagation API was established by
these matching sources.

## Capacity and semantic-boundary checks confirmed in source

| Surface | Matching source | Finding relative to chapter 04 |
|---|---|---|
| First-order sensitivities | `self-hosted/ir/lower.sio::fo_seed_from_variance`, `fo_emit_var_from_sens` | Storage/diagonal traversal covers 32 channels; saturation clears sensitivities. The chapter's capacity warning is supported. |
| Registered off-diagonal correlation | `fo_cov_rho_set`, `fo_cov_rho_reg_set`, `fo_cov_rho_live` | Pairs outside the first eight source slots are ignored; both literal and register-backed rho exist. The chapter correctly distinguishes this from first-order capacity. |
| Correlation operand selection | `fo_primary_channel` | Scans only the first eight sensitivity slots and treats live register entries as candidates; it is not a general composite-covariance registration API. Additional mixed low/high-channel and cancelled-gradient witnesses are needed. |
| Hessian | `so_hess_idx`, `so_expr_uses_hessian` | Eight-source symmetric storage and syntactic activation match the chapter; absent slots are not an affinity certificate. |
| General covariance | `stdlib/epistemic/covariance.sio::cov_new`, `cov_get`, `cov_set`, `cov_propagate_full` | The 24-input capacity, clamping and invalid-index fallbacks are present. Constructor success is not PSD validation. |
| Covariance-aware scalar library | `stdlib/epistemic/knowledge.sio::ep_cov_input_verdict`, `ep_cov_output_verdict` | The chapter accurately reports implementation tolerances and nonfinite/underflow qualifications; their constants were not independently numerically validated here. |
| Stdlib GUM components | `stdlib/epistemic/gum.sio::gum_combine2`, `gum_combine3`, `ws2`, `gum_add`, `gum_sub` | These use diagonal component sums and no cross-covariance state. Clarify zero-covariance component assumptions separately from full-matrix propagation; effective DOF and coverage also need their statistical assumptions. |

Type-B constructors' large effective-DOF sentinel and tabulated coverage
factors are policies of this implementation; Type-B input alone does not prove
that every distribution has Gaussian tails or known variance. The chapter's
coverage disclaimer is appropriate. The GUM module's fixed-iteration sqrt
helper also needs a declared numerical domain before a full-range guarantee;
its function name is not such a guarantee.

## Recommended parent actions and checks performed

The chapter author should adjudicate Findings 1–3 and update the existing rule
prose/catalog without changing rule identity. This reviewer did not edit those
files or the implementation. Compiler/library repairs and new regression cases
remain separate work with appropriate ownership and numerical validation.

Checks performed: targeted source and fixture reads; identifier and branch
inspection; analytical audit of the printed moment expressions; chapter/catalog
SHA-256 capture; and confirmation that the inspected matching implementation
paths had no tracked local modifications. Every quoted behavioral witness is
unrun. No runtime pass/fail, exact floating tolerance certification, full-corpus
claim, or proved theorem is asserted.
