# Checked finite probability information measures

Module: `math::probability`. All five quantities are in **nats**, using natural
logarithms. These are numerical evaluations of finite probability-vector
functionals, not estimators with sampling-error guarantees, exact arithmetic,
or posterior confidence measures.

## API

```sounio
pub struct InformationResult {
    pub status: i64,
    pub value: f64,
    pub infinite: bool,
}
pub fn information_shannon(p:&[f64;8], k:i64) -> InformationResult
pub fn information_collision(p:&[f64;8], k:i64) -> InformationResult
pub fn information_min(p:&[f64;8], k:i64) -> InformationResult
pub fn information_kl(p:&[f64;8], q:&[f64;8], k:i64) -> InformationResult
pub fn information_js(p:&[f64;8], q:&[f64;8], k:i64) -> InformationResult
```

Each function declares Mut, Div, Panic effects. Input arrays are never mutated.

For normalized vectors P and Q:

- Shannon entropy: H(P) = -sum P_i ln(P_i), with 0 ln(0)=0.
- Collision entropy: H2(P) = -ln(sum P_i^2).
- Min entropy: Hinf(P) = -ln(max P_i).
- Kullback-Leibler divergence: D(P||Q) = sum P_i ln(P_i/Q_i).
- Jensen-Shannon divergence: J(P,Q) = [D(P||M)+D(Q||M)]/2, M=(P+Q)/2.

JS returns the divergence, not its square root metric. Convert a result to
bits explicitly by dividing by ln(2). These APIs do not silently change units.

## Input and status contract

k must lie in 1..8 and is checked before reading any array element. Only
the active prefix is inspected; poison values after k are irrelevant.
Active entries must be finite and nonnegative. Negative zero is accepted
as zero. Each vector's mass is accumulated with compensation.

Mass must be positive and within 1e-12 of one, measured in binary64
arithmetic. Accepted vectors are normalized internally using their actual
compensated mass. This tolerance permits small serialization or rounding
drift; it does not allow arbitrary counts or weights. Normalization and
subsequent arithmetic remain rounded binary64 operations.

| Status | Meaning |
|---|---|
| 0 | Available finite result |
| 1 | k outside 1..8 |
| 2 | Nonfinite active entry |
| 3 | Negative active entry |
| 4 | Nonpositive, nonfinite, or insufficiently normalized mass |
| 5 | KL support mismatch: some P_i>0 while Q_i=0 |
| 6 | Unexpected arithmetic failure |

Every ordinary error has value=0 and infinite=false. KL support mismatch
has status=5, value=0 and infinite=true: its payload is unavailable, and the
flag represents positive infinite divergence. The APIs do not encode that
case as a successful zero, discard the offending bin, or return a large
finite sentinel.

For a single vector, validation checks size, all active entries for
nonfiniteness, all active entries for negativity, then mass. Two-vector
functions completely validate P before validating Q, then check support.
Consequently an invalid P may be reported before an invalid Q. Read status
before using value; a numerical payload of zero alone is not a validity flag.

## Numerical methods

Shared checked natural logarithms supply range reduction.
`math::logarithmic::{log1p_checked,log1p_phi_checked}` owns the local
logarithmic increment and phi evaluations; the former private numerical
implementations are replaced by checked delegation. Helper failure maps to
the existing InformationResult arithmetic status6. All summations of probability
contributions use compensated accumulation. See [shared contract](LOGARITHMIC.md).

KL is accumulated through nonnegative generalized contributions

```text
g(p,q) = p ln(p/q) - p + q
```

whose sum equals normalized KL. p=q returns exact zero; p=0 contributes q.
When |p-q|<=q/4, use q*phi((p-q)/q), where

```text
phi(d) = (1+d)ln(1+d)-d
       = d^2/2 - d^3/6 + d^4/12 - ...
```

The shared factored local phi series avoids subtracting first-order terms near equality.
Away from equality, log differences avoid overflow in p/q. No blanket
negative-result clamp conceals failed arithmetic.

For close JS bins, a positive even-power series in
(P_i-Q_i)/(P_i+Q_i) uses the direct difference, avoiding roundoff from
first dividing both inputs and rounding their midpoint. Other JS bins scale
P_i and Q_i by their maximum, evaluate generalized terms against the resulting
mixture, then restore scale. This avoids
underflow in (P_i+Q_i)/2 at the smallest subnormal bin. Shannon, KL and JS accumulate
very small contributions separately in units of the smallest positive
normal binary64 value and apply that scale once after summation. This
retains tiny aggregate divergences that individual subnormal products
could otherwise round away.

Near a concentrated distribution, collision entropy uses positive cross
products 2*sum_(i<j) P_i P_j and shared checked log1p evaluation. Min entropy uses the
mass outside the selected largest bin with log1p. Shannon likewise uses
the tail mass to evaluate the dominant-bin logarithm. These avoid losing
small entropy solely because a normalized largest bin rounds to one.
The log1p helper returns its argument directly for magnitude below 1e-16,
where higher terms are below binary64 relative precision; this also avoids
halving a minimum-subnormal argument to zero before evaluation.
The method does not promise exact identities across all independently
rounded probability representations.

No heap or working probability arrays are allocated. The capacity is fixed
at eight; collision entropy performs at most 28 distinct pair products.
Other passes are bounded by that same eight-bin capacity.

## Numerical limits and validation boundary

Representable subnormal inputs and outputs are supported. Positive terms
or aggregate results may still round to zero at binary64 precision; status0
does not certify that every mathematically positive divergence remains
nonzero. Unlike the separate checked thermodynamic entropy API, this module
does not reject all individual underflowed contributions.

The implementation is not a correctly-rounded transcendental library or
a formal error-bound proof. Full-support KL can be numerically large and
asymmetric; JS is symmetric and bounded by ln(2) in exact arithmetic.
Finite precision must be considered in comparisons near those bounds.

An independent C++23 quadmath reference and native Sounio execution through
Forge are the intended acceptance route. A source edit alone is not evidence
of native execution, complete-corpus acceptance, or comparative performance.

## Shared-logarithm integration state

This refactor changes scalar-helper ownership, not public units, mass/support
validation or zero-bin conventions. Existing p0 contributions remain explicit.
Selected current-source typecheck is complete in the final seven-target
Slurm15825 closure; the math demonstration and count diagnostic both compiled
and executed, with actual SVG/XML/full-preview inspection. Frozen executable
source:27ea4ec6c2b97892e9e92cf982b0af90e2241023. The attested compiler was
reused only after exact unchanged bin/scripts/self-hosted/stdlib comparison.
See [execution receipt](../../docs/plans/LOGARITHMIC_EXECUTION_2026_10_21.md).

No tests or independent numerical oracle were added/run. In particular these
demonstrations do not establish full-vector probability concordance, boundary
coverage, ULP accuracy, complete-corpus acceptance or CI/merge. Historical
probability acceptance does not automatically validate every changed result.
The documentation update follows the frozen executable source.
