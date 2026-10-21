# Shared checked logarithmic increments and remainders

Module: `math::logarithmic`. This module owns checked evaluation of three
related scalar functions used by probability, channel-capacity and Poisson
consumers. It depends only on `math::pure` and reuses its existing CheckedLog
record. The public `math::pure` API is unchanged.

## API and common domain

```sounio
pub fn log1p_checked(x:f64)->CheckedLog with Mut,Div
pub fn log1pmx_checked(x:f64)->CheckedLog with Mut,Div
pub fn log1p_phi_checked(x:f64)->CheckedLog with Mut,Div
```

The functions evaluate, respectively:

    log1p(x)     = ln(1+x)
    log1pmx(x)   = ln(1+x)-x
    log1p_phi(x) = (1+x)*ln(1+x)-x

CheckedLog contains status:i64 and value:f64. The scalar APIs require only
Mut/Div; existing consumers retain Panic for their other operations. All three APIs use the same
finite-input domain x>-1. Although phi has a finite continuous limit at-1,
this API does not expose that extension. It does not silently evaluate a
logarithm at or below its domain boundary.

| Status | Meaning | Payload |
|---|---|---|
| 0 | Available finite evaluation | value available |
| 1 | Finite x<=-1 | value0 |
| 2 | Nonfinite input or unavailable arithmetic, including an invalid expected sign | value0 |

Read status before value. Zero alone does not distinguish a legitimate result
from an unavailable payload. Nonfinite inputs are unavailable rather than
successful infinities. A finite input does not guarantee every requested
remainder is representable.

## log1p arithmetic

For abs(x)<1e-16, log1p_checked returns x directly. This preserves the input
signed zero and subnormal argument rather than halving it away in an
intermediate. It is an explicit binary64 branch, not an exact real-arithmetic
identity for every nonzero x.

For abs(x)<=0.5 outside that tiny branch, use the atanh identity with
t=x/(2+x) and32 compensated series terms. The bounded local series avoids
forming1+x when its small increment would be lost.

Outside that interval, form the rounded sum u=1+x and retain its low part:

    x>1: c=1-(u-x)
    otherwise: c=x-(u-1)

Then evaluate ln_checked(u)+c/u. Domain, finitude and checked-log availability
are enforced. This correction addresses rounded addition; it is not an
independent correctly-rounded logarithm implementation or a formal error bound.

## Cancellation-safe local remainders

The small remainders should not be obtained by subtracting a first-order
term from an independently rounded logarithm near zero. For abs(x)<=0.25,
the implementation uses bounded compensated series in factored form:

    result = x * (x * factor)

For log1pmx, the initial factor term is-1/2; its recurrence is
-term*x*n/(n+1), beginning at n=2. For phi, the initial factor term is+1/2;
the recurrence is -term*x*(n-1)/(n+1), beginning at n=2. The bounded update
range ends at n=31: the polynomial has degree32 and31 terms from degree2
through degree32, not32 quadratic-series terms. This preserves the local quadratic scale without forming
x*x before multiplication by the factor.

Outside the local interval, the functions use the shared checked log1p result
and their defining subtraction/formula, followed by finite/sign guards.
The expected signs are log1pmx<=0 and phi>=0; a violated sign is unavailable
status2. There is no epsilon-sized negative/positive clamp hiding a sign error.

Exact-zero and rounded-underflow quadratic outputs are normalized to positive
zero. This differs deliberately from log1p's preservation of signed input zero.
The functions do not reject every underflowed contribution or guarantee that
all mathematically positive quadratic values remain nonzero.

## Shared ownership and consumers

- `math::probability` delegates its local log1p/phi work to this module;
  shared-helper failures become its existing arithmetic status6.
- `math::channels` delegates AWGN log1p evaluation to this module and keeps
  its existing status6 arithmetic policy, including positive-capacity
  unrepresentability.
- `stats::regression::poisson` delegates its local logarithmic phi evaluation
  to this module; unavailable helper arithmetic becomes evaluation status4,
  subject to the solver's existing trial-rejection and failure handling.

The consumers keep their public APIs, units, input validation and zero/support
special cases. Probability's zero-mass contributions and Poisson's zero-count
deviance are not replaced by an invalid logarithm. The shared module owns these
scalar formulas; it does not take ownership of complete KL/JS algorithms,
channel models, Poisson iteration or compensated outer aggregations. The
specialized JSD/BSC even series and Poisson exponential objective-change
remainder remain separate and unchanged.

## Sources and validation boundary

The [fdlibm log1p source discussion](https://netlib.org/fdlibm/s_log1p.c)
describes correcting rounded1+x before logarithmic evaluation. The official
[Boost log1p documentation](https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/powers/log1p.html)
explains why small increments require dedicated evaluation.
These references motivate the numerical problem and reduction ideas.
No implementation code or their accuracy guarantees are copied or claimed here.

Source preparation and consumer integration are not executable validation.
Initial typecheck, independent numerical concordance, native demonstrations,
figure inspection and any prior-consumer corpus checks remain pending.
No tests have been added or run for this checkpoint. In particular no
correct-rounding, uniform ULP accuracy, formal proof, universal representability
or comparative performance claim follows from the series length.
