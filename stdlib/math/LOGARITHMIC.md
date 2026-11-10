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

## Current-source execution evidence

Slurm15823 completed0:0 in7:13 after a fresh seed/modular Madaros build,
seven selected typechecks and two native demonstration executions.
The caption-only final source is27ea4ec6c2b97892e9e92cf982b0af90e2241023.
Final Slurm15825 completed0:0 in25s with seven selected typechecks and both
native examples compiled/executed successfully. The attested compiler was reused
only after an exact empty Git diff for bin/scripts/self-hosted/stdlib.
Compiler SHA256:
0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3.

| Artifact | SHA256 |
|---|---|
| Math demonstration ELF | c137e7474e1ca66b483c3451764cd9dfb202a913fe574d0607d5eedd8a129669 |
| Count diagnostic ELF | cd514796f69585d85b6ca85a91564d0b82322c78aaaa85cbafb010bc01ce1465 |
| Math complete stdout | 17018168d668ee34b4f3d1f48788adeeeb059690ae97167f0eaf461dc1a4e352 |
| Count complete stdout | c4638d32173f59987360db4d33fae97f903ca71e1fe884ba44e8ac9a6709f9bc |

Both complete HTML/SVG artifacts passed xmllint (exit0). The root converted each
SVG with sips to a1380×620 PNG and inspected the full previews. Math axes, legends,
captions and seven samples were visible; count panels, negative labels and units
were visible. These observations establish bounded artifact inspection, not
numerical agreement. The math example is SVG-only. The count example completes
its in-memory Canvas render, without pixel capture/comparison.

GNU time for one final fixed run of each child program recorded math elapsed0.00s
at centisecond resolution and maximum RSS1576KiB; count elapsed0.01s and maximum
RSS8768KiB. The displayed0.00s is below resolution, not zero runtime. These are
whole-demonstration observations, not scalar-kernel benchmarks or comparative
performance evidence.

Exact execution/source custody is in the
[execution receipt](../../docs/plans/LOGARITHMIC_EXECUTION_2026_10_21.md).
These documentation edits follow the frozen executable source commit.

No tests or independent numerical oracle were added/run. Independent numerical
concordance, ULP/error characterization, boundary coverage, prior-consumer corpus
and repository CI/merge remain pending. No correct-rounding, uniform ULP accuracy,
formal proof, universal representability or comparative performance claim follows
from the series length, successful typechecks or visible demonstrations.

## Reusable checked results

`math::pure::ln_checked_into` and this module's `log1p_checked_into`,
`log1pmx_checked_into` and `log1p_phi_checked_into` accept one caller-owned
`CheckedLog` and return its scalar status. Construct the slot once outside a
row/iteration loop. Every call replaces both status and value; failures leave
the corresponding unavailable status and positive-zero placeholder, including
failures after a successful nested logarithm. Tiny log1p inputs preserve their
signed zero; the quadratic kernels retain their documented positive-zero
normalization. The algorithms, domains and status values match the by-value
APIs, which remain as adapters owning one fresh result per invocation.

The into call chain uses primitive temporaries and borrowed storage. This is
a source property of these entrypoints, not a universal allocator or IEEE
accuracy guarantee. The native slot fixture and independent direct-identity
C++23 reference check selected values, raw bits, stale-state handling and
repeated-call behavior. Resource claims require a separate execution receipt.
