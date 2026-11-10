# Checked PCG-backed Poisson, Gamma and NB2 sampling

This additive module has focused central native evidence from Slurm job16101:
library/status/reference checks and the generated trace and moment programs
completed with exit0, as reported by the coordinating execution receipt.
This does not promote legacy distributions or certify full statistical
calibration, a complete boundary corpus, CI or comparative performance.

The module reuses `random::pcg64_core` for every RNG transition and uniform.
There is no second PRNG implementation in production. `count_rng(seed)` uses
the exact repository distributions constructor: initial state zero, fixed
increment high6364136223846793005/low1442695040888963407, one canonical step,
add the seed's 64-bit pattern to the low limb with carry, then another step.
A negative seed is not sign-extended into the high limb. This is the repository
default-stream convention, not a claim that its increment equals an external
library's explicit stream-number-zero convention.

## Public surface and budgets

All three sampling calls take `PcgState` by value and return its advanced state:

- `poisson_count_draw(rng,mean,max_uniforms,max_proposals)->CountDraw`.
- `gamma_unit_draw(rng,shape,max_uniforms,max_proposals)->GammaDraw`.
- `nb2_count_draw(rng,mean,alpha,max_uniforms,max_proposals)->CountDraw`.

The sampling functions have Mut,Div,Panic effects and no Alloc. No sample history,
normal cache, trajectory or heap owner is retained. Invalid requests preserve
the input RNG. Failures after sampling return the actual advanced RNG and work
counts; a caller must use that returned state to continue without replaying draws.

CountDraw exposes status,count,rng,uniforms,proposals,resolution_limited,
rate_underflow. GammaDraw exposes status,value,log_value,value_available,
log_available,rounded_underflow,rng,uniforms,proposals. Count is zero on error.
Unavailable Gamma payloads are zero; availability flags distinguish them from
an available numerical zero. A successful Gamma draw can expose a finite log
while its linear value is unavailable or rounded to zero.

Negative budgets or an even RNG increment-low limb return status1. Zero budgets
are valid: a positive request that requires work returns budget3 at the relevant
limit. A zero mean is exact count0 with no proposals/uniforms, after input
validation. The NB2 zero-mean case still validates alpha before returning.
One proposal means one log-product Knuth factor, one PTRS candidate or one MT
normal candidate. NB2 shares both budgets across its Gamma and Poisson phases.
Counters never reset between those phases.

Every canonical uniform draw, including an exact zero that is rejected, consumes
one uniform budget unit. The open-uniform helper accepts only 0<U<1. Exhaustion
returns status3; it never floors zero to an arbitrary positive constant.
No dangerous access or division relies on a short-circuit logical operator.

Statuses are:

| Status | Meaning |
| --- | --- |
| 0 | Requested count or Gamma draw available under the flags |
| 1 | Negative/invalid parameter, negative budget or invalid RNG increment parity |
| 2 | Nonfinite supplied parameter |
| 3 | Uniform or proposal budget exhausted |
| 4 | Arithmetic, checked logarithm/PMF or integer proposal unavailable |
| 5 | Supported parameter/rate range exceeded |

Configuration validation precedes parameter validation. Poisson then checks
finite mean, nonnegative mean and the upper mean bound. Gamma checks finite,
strictly positive shape and its shape bound. NB2 checks mean first, then finite
nonnegative alpha, then its positive-alpha reciprocal shape range.

## Poisson

The mean is finite in [0,2^52]. Returned counts are exact integers in [0,2^53].
For mean<30, Knuth's product test is evaluated as a compensated sum of log(U)
against -mean, avoiding exp(-mean) rounding to one. For mean>=30, Hörmann PTRS
uses the squeeze and rejection stages. Slow acceptance compares the transformed
log threshold against the checked Poisson log PMF from
`nb2_log_probability(k,mean,0)`, reusing its deviance/normalizer ownership.

PTRS separates the exact integer part of mean from its fractional part before
flooring the random offset. This avoids adding a fractional offset to a mean
near2^52 and losing it before the floor. Offset conversion is bounded before
casting; a nonrepresentable proposal returns status4 rather than being silently
rejected or clamped. A finite negative count candidate is an ordinary rejection.

The underlying uniform has exactly53 random fraction bits. This is a digital
sampler, not exact sampling from continuous real uniforms. A positive mean
below2^-53 sets resolution_limited=true. Such results must not be used as
evidence of calibrated rare-event probabilities. The flag is an explicit
minimum warning region, not a complete certificate of finite-grid tail accuracy.

## Gamma and NB2 mixing

Gamma has unit scale and finite shape in (0,2^53]. Marsaglia-Tsang sampling uses
Box-Muller from two fresh open uniforms (no cached second normal), a third
acceptance uniform when its cubic base is positive, and c=(1/3)/sqrt(d).
For shape<1, one extra open uniform supplies the boost in log space.

The slow log-acceptance expression is rewritten using log1pmx. For |t|<=1/8,
the cancelled quadratic remainder is evaluated as

```
A = -x^4/(27*d) * sum(k=0..31, (-t)^k/(k+4)).
```

Terms are compensated. Outside that interval the checked log1pmx expression
is used directly; nonfinite or unexpectedly positive bounds are unavailable,
with no epsilon clamp. The identity is real-arithmetic algebra; computed c,d
and the finite series introduce rounding/truncation. No correctly rounded
acceptance threshold, uniform ULP bound or complete calibration is certified.

Gamma returns its log before exponentiation. A rounded linear zero is explicit;
an overflowing linear value does not destroy a finite available log. In the
small-shape boost, a negative log that itself overflows below the finite range
is reported log-unavailable with an available rounded linear zero. There is no
positive floor.

NB2 uses the supplied mean/alpha parameterization Var(Y)=mean+alpha*mean^2.
Alpha=0, including negative zero, dispatches exactly to Poisson with identical
RNG stream, work counts and count. A positive alpha uses shape=1/alpha and
Gamma-Poisson mixing in LOG rate: logGamma+log(alpha)+log(mean).
If shape exceeds2^53, status5 is explicit; it is not approximated as Poisson.

Linear Gamma underflow alone does not zero the mixed rate: its finite log can
be rescaled by mean*alpha first. If the final mixed rate rounds to zero,
the returned count0 has rate_underflow=true. A Gamma log below the finite
range also returns that explicit outcome: finite input logarithmic rescaling
cannot restore such a value. Overflowing mixed rate is unavailable; a finite
rate above2^52 returns range5. Remaining positive rates use the Poisson sampler
and can set resolution_limited.

This is a supplied-parameter random-variable kernel. It does not estimate
dispersion, validate biological count assumptions, provide likelihood inference,
assess independence or certify arbitrary caller RNG state quality.

## Reference and evidence boundaries

`tests/reference/deep_count_sampling_reference.cpp` is the independent C++23
unsigned128/binary128 reference route; generated fixture and moment results
is paired with generated native trace and moment fixtures. The status fixture covers budgets,
no-RNG zero means, parameter/range states, Poisson-limit stream identity,
resolution warning and separate Gamma log/linear availability.

The focused executed trace scope includes five extreme seeds with eight PCG
steps each, ten Poisson, seven Gamma and five NB2 cases, a constructed exact-zero
uniform state, and Gamma linear underflow rescued by logarithmic NB2 mixing.
The independent C++ route uses unsigned128 state arithmetic and binary128
rejection/reference arithmetic; it does not call the production RNG helpers.

Three deterministic moment programs use20000 samples each: Poisson mean40,
unit-scale Gamma shape2.5 and NB2 mean12/alpha0.4. Native compensated sum and
sum-of-squares are compared with independent C++ moments and exact final RNG,
uniform and proposal counts. The coarse mean/variance sanity bounds are
8 standard errors and15% variance; C++/native moment concordance uses relative
1e-10 plus absolute1e-12. These checks are not goodness-of-fit testing, a rare
tail calibration certificate or proof that every supported parameter is accurate.
Exact generated-artifact/compiler identities belong to the central job receipt;
the implementation agent did not execute the programs locally.

Algorithm attributions:
[Hörmann, The transformed rejection method for generating Poisson random variables](https://research.wu.ac.at/files/18953249/document.pdf)
and [Marsaglia and Tsang, A simple method for generating gamma variables](https://doi.org/10.1145/358407.358414).
These sources identify the algorithms; they do not validate this implementation,
its finite-grid policy, parameter limits or compiler route.

## Reusable borrowed cursor and result slots

The additive hot-loop API keeps one `CountCursor` and one result slot outside
repeated draws. Construct with `count_cursor_new(rng)`, `count_draw_value()` or
`gamma_draw_value()`. `CountCursor` fields remain private; `CountDrawValue` and
`GammaDrawValue` expose the same per-draw status, counters, payload/availability
and resolution flags as the corresponding legacy result, **without an RNG**.

- `poisson_count_draw_into(&!cursor, mean, max_uniforms, max_proposals, &!slot)`
- `nb2_count_draw_into(&!cursor, mean, alpha, max_uniforms, max_proposals, &!slot)`
- `gamma_unit_draw_into(&!cursor, shape, max_uniforms, max_proposals, &!gamma_slot)`

Each returns an i64 status and overwrites every slot field on success or failure.
Counters and flags reset for each draw; RNG progress is preserved, including
consumed uniforms before failure. Gamma scratch fields reset before and after
every public call. A driver that imposes a batch budget must accumulate slot
counters and pass its remaining per-draw budgets explicitly. Do not interpret a
failed slot's count/value placeholders as samples. Preserve the resolution and
rate-underflow flags when deciding whether a simulation supports calibration.

`count_cursor_rng(&cursor)` makes an explicit RNG snapshot for a final result or
checkpoint. Avoid this getter per draw in a large loop. The existing by-value
APIs remain adapters for small calls and still return their complete RNG state;
their stream, acceptance order, budgets, counters, flags and failure semantics
are unchanged.

The shared `pcg_step_into(&!rng)` and `pcg_next_f64_into(&!rng)` use scalar-only
state updates with the same post-step XSL-RR output and top53-bit uniform rule.
They do not call tuple/struct-returning u128 helpers. The fixed legacy increment
and existing seed scheme are retained; this is not a new PRNG or stream selector.
Sampler open uniforms use the borrowed route, and Gamma updates cursor scratch
instead of allocating a `GammaValue` result per proposal. Validated scalar logs
call the same canonical ln_core through `math::pure::ln`, preserving checked
finite/domain handling without a CheckedLog record per log call. Remaining
checked log1p and likelihood helpers can still create managed results; this
change does not claim that every sampler arithmetic operation is allocation-free.

The C++23 reference now emits paired old/borrowed PCG vectors and a million-step
uniform program (`--emit-pcg-into`) with exact unsigned128 state/checksum checking
(`--check-pcg-into`). It can emit a legacy-route million-step control separately
(`--emit-pcg-legacy-million`). A million-draw Poisson0.01 program
(`--emit-count-into-million`) uses one cursor/slot and is checked independently by
a binary128 product-form Knuth route (`--check-count-into`); a legacy API control
is available with `--emit-count-legacy-million`. These programs retain the normal
compiler managed-handle limit. Source preparation is not an execution, memory or
speed claim; central current-source Slurm gates must measure both routes.

Cursor construction explicitly detaches all four RNG scalar fields into a fresh
PcgState. The constructor does not share the caller's mutable RNG record handle.
Likewise, count_cursor_rng returns a fresh scalar-field snapshot: subsequent
draws cannot mutate an earlier checkpoint, and mutating a checkpoint cannot
alter the private cursor. The direct-into fixture captures primitive field bits
before draws so shared-handle aliasing cannot masquerade as equal states. Legacy
by-value sampler inputs retain their previous immutability contract.
