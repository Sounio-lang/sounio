# Dynamic fixed-bandwidth Gaussian KDE

`stats::dynamic_kde` evaluates a one-dimensional Gaussian kernel density
estimate on caller-supplied grid points. The bandwidth is supplied by the caller,
finite and strictly positive. The module does not select an optimal bandwidth,
assess sampling independence, validate a density model or construct confidence
intervals. Existing KDE modules are unchanged.

Focused current-source checks and native execution passed centrally at Slurm
job16118: the status fixture and the12-case generated independent C++23
reference both completed with exit0. This is bounded concordance, not a full
parameter-boundary corpus, complete calibration, CI or comparative performance
claim. Exact executable/compiler/receipt identities belong to the central
execution record; no execution was performed locally by the implementation agent.

## API and ownership

```sounio
gaussian_kde_dynamic(
 observations:&NumericBuffer, mask:&IntegerBuffer, grid:&NumericBuffer,
 h:f64, max_pairs:i64,
 density:&!NumericBuffer, log_density:&!NumericBuffer, flags:&!IntegerBuffer,
 work:&!NumericBuffer, flag_work:&!IntegerBuffer
)->DynamicKdeResult with Mut,Div,Panic
```

All borrowed owners must be distinct as required by the linear borrow contract.
No raw pointer views or owner copies are used. The kernel has no Alloc effect
and does not allocate a trajectory or sample history. Active input length is N;
grid length is G. Mask length must equal N; each mask is exactly0 or1.
A zero mask excludes that observation completely, including a NaN/Inf payload.
A mask1 observation must be finite; nonfinite selected values are errors rather
than silently filtered. Grid points must all be finite.

At least one observation must be selected. G=0 is a valid empty evaluation after
the complete input/options preflight. Output buffers require length>=G;
numeric staging requires length>=2G and integer staging length>=G. Multiplication
by2 is guarded before computing required sizes. Scratch and outputs beyond
their active prefixes are ignored.

Full input preflight precedes any scratch writes. All G results are staged
before the public prefixes are committed. Every failure leaves all public
output buffers unchanged. Scratch can change after preflight. There is no
separate partial-success prefix on error.

## Definition and numerics

For selected observations x_i and grid point q, the target is

```
f(q) = [sum_i exp(-0.5*((x_i-q)/h)^2)] / (N_selected*h*sqrt(2*pi)).
```

The implementation first finds the maximum finite log kernel at each grid
point, then recomputes the kernels for a compensated sum of exponentials
relative to that maximum. It combines the normalization in log space:

```
log f = max_log_kernel + log(sum(exp(kernel-max)))
        - log(N_selected) - log(h) - 0.5*log(2*pi).
```

This can recover a finite positive density when every unnormalized linear
kernel would round to zero. In particular, a tiny h can offset a very negative
log kernel before final exponentiation. The raw exp(kernel) is never required.

A finite raw difference uses one final IEEE division by h, accepting a rounded
zero standardized distance rather than dropping its Gaussian contribution. If the raw
difference of opposite finite endpoints overflows, the implementation forms
their half-difference and applies its factor2 through the existing balanced
scale helper.
It does not independently divide the endpoints by h. Nonzero standardized
distances that round to zero are allowed: their kernel is rounded to its center
value. The log kernel is computed as -(z*0.5)*z, avoiding an overflowing raw
z*z where the halved product is still finite.

If z or its negative squared log kernel is unrepresentable, that individual
contribution is treated as numerically negligible. If no finite log kernel
remains at a grid point, neither finite density nor finite log density is claimed.
This is a conservative staged-availability policy, not an iff certificate
for the exact normalized expression's representability. When a finite maximum exists, smaller kernels may still underflow in the
relative exponential sum, under ordinary binary64 rounding.

The implementation uses canonical checked logarithms and exponential, finite
precision compensated sums and an integer-to-binary64 selected-count conversion.
It is not a correctly rounded density/log-density implementation or a uniform
relative-error certificate. In a staged log-density expression near the finite
range boundary, availability can be conservative. No boundary renormalization
or truncated-kernel correction is applied.

## Availability and statuses

For each grid point, flags are a bit mask:

| Flags | Payload |
| --- | --- |
| 3 | Positive finite density and finite log density both available |
| 2 | Finite log density available; linear density unavailable, payload0 |
| 0 | Neither finite output available; both payloads0 |

Bit1 is positive finite linear-PDF availability; bit2 is finite log availability.
A density rounded to zero is deliberately not marked as an available positive
density. A linear overflow likewise leaves the finite log available. Zero
payloads are interpreted with flags; they are not assertions that the exact
Gaussian density is mathematically zero.

DynamicKdeResult exposes status,observations,selected,grid_size,pairs,
linear_available,log_available. The last two count grid points with the
corresponding availability bit. On any main failure, only the requested N/G
metadata are retained and the other metadata are zero. Status0 means the
evaluation completed under its per-grid flags; it does not require every
grid point to have both representations.

| Status | Meaning |
| --- | --- |
| 0 | Evaluation completed and all staged results committed |
| 1 | Invalid owners, lengths, bandwidth domain or negative pair budget |
| 2 | Nonfinite h, selected observation or grid point |
| 3 | No selected observations |
| 4 | Unexpected checked-log/sum arithmetic failure |
| 5 | Selected-observation × grid budget exceeded |
| 6 | Mask element outside {0,1} |

Precedence follows owner/size/options validation, observation/mask preflight in
input order, grid preflight, empty-selection check, then pair budget. An early
nonfinite selected value can precede a later invalid mask. There is no promise
to accumulate all input defects.

The budget bounds the logical pair count N_selected*G. Its product is checked
through division before multiplication, including i64 overflow. Evaluation
uses two streaming kernel passes per grid, plus mask scans; the budget is not
a timing deadline or count of every arithmetic operation. Workspace is O(G),
not O(N*G); work remains O(N*G), with no fixed8/256/1024 observation cap.

## Verification sources and open evidence

`tests/run-pass/stdlib_deep_kde_status.sio` checks invalid masks, selected versus
excluded poison, pair-budget rejection, transactional output preservation,
empty selection, G=0 and insufficient staging.

`tests/reference/deep_kde_reference.cpp` uses independent binary128 raw
Gaussian exponentials, sums and normalization for its bounded cases. It emits
native checks for 12 cases: ordinary mixture, duplicate observations, selected
mask, 1024 observations ×64 grid points, tiny bandwidth with40h/50h evaluation
whose linear kernels underflow, opposite maximum endpoints, a minimum-subnormal
difference divided by maximum h, an overflowing standardized distance, finite-log/linear
overflow, finite-log/linear underflow and unrepresentable-log availability.
The extreme unrepresentable-log case checks the flags; it does not manufacture
a finite log from a zero raw binary128 sum.

Generated relative density checks have no unit-sized absolute floor. A
two-minimum-subnormal allowance is explicit, and positive references require a
positive result. Log comparisons have relative tolerances for these nonzero
reference logarithms. An independent emitter and written checks are necessary
inputs to concordance, not proof that the native implementation passed.

Build the C++23 emitter on the authorized worker with -O2 -Wall -Wextra
-pedantic -lquadmath. Generate
`tests/run-pass/stdlib_deep_kde_reference.sio`, then typecheck/compile/execute
both fixtures through the central current-source Slurm route. No execution was
performed by the implementation agent.
