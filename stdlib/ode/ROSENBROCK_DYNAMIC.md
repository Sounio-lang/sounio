# Dynamic Rosenbrock4 and accepted-state streaming

## Method and source custody

`ode::rosenbrock_dynamic` adapts the Rosenbrock4 stage engine from the official
Boost.Odeint1.83 header installed on the cpuops worker:
`/usr/include/boost/numeric/odeint/stepper/rosenbrock4.hpp`.
Observed SHA-256:
`db74ad85477f90c69791b3aa31488dde537c4c3e4318bbe30eae47f98d51fa91`.
The coefficient/stage equations retain attribution to Karsten Ahnert,
Mario Mulansky and Christoph Koke and the accompanying
[Boost Software License1.0](BOOST_LICENSE_1_0.txt).
The time-Jacobian coefficient d4 is explicitly corrected against the pinned
WRF/KPP RODAS4 source described below; it differs from the installed Boost
default. No identity with unrelated RODAS variants is asserted. Dense-output
coefficients are not used.

Focused current-source native validation passed centrally at Slurm job16118:
status and generated analytic/corrected-coefficient references completed with
exit0. This is not full corpus, CI, general error calibration or performance
acceptance. The installed header alone was insufficient: see the explicit d4
correction and failing original control below.
The old ODE APIs remain unchanged.

## Caller-owned storage

`ros4_workspace(n)` returns checked element requirements for four n-element
vectors(state,trial,error,derivative),8n stage values,two n*n matrix owners,
factor work of max(n*n,3n), and two n-element IntegerBuffer owners.
total_payload_bytes includes these numerical/pivot payloads, including state;
observer context is separately caller-defined. All compound arithmetic is
checked against the shared numerical payload limit before multiplication.
It is not a process-RSS estimate or a reservation.

The kernel has no Alloc effect and creates no NumericBuffer owner, raw view,
owner copy or resize. Managed temporary records still depend on the compiler
representation and must be covered by native resource evidence. A stage buffer
stores initial RHS,partial time derivative,g1..g5,error correction. The derivative
owner is reused after saving its earlier contents. Factor work is reused for
the six triangular solves after factorization. Callers retain and release owners.
Invalid shapes are rejected before callbacks. Active initial states must be
finite. Input n is discovered from the state owner, without a hidden64 cap.

## Callbacks

RHS:
`fn(t:f64,y:&NumericBuffer,dy:&!NumericBuffer)->i64 with Mut,Div,Panic`.

Explicit Jacobian:
`fn(t:f64,y:&NumericBuffer,J:&!NumericBuffer,dfdt:&!NumericBuffer)->i64
with Mut,Div,Panic`.

J has exactly n*n row-major elements and dfdt exactlyn elements.
dfdt is the PARTIAL time derivative at fixed state, not the total derivative
along the trajectory. Both outputs must be filled on every invocation.
Every destination is poisoned before invocation; incomplete writes/nonfinite
values are unavailable. Callback nonzero status is preserved as callback failure.
No finite-difference/autonomous-default substitution is silently performed.
Callback panic is not caught.

## Shared trial kernel

`ros4_step_dynamic(rhs,jacobian,state,trial,error,derivative,stages,matrix,
factor,factorwork,pivots,pivotwork,t,h,rcond)` returns Ros4StepResult.
state is borrowed immutably and never changed. trial/error are available only
when status0; scratch can be partially changed on failure. Status0 means a
completed trial, not acceptance under a tolerance.

One full trial makes6 RHS calls,1 Jacobian call,1 LU factorization and6 linear
solves. The matrix equation is evaluated equivalently as:

```
M = I - gamma*h*J
M*g_i = gamma*h*f_i + gamma*h*h*d_i*dfdt + gamma*sum(c_ij*g_j)
gamma = 0.25
```

This avoids constructing1/h. Balanced scalar products prevent premature
underflow/overflow of simple products; required nonzero terms that round
entirely to zero return arithmetic unavailability. Finite terms are accumulated
with compensation. Cancellation in the completed sums remains binary64 rounding,
not an exact-arithmetic certificate. This can conservatively reject some finite
mathematical problems.

LU is the shared checked dynamic module. Its pivot threshold is explicitly
supplied by rcond (zero selects that module's default). A factor/solve failure
is reported with its original nested linear_status; it is not silently accepted
or hidden as convergence. No automatic stiffness or Jacobian-quality assessment
is made.

## Pure adaptive driver

`solve_rosenbrock4_dynamic` has the same numerical owners/callbacks plus
t0,t_end,ODEOptions,rcond. Effects Mut,Div,Panic; IO is not imposed.

Adaptive acceptance uses a LASSQ-style weighted RMS of
error_i/(atol+rtol*max(abs(old_i),abs(trial_i))). All terms and denominators
must be finite. A trial is accepted only at error<=1, after which the entire
state vector is committed. Rejected trials leave the last accepted state intact.
The controller uses the embedded error order3 exponent-1/4 and supplied bounded
safety/growth/shrink parameters. Rejected retry factors are capped at0.9.
This is not Boost's PI-predictive controller or a promise of identical time meshes.

Both directions use positive step-size magnitudes. Budgets count attempts,
including rejected trials. Equal endpoints succeed after validation without
RHS/Jacobian calls. The driver checks representable directional advancement and
does not label a rounded t+h as a different requested endpoint. If a clipped
interval cannot round-trip that endpoint, it returns no-progress rather than
inventing a time. No dense interpolation, event localization, stiffness detector,
history or global-error guarantee is provided.

## Observed driver

`solve_rosenbrock4_observed` adds:
`observer:fn(t,&NumericBuffer,&!NumericBuffer)->i64 with IO,Mut,Div,Panic`
and a distinct caller-owned context buffer. Effects IO,Mut,Div,Panic.

The observer is called once on the validated initial state and once after each
accepted commit, never on rejected trials or invalid requests. It sees borrowed
read-only state; the context is mutable but cannot escape as a raw owner/view.
It can stream output or update bounded caller-managed summaries without a
trajectory allocation. A zero code continues; positive stops with7 unless the
actual endpoint was already reached(status0); negative returns callback failure6.
Accepted state remains available at a stop/failure. No separate cursor owner
or hidden storage is needed; the public single-trial kernel also permits a
caller-managed stepper/controller.

## Results and states

Step results expose n,stage/callback/linear failure metadata and actual evaluation
counts. Driver results additionally expose accepted time,target,next step,
steps,attempts,rejections,observer_calls,last_error. Counts include attempted
callbacks/linear solves, not only successful ones.

- 0 completed trial/kernel or completed endpoint/driver, as applicable.
- 1 attempt budget exhausted.
- 2 timestep/endpoint cannot advance in the stated represented arithmetic.
- 3 invalid buffers, initial values or options.
- 4 unavailable stage/error arithmetic or nonfinite callback payload.
- 6 explicit callback failure(kind1 RHS,2 Jacobian,3 observer).
- 7 observer requested an early clean stop.
- 8 checked factor/linear solve unavailable; inspect linear_status.

Numerical scratch/factors are not published as a separate reusable factorization
contract. On driver failure, state is only the last accepted prefix, not a complete
integration. No covariance, CI, p-value, positivity, energy conservation or
physical-model acceptance is implied.

## Verification sources

`tests/run-pass/stdlib_deep_ros4_status.sio` covers capacity/no-progress,
incomplete Jacobian output, callback rejection, state-preserving trial evaluation,
actual IO observation, early stop, observer failure, invalid-request silence and
attempt-budget exhaustion.

`tests/reference/deep_ros4_reference.cpp` calls the installed Boost
Rosenbrock4 engine with the explicitly corrected d4 coefficient, rather than
its unmodified default, to emit one-trial state/error expectations for a stiff
128-component diagonal system and a nonautonomous system with explicit dfdt.
Independent analytic exponential and quadratic solutions check small adaptive
integrations. The error-correction comparison has a declared absolute roundoff
allowance near cancellation zeros; nonzero trial states use relative tolerance.
Initial scope does not claim a full128-state adaptive cost result.

Compile/run the C++23 emitter on the authorized worker, recording Boost header/
version identity, generate `stdlib_deep_ros4_reference.sio`, then check/compile/
run through root-coordinated Slurm. Implementation agent did not execute it.

## Managed-runtime failure and accepted recovery

The initial generated reference exhausted the current generated-program handle
table (exit182) even though the status fixture passed; this is a runtime failure,
not numerical acceptance. Raising that capacity or reducing the reference
assertions is not the remedy used here. The adaptive drivers now hoist one
step-result record and two five-coefficient arrays outside the attempt loop,
write result metadata through an exclusive borrow, and borrow ODEOptions in
private helpers. The public step and solve signatures remain unchanged.
The private driver skips repeated buffer-shape validation only after the complete
initial validation; callbacks have no Alloc effect and cannot resize owners.
This layout-only recovery retained coefficient values, stage sums, tolerances
and acceptance rules; the separate d4 correction below is explicitly numerical.
The LU exact-zero Schur-update skip is separately documented in
../linalg/DYNAMIC_FACTORIZATION.md. The same generated reference subsequently completed without handle exhaustion
(exit1, empty stderr, MaxRSS1604KiB reported centrally at job16101), exposing
a separate nonautonomous numerical assertion failure. That intermediate run
was not a passing whole-program result. The separate d4 correction below then
passed the focused reference and controls at16118; the memory change alone
was insufficient to establish numerical correctness.

## Nonautonomous coefficient correction

Boost.Odeint1.83's installed default coefficient table has positive d4.
The WRF/KPP RODAS4 primary implementation instead specifies
d4=-0.3620000000000023e-01. The Sounio port now uses that negative sign;
all other coefficient values, stage operations, tolerances and budgets are
unchanged. This is an explicit correction to the initial upstream adaptation,
not bitwise concordance with the unmodified Boost default table.

The primary source is pinned at WRF commit
06d4240ae989cc3e50af412bb472df3d9048783c:
[WRF/KPP RODAS4 coefficient table](https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/chem/KPP/kpp/kpp-2.1/int/rosenbrock.f90#L1052-L1058).
Its complete raw source SHA256 is
44eb88bcbd91ab2f12f37ba71d314038d6a771f1a8750f5f8b8c2b317940fa2a.
Only the mathematical coefficient/sign and attribution are adopted from that
source; the production stage implementation remains the attributed Boost-based
port. No Fortran numerical execution was used as an oracle.

The original official Boost default is retained as a separate C++ control in
`deep_ros4_adaptive_reference.cpp`. The --pure-time mode checks y'=2t,
J=0, partial df/dt=2 over a step-halving sequence against the exact quadratic;
the unchanged positive-d4 default failed this control centrally. The corrected
modes use a coefficient subclass that sets ONLY d4 to the explicit negative
constant, avoiding a future upstream sign change being negated again. The red
control guards Boost version1.83 and a positive default d4; other versions or an
already corrected table are unavailable(exit77), not a claimed failing control.
The controls also compare direct time dependence with an autonomous
clock-augmented system.
An earlier attempted augmented control with an obsolete included source lacked
its new model variant and is invalid evidence; it is not cited as a failure.

The canonical generated reference now explicitly uses the corrected coefficient
subclass, tests an additional zero-initial-state nonautonomous trial and the
autonomous augmented-clock problem, and retains the original global1e-8 checks.
Corrected native concordance passed at Slurm job16118, including the original
analytic assertions and the added autonomous clock-augmentation case. The
paired C++ controls completed as expected: original --pure-time exit1,
--pure-time-corrected exit0, --corrected exit0 and --augmented exit0. Both
included sources were transferred together; the earlier obsolete include
closure was not used for this accepted control.
Agreement with the formerly wrong coefficient table alone cannot establish
nonautonomous accuracy. Even after these controls pass, adaptive tolerances are
local error controls, not a universal global-error guarantee.

## Focused resource observation

The original positive-d4 generated program, after the independent handle/borrow
fix, completed in140.05 seconds with exit1 and MaxRSS1604KiB. The corrected
reference family completed centrally in1.28 seconds with exit0 and
MaxRSS3072KiB at job16118, retaining the original checks and adding controls.
The failed-before result is preserved; it is not a valid numerical solution.
These are fixed-fixture observations, not a universal speedup, equal-work
benchmark, memory-improvement claim or guarantee for every stiff system.
The earlier handle-exhaustion failure and the later d4 accuracy failure are
distinct causes and must not be combined into one unsupported resource claim.

## Selected exact-rational proof

[formal/lean4/SounioRodas4TimeConsistency.lean](../../formal/lean4/SounioRodas4TimeConsistency.lean)
encodes the displayed decimal coefficient strings as exact rationals and proves
that the frozen fourth clock-stage expression is negative and lies between
-0.04 and-0.03; it also excludes the positive printed d4. It does not assert
exact equality to the independently rounded negative decimal coefficient.

The Mathlib-free Std proof passed Lean4.33 centrally at job16124 using explicit
unfolding and decide +kernel. The printed axiom dependencies are the standard
propext, Classical.choice and Quot.sound; no sorryAx was present in the accepted
build. The earlier failed proof build is not acceptance evidence.
This is a selected coefficient-consistency proof, not a formal proof of the
solver implementation, binary64 accuracy, convergence, stability or order.
