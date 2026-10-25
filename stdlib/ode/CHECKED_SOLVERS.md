# Checked ODE foundation

## Scope and availability

This increment repairs checked termination in `ode::solver` and adds
`ode::checked64::{solve_dp54_64,solve_verlet64}`. The latter is an explicitly
transitional fixed-capacity interface: active dimension1..64, borrowed fixed arrays,
typed callbacks and bounded storage. It is not the dynamic-buffer API.

Focused library/status checks and native status/generic witnesses passed through
the central Slurm gates, including job16055. Numerical reference acceptance and
compiler/source identities must be read from their exact central receipts; this
paragraph does not promote a focused status check to a full boundary corpus,
CI or a universal integration-error certificate. For the canonical caller-buffer
API see DYNAMIC_SOLVERS.md rather than treating the bounded64 interface as dynamic.

## Existing scalar and three-compartment solvers

`ODEOptions`, `SolverODESolution`, `ODESolutionVec3`, `State3` and
`SolverPKParams3` now expose public fields so callers can inspect/configure results.
Result layouts are unchanged. Existing analytical wrappers now explicitly declare
Mut,Div, matching their shared exponential callee; callers must propagate these effects.

Statuses:0 completed;1 attempt budget;2 time/step cannot advance;3 invalid request
or configuration;4 unavailable nonfinite arithmetic. `success` means status0.
Invalid requests clear numeric result payload. Arithmetic, progress and budget
failures retain the last completely accepted state with counters; partial results
are explicitly unavailable as completed integrations.

Fixed-step scalar RK4, backward Euler and PK3 require integer steps1..1000000.
Times and initial states must be finite; PK3 rates finite and nonnegative.
Signed time spans support forward/backward stepping. An equal-time request is
valid after request validation and performs zero steps. The time mesh is anchored
to the supplied endpoints; every internal time must make directional progress.
A nonfinite span, candidate state or intermediate step terminates explicitly.
No raw division by zero from invalid step counts precedes validation.

The PK3 RK4 peripheral update now uses k1+2*k2+2*k3+k4; the previous code repeated
k3 and dropped k2. A one-step transfer-chain fixture discriminates that defect.

The legacy scalar adaptive DP54 is still an exponential-decay demonstration
with fixed rate0.1, not a generic system API. It supports forward time only.
Its budget now counts attempts, so rejected trials cannot loop indefinitely.
Rejected trials at the minimum step or without a smaller representable retry
return2; status0 requires the actual endpoint. Step control now uses canonical
checked logarithm and exp rather than a short unscaled fifth-root Newton iteration.
Options require finite rtol>=0, atol>0, dt_init>=0, dt_min>0, dt_max>=dt_min,
safety in(0,1), min_shrink in(0,1), max_growth>=1 and attempts1..1000000.

## Transitional generic Dormand-Prince5(4)

`solve_dp54_64(rhs,y0,n,t0,t_end,options)->Ode64Result`
has Mut,Div,Panic effects. Callback type:
`fn(f64,&[f64;64],&![f64;64])->() with Mut,Div,Panic`.
The callback must write every active output component on every call; active
destinations are poisoned before invocation and checked for finite values.
Callbacks that panic are not caught or converted to a result status.
Inactive input/output cells are ignored by the solver, and result inactive cells
are zero. Callbacks must respect their declared active dimension and destination.

The result exposes status,n,t_final,steps,attempts,nfeval,nreject,state[64].
The original import and result layout remain available. The implementation now
resides in ode::checked_dp54, re-exported through ode::checked64. FSAL reuses the
accepted endpoint derivative and retains the base derivative across rejected
trials. For complete trials, nfeval=1+6*attempts; failed intermediate trials count
only calls actually made. Each new solve starts a fresh cache. RHS behavior must
remain stationary during a solve; changing equations requires a stop and new solve.

The new checked_dp54 module also exposes caller-owned linear Dp54Dense64 and
Dp54Event64 records, quartic dense value/derivative queries, explicit callback/root
budgets and bracketed sign-change events. Status5 is intentional terminal-event
success, recognized by dp54_is_terminal_event; it does not change the legacy
meaning of status0 (the actual requested endpoint). Status6 means root-iteration
budget exhaustion. See ../../docs/evidence/stdlib-SIO-174/README.md for the exact
API, zero/direction policy, status and qualification boundaries. There is no
trajectory history, stiffness detection or inferred global accuracy certificate.

Error control uses the maximum over active components of
abs(embedded_defect)/(atol+rtol*max(abs(old),abs(trial))). Every trial state,
callback result, tolerance denominator and normalized error must be finite.
A trial is accepted only when this norm<=1. Safety-scaled fifth-order control
is bounded by the supplied shrink/growth factors. Rejection cannot increase
the next trial. Finite representable endpoint advancement is checked each time.
Both directions are supported with positive step-size magnitudes. Each trial
uses its actual represented endpoint difference, requires ordered distinct
interior stage times and verifies that start+h reaches the candidate endpoint.
A final unresolved sliver may be redistributed into two representable steps,
each within the caller step cap; the solver never stamps a time it did not reach.

Unrepresentable intermediate coordinates, stage combinations or tolerance scales
return4 conservatively, even if a rearranged exact expression could be finite.
Floating-point cancellation in the embedded defect is not independently bounded.
Passing local tolerance is not a proof of a global error bound.

Working storage is seven64-component derivative arrays plus a64-component trial
array and the result state. Dense records contain eight64-component arrays; event
records contain one state array. The legacy wrapper constructs a bounded dummy
dense record too. There is no trajectory growth or new source-level array
construction inside the step loop; this is not a universal allocator/compiler
storage guarantee.

## Velocity Verlet

`solve_verlet64(acceleration,q0,v0,n,t0,t_end,n_steps)->Verlet64Result`
has the same typed callback shape. The callback computes acceleration from
time and position ONLY; velocity-dependent forces are outside this contract.
Result fields:status,n,t_final,steps,nfeval,position[64],velocity[64].
After validation, an equal-time request makes no callback. Otherwise there is
one initial acceleration call and one per accepted step, reusing the endpoint
acceleration. Candidate position and velocity are committed together only after
the complete step is finite. Input arrays are never mutated.

The implementation is kick-drift-kick velocity Verlet. Symplectic/energy behavior
requires appropriate physical equations; no energy-conservation, stiffness,
adaptive-step or model-assessment claim is made. Endpoint-anchored floating mesh
increments may differ slightly through rounding, so exact equal-step or exact
time-reversal claims are not made.

## Verification sources and commands

Status regressions:
`tests/run-pass/stdlib_deep_ode_status.sio`,
`tests/run-pass/stdlib_deep_ode_generic.sio`.
Independent C++23 analytic fixture emitter:
`tests/reference/deep_ode_reference.cpp` uses long-double exp/sin/cos, not
production RK coefficients. Its generated fixture covers all64 decay components,
a coupled oscillator and Verlet, with relative tolerances on nonzero expected
values. The direct PK regression uses a rational one-step transfer-chain result.

Run the emitter ONLY on the authorized cluster lane:
`g++ -std=c++23 -O2 -Wall -Wextra -pedantic tests/reference/deep_ode_reference.cpp -o EMITTER`,
then `EMITTER > tests/run-pass/stdlib_deep_ode_reference.sio`.
Compile/check/run the Sounio fixture through centrally coordinated Slurm.
Do not interpret source preparation or a C++ emitter exit0 as native agreement.
