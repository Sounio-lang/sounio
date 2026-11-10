# Caller-owned dynamic ODE kernels

## Implementation contract

`ode::checked_dynamic` adds dynamic-length DP5(4) and velocity Verlet using
`collections::numeric_buffer::NumericBuffer`. This is a separate additive API;
the earlier bounded64 API and existing exponential/PK solvers are untouched.

Focused current-source native status and callback/owner execution passed
centrally. The independent128-state analytic DP54/Verlet fixture completed with
exit0 at job16059, checking every active component against its closed form.
This is executed callback/borrow evidence beyond typechecking, not a full
parameter-boundary corpus, comparative performance or assessed physical model.
The exact compiler/source/ELF identities remain part of the central receipts.

## Ownership and storage

No internal allocation, owner copy, raw-pointer view, resize or release occurs.
The caller creates distinct live owners and eventually closes them exactly once.
Every function has Mut,Div,Panic, without Alloc. Callback type:

```sounio
fn(f64,&NumericBuffer,&!NumericBuffer)->i64 with Mut,Div,Panic
```

Callbacks return0 after writing ALL components of their derivative destination;
any other code returns solver status6 and is preserved in callback_status.
The active destination is poisoned with NaN before every invocation. Incomplete
writes or nonfinite results return4. Callback panic is not caught.
The callback receives a destination of exactly the state dimension, so it can
discover n without globals, captured pointers or fixed64 assumptions.

DP requires state,trial,derivative lengths all equal n>0; workspace length>=7n.
Verlet additionally requires velocity lengthn and uses the same workspace contract
to allow scratch reuse. Construction statuses must be0. Dimensions are validated
using workspace_length/7>=n before any stage offset multiplication. Memory limits
are the caller's checked payload budgets, not a hidden dimension maximum.
There are no per-step or per-callback allocations in the solver contract.

Scratch owners are modified and unavailable as previous contents on all attempted
runs. Caller state commits only after a whole accepted trial. Verlet commits
position and velocity together only after both candidate vectors are finite.
On failure, state is the last fully accepted state; it is not a completed solution.

## APIs

`solve_dp54_dynamic(rhs,state,trial,derivative,work,t0,t_end,options)`
returns scalar-only DynamicOdeResult:status,n,t_final,steps,attempts,nfeval,nreject,
callback_status. options is the public `ode::solver::ODEOptions`.
The standard Dormand-Prince5(4) tableau, maximum weighted component error and
controller match the bounded implementation. Derivatives occupy seven n-element
rows in workspace; each callback writes the dedicated derivative owner, then the
validated result is copied to its row. No unsafe owner slicing is needed.
No FSAL reuse, history, dense interpolation, events, stiffness detection or
global-error certificate is provided.

`solve_verlet_dynamic(acceleration,position,velocity,trial,derivative,work,
t0,t_end,n_steps)` returns the same metadata. Acceleration depends on time and
POSITION ONLY. Velocity-dependent forces are outside this interface.
Workspace rows0,1,2 store previous acceleration, half velocity, trial velocity;
the remaining rows are unused. One initial acceleration plus one per attempted
trial; no initial callback for an equal-time request.
Symplectic/energy claims need appropriate equations and separate evidence.

## States and numerical limits

- 0 completed actual supplied endpoint.
- 1 trial-attempt budget exhausted.
- 2 no representable time progress, or rejected step cannot shrink.
- 3 invalid buffer shapes/statuses, options, initial finite values or step budget.
- 4 nonfinite/unrepresentable intermediate arithmetic or callback output.
- 6 explicit callback failure; callback_status records its code.

Invalid requests do not change state or scratch. After a valid request starts,
scratch may change even if no step is accepted. Equal endpoints succeed only
after validation. Forward and backward integration are supported with positive
step-size magnitudes. Fixed-step count and adaptive trial budgets are1..1000000.
Times and initial active coordinates must be finite; nonfinite spans return4.
Step minima/maxima, tolerances and controller parameters have the same validation
as [bounded checked solvers](CHECKED_SOLVERS.md).

Every stage state, callback output, embedded defect and tolerance denominator
must be finite. Some exact expressions could be representable after different
scaling, but this staged implementation conservatively reports4. Local error
control and finite values are not proofs of global integration error or model
validity. Endpoint-anchored Verlet increments can differ slightly through binary64
rounding; exact constant-step or exact reversibility is not claimed.

## Verification sources

`tests/run-pass/stdlib_deep_ode_dynamic_status.sio` exercises128 active values,
workspace rejection, incomplete callback writes, explicit callback failure,
forward/backward DP, fixed Verlet, budget status and explicit owner release.

`tests/reference/deep_ode_dynamic_reference.cpp` emits a separate native fixture
using independent C++23 long-double exp/cos/sin expected values for all128 decay
coordinates and all128 oscillator positions/velocities. No production tableaux
are copied into its expected-value calculation. Relative tolerances apply to
nonzero expected values, without a unit-sized absolute floor.

Compile/run emitter only on the authorized cluster lane, generating
`tests/run-pass/stdlib_deep_ode_dynamic_reference.sio`. Then typecheck, compile
and execute through root-coordinated Slurm with exact source/compiler identities.
These sources have not been executed by the implementation agent.
