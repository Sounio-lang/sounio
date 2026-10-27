# Accepted-state quantity drift and sampled event brackets

`ode::diagnostic_observer` records finite scalar observations supplied by a
caller. The demonstrated ROS4 callback evaluates quantity A+B and event A−0.5
on the immutable accepted state. Generic monitors do not evaluate a physical
law, assess its appropriateness, certify conservation, locate continuous roots
or provide dense interpolation. Sign changes imply a continuous root only with
separate continuity assumptions. An exactly zero sampled floating value is an
observed zero, not an exact-real event-time theorem.

## ABI, storage and ownership

The existing `solve_rosenbrock4_observed` callback receives time, immutable state
and one mutable NumericBuffer context. This module matches that ABI without
captured owners, raw-pointer views, internal allocation or per-observation
managed record returns. `diagnostic_observer_workspace(C)` checks and returns
16+6C elements, or−1. C is positive, indices/counters are at most2^53 and every
sum/product also respects the native signed byte-budget bound.

`diagnostic_observer_init(context,C,max_observations,direction)` requires the
exact queried length. It initializes all16 header cells and every inactive row
flag explicitly, regardless of prior payload contents. Record values are not
initialized or available until published. max_observations is0..2^53; zero is
an immediately exhausted work budget. Invalid initialization changes nothing.
All buffers remain caller-owned. A successful init is required before push.

Six SoA blocks hold time, quantity, event, absolute drift, relative drift and
flags. The16-cell header contains magic, capacity, observation budget, used,
event-direction filter, established time order, reference quantity, last time,
last event, event count, first bracket kind/left/right, sticky status and the
absolute/relative available counts. All finite/integer/range/header checks run
before indexed record access. The context is NOT an authenticated capability:
caller modification can fabricate plausible valid history. Keep it unmodified
between conforming calls except through these APIs.

`diagnostic_observer_push(context,time,quantity,event)` returns scalar status.
It preflights owner/header, existing sticky status, finite observation, capacity,
budget and strict time ordering before publishing any record. After a verified
header failure, the sole permitted change is sticky status cell13; the previous
records and other header fields remain intact. Malformed owners/headers return1
without writing even the latch. Once latched, pushes preserve that failure.
Initialize a new stream to resume; no lost observations are silently skipped.

The second observation establishes increasing or decreasing time order. Later
times must remain strictly ordered in that direction; duplicate times fail.
No time subtraction is needed, so opposite large finite times do not overflow a
comparison. Callback clients must propagate nonzero push status to stop the
ODE driver and inspect the monitor status after integration. The example uses
negative callback return and checks both driver and monitor status.

## Drift and event semantics

The first accepted quantity is the fixed reference. Absolute drift is
abs(quantity−reference), when that subtraction is finite. Relative drift is
absolute drift divided by abs(reference). A zero reference makes relative drift
unavailable even when absolute drift is zero. Opposite-sign subtraction overflow
can leave relative drift available through a half-difference ratio; a half-ratio
lost to zero is conservatively unavailable. Finite final division rounding to
zero is ordinary binary64 rounding. Unavailable drift is a zero placeholder with
its flag unset; it does not invalidate the finite observation or event bracket.
No blanket relative-accuracy or correctly-rounded certificate is claimed.

Flags:1 absolute available,2 relative available,4 event exactly zero at this
sample,8 accepted strict sign-change bracket. A constant zero event flags each
sample but does not repeatedly count the same zero episode. A first-sample zero
is recorded regardless of direction because no preceding direction exists.
Thereafter direction0 accepts either crossing, +1 negative→positive/zero and−1
positive→negative/zero, **along observation order**, including backward-time
integration. Consecutive nonzero signs are compared without multiplication.

The first detected event is latched as kind1 for an exact sampled zero with
left=right=sample time, or kind2 for opposite nonzero signs with the two observed
times in stream order. Times are not sorted and no midpoint is reported as a
root. All records remain available for reconstructing subsequent brackets;
the header counts detected events but retains only the first bracket.

Statuses:0 progress/available,1 owner/shape/header/configuration,2 nonfinite
observation,3 duplicate or inconsistent time order,4 capacity exhausted,
6 observation budget exhausted. Status5 is unused. Header/status getters
validate metadata before casting/indexing. `diagnostic_observer_get_into` copies
one available row to a reusable length6 slot, preserving it on invalid requests;
it can read the valid prefix after a later sticky failure. `info` constructs a
checkpoint snapshot; it is not a hot-loop API.

## Verification boundary

The source status fixture covers exact dyadic drift, zero reference, exact-zero
and sign-change distinction, forward/backward stream order, direction filtering,
nonfinite input, capacity/budget failure, malformed integral header, transactional
init and opposite-maximum finite observations with independent drift availability.
The C++23 reference checks all emitted synthetic rows. Its --figure mode checks
EVERY accepted observation against the independent analytic transfer solution
A(t)=exp(−2t), B(t)=1−A(t), quantity1 and the sampled bracket containing log(2)/2
within the declared solver tolerance. The bracket itself is never replaced by
that analytic event time. It also checks document completion and marker count.

`examples/viz_dynamic_ode_diagnostics.sio` uses the actual ROS4 accepted-state
observer and a bounded4096-record context, then renders event samples and
absolute quantity drift through the shared scientific document layer. The right
axis range is a declared display range, not a floor applied to measured drift.
The central current-source focused check07 completed this fixture and the actual
ODE example with typecheck, native compilation, execution and markers all exit0.
The C++ synthetic-record and actual-example analytic/structural concordances both
exited0. These checks cover the emitted accepted-state samples, declared analytic
tolerances, sampled bracket, provenance and document completion; they do not
supply dense output or prove numerical conservation. The later full02 gate is
still running as of this update. No independent image inspection of this ODE
figure, physical-output acceptance or Canvas pixel comparison is claimed here.

The coordinator maintains artifact identities and the precise acceptance scope
in the [evidence ledger](../../docs/evidence/stdlib-inference-document-2026-10-23/README.md).
Focused execution is separate from full-suite, inferential or human acceptance.
