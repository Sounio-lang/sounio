<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-174.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-174.readme
-->

# SIO-174: bounded DP54 FSAL, quartic dense output and bracketed events

This package extends the real production Dormand–Prince 5(4) solver for active
dimension 1..64. The existing ode::checked64 import and Ode64Result field layout
are preserved; ode::checked_dp54 owns the implementation. Velocity Verlet's
function body is byte-identical to c21, SHA256
72c6fdb2472b0a465b28bd6092be49bb85cf18cfbf257f127194184a629ae301.
The ordinary dependency import alongside the public re-export is intentional:
the committed seed must load the new implementation for unchanged legacy callers.

## Public interface and ownership

All solves use the existing ODEOptions and callback type
fn(f64,&[f64;64],&![f64;64])->() with Mut,Div,Panic.
Every callback must write all active components. The solver poisons destinations
and checks finite outputs after every actual call; callback panics are not caught.
The RHS must be stationary during one solve. Changing equations requires a
terminal stop followed by a new solve, which starts a fresh derivative cache.

- solve_dp54_64 keeps the original arguments and result.
- solve_dp54_dense64 adds Dp54Limits and a mutable borrowed Dp54Dense64.
- solve_dp54_event64 adds the scalar event callback, Dp54Limits,
  Dp54EventOptions, a mutable dense record and mutable Dp54Event64 receipt.
  Event callback type: fn(f64,&[f64;64])->f64 with Mut,Div,Panic.
- dp54_dense_init/close and dp54_event_init/close construct and consume linear
  owners. Their fields are private. Borrow records for queries and consume each
  owner exactly once; copying a mutable owner is not an API operation.
- dp54_dense_domain returns status,n,t0,t1; dp54_dense_value and
  dp54_dense_derivative return status,value for a finite in-domain query and
  active component. Unavailable values have numeric payload zero.
- dp54_event_info returns a scalar snapshot including status,found,count,time,
  value,bracket_lo,bracket_hi,g_lo,g_hi,bracket_ready,n_event_eval,n_root_iter.
  dp54_event_state returns one active root-state component; it is unavailable
  until a root is found. A nonterminal receipt stores the latest matched root
  and the count of distinct recorded times, without aliasing the working state.
- dp54_is_terminal_event recognizes Ode64Result.status==5. Legacy status0
  continues to mean the actual requested endpoint was reached.

Dense and event records reset before every request, including invalid requests.
The event callback is evaluated at the initial state before any RHS call, even
for equal-time requests. An equal-time request with an included, undirected
initial zero can intentionally return terminal status5; otherwise completion0.

## Derivative reuse, represented time and bounded storage

Within a solve, the first complete trial uses seven actual RHS calls. Each later
complete trial uses six, including rejected retries: the valid derivative at
the unchanged base state survives rejection; only an accepted endpoint's k7
becomes the next base derivative. Counters measure calls actually made, including
a callback returning partial or nonfinite output. Denied budget calls do not
increment counters. A new solve never inherits an old cache. Nonterminal event
refinement keeps the accepted endpoint, while a terminal event replaces the
endpoint and exits, so no endpoint derivative is reused at a different state.

Each trial uses h=represented_endpoint-start and verifies start+h==endpoint.
Interior c2,c3,c4,c5 stage coordinates must be strictly ordered between endpoints;
the intentionally shared k6/k7 endpoint is retained. If a candidate exceeds the
caller cap through rounding, an arithmetic adjacent-value nudge is checked before
use. A final irreducible sliver can be redistributed into two represented spans
only when both have distinct stages and respect the caller cap. Neither a
requested nominal h nor a final time stamp substitutes for a performed span.
Arithmetic neighbor controls use exact dyadic inputs and C++23 std::nextafter on
both compilers, including signed zero, subnormal, normal and extreme cases.

Working arrays are seven derivative arrays and one trial array, each capacity64.
A dense record contains eight arrays of64, including preserved endpoint values
and slopes; an event receipt contains one state array. Storage is bounded, with
no trajectory accumulation and no new source-level arrays constructed inside the
step loop. Even the legacy wrapper creates a bounded dummy dense record.
These source properties do not establish universal allocator/backend behavior
or a memory reduction against the older solver.

## Quartic dense contract

The interpolant is Shampine's quartic continuous extension, using the published
DOPRI5 coefficients, not a linear mix of accepted endpoints. Value and derivative
queries require a ready record, finite time, valid component and time within the
last accepted interval. Original endpoint values and slopes are preserved.
A terminal interior root clips the public query interval while retaining the
original polynomial scale. Pure interpolation performs no RHS or event calls.

The primary algorithm source is Hairer/Wanner's
[DOPRI5 CDOPRI/CONTD5 implementation](https://www.unige.ch/~hairer/prog/nonstiff/dopri5.f);
their [software page](https://www.unige.ch/~hairer/software.html) describes the
fourth-order dense extension. No external implementation is executed as an
oracle. The C++23 supplement independently checks analytic constant, polynomial,
decay and growth solutions; exact rational coefficient/endpoint identities are
computational witnesses, not formal theorems or global accuracy proofs.

## Event policy and explicit budgets

Dp54EventOptions defaults to direction0, terminal=true, include_initial=true,
root_atol=1e-10. Direction is -1 falling, 0 either, +1 rising in increasing
physical time, including backward integration. root_atol must be finite and
positive; it bounds the returned time bracket width, not the residual or
trajectory/global error. Continuous stationary event functions are required for
a sign-change bracket to imply a root.

An undirected initial zero is recorded immediately when included. A directed
initial zero waits for the first accepted segment to establish orientation.
Set include_initial=false to suppress a known zero on restart. An exact endpoint
zero reached from a nonzero start value is recorded once when its direction
passes the filter; the following segment
does not duplicate the same time. Segments whose endpoint event values are both
zero are not resolved. An included undirected initial zero is still recorded
before stepping; a directed initial zero needs a nonzero endpoint to establish
orientation. Thus an undirected identically zero event records only its included
initial zero, while a directed identically zero event remains unresolved.
Strict endpoint sign changes trigger bisection on the actual quartic state.
The receipt exposes the actual ordered time bracket and endpoint event values.
An exact zero collapses the bracket. Otherwise refinement chooses a known
bracket endpoint with the smaller absolute event value once width<=root_atol.
This policy does not detect every grazing root or multiple crossings inside
one accepted segment. The grazing fixture explicitly preserves this limitation.

Dp54Limits defaults: max_rhs=7000000, max_event=2000000,
max_root_iterations=1000000. Valid ranges are 0..7000000 RHS,
0..10000000 event callbacks, 0..1000000 root iterations.
ODEOptions.max_steps is an attempt budget in 1..1000000.
Root iterations count actual midpoint event calls, not denied calls or pure
interpolation. Event counts include initial, accepted endpoint and root calls.
Receipts reset on reuse; caller-supplied stale found/count fields cannot prime
the cache or bypass evaluation.

| Status | Meaning |
|---|---|
| 0 | Actual requested endpoint completed |
| 1 | Attempt, RHS or event callback budget exhausted |
| 2 | Required represented step, retry or root midpoint cannot progress |
| 3 | Invalid request/configuration, or unavailable query |
| 4 | Nonfinite arithmetic/callback or partial active RHS write |
| 5 | Intentional terminal-event stop; root receipt is available |
| 6 | Root-iteration budget exhausted; bracket may be retained, root unavailable |

Invalid requests return cleared numeric payload and reset records before any
callback. Runtime refusals preserve the last wholly accepted solution state and
actual counters. A root failure does not commit the candidate step; its bracket
may remain diagnostic while found=false. Successful nonterminal roots do not
replace the accepted endpoint or invalidate a valid endpoint derivative.
A dense record left after a later RHS refusal describes its last accepted
interval; status and domain must be checked before every query.

## Qualification and provenance

Source basis: c21b0e139830ca4949f80cba62887a8922a972d7.
Two distinct compiler pins, retained immutably on the worker:

- Modular c21 build: a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3;
  CI artifact12580970584, run40144792765, witness121024346990, with the parent's
  full5295-entry input-catalog verification.
- Committed c21 lean_single seed:
  8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc.
  This is the committed seed, not a claim of a new seed rebuild.

The later main7ce advance changed only workflow/internal release documentation;
its numerical/compiler input trees are unchanged. The c21 binary remains labeled
as a c21 build. A source sync is a separate action from compiler provenance.

Immutable worker root:
 /scratch/sounio-stdlib-inference-scale-20261024/SIO174.
Full15/job17339 passes both engines: 304 values, seven dense nodes, 34 metadata
records, 22 event cases, and 12 invalid-request/no-callback/reset cases.
Unchanged legacy imports, status witnesses, 64-component decay, coupled oscillator
and Verlet reference callers also pass. The only changed old reference
expectation is the now-reused RHS evaluation count.

Guard14/job17330 preserves original c21 native controls on both compilers:
four fixed steps use28 RHS calls before and25 after; the old nonzero-start
nominal-span update differs from the actual-span analytic control by0.0591079.
The old cross-epoch case reports -1 despite its performed floating addition
reaching0. The revised case takes two represented steps and13 calls.
These are separate native before/after witnesses, not a relabeled nearby case.

Reachable compile-fail probes reject private dense/event writes and duplicate
linear-owner use on both pinned compilers. This qualifies these three concrete
forms; it does not certify all language privacy/linearity paths.

The independent C++23 reference uses decimal100 analytics plus exact binary64
decoding and rational coefficient checks. It verifies full counter agreement
with callback-side instrumentation, acceptance/rejection/restart, nonconstant
dense values/slopes, terminal/backward/directed/endpoint/initial/multiple-segment
events, real root brackets, all budgets, partial writes and nonfinite cases.
No theorem, stiff-method dense extension, unbounded/dynamic solver, universal
event detector, physical calibration or CI-wide pass is claimed.

Final18/job17357 passes the committed gate on both engines in27s total job time,
including C++ compilation. The core native fixture reports elapsed0.00s (timer
resolution, not zero cost), RSS1636KiB modular and3108KiB seed. All nine corrupted
record controls per engine are rejected. Final17/job17354 first verifies the
recognized error-pattern harness headers; final18 additionally removes trailing
fixture whitespace. Scientific equations, assertions and output records are
unchanged between these runs. Source/compiler/generated-fixture hashes
verify after the run. Compact text receipts preserve file byte counts and SHA256;
ELFs remain in the immutable worker packets rather than inflating Git history.

## Reproduction and compiler boundaries

Run scripts/stdlib/dp54_dense_events_gate.sh only on Linux UID1000 with stack
524288KiB. Set MADAROS_RAW_BIN to the absolute verified modular pin,
DP54_SEED_BIN to the absolute seed to qualify both engines, and
DP54_GATE_OUTPUT to a new absolute output directory. Omitting the seed qualifies
only modular behavior, recorded explicitly. The gate hashes source, compilers
and the generated legacy fixture, runs original and new native imported callers,
checks independent C++ concordance, rejects corrupted oracle records and checks
the reachable compile-fail forms. Per-native-fixture acceptance bounds are15s
and128MiB RSS; these are regression ceilings, not comparative benchmark claims.

A preserved minimal field-scalar-reference compiler witness in counter06 shows
bump(&!record.n) returning the old value under modular c21, while seed updates it.
The production counter uses a separate local and assigns the result after the
trial. This is semantically equivalent library bookkeeping, not a compiler fix.
Core05 preserves the original 25 actual callbacks versus returned0 discriminant.
The compiler owner was notified on the coordination bus, message
1792893232-1484169-12759; no compiler-owned source was changed.

Probe02 preserves the signed-zero nudge failure; probe03 preserves the seed's
near-minimum-normal decimal literal input mismatch. Probe04 uses exact dyadic
constructed inputs and qualifies76 input/neighbor bit pairs on both engines.
Full10's original tiny-time failure and time_diag11's C++ discriminant preserve
the mistaken distant-tail classification before the <=cap guard.
Full12/job17318 was an empty packet caused by a failed transfer and is explicitly
INVALID_STAGE, not numerical/compiler evidence. Historical packets and ELFs
remain untouched; no quota was changed or historical artifact removed.
