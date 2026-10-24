# Caller-owned count-fit work counters

`CountFitTraceSlot` is opaque scalar state. `count_fit_trace_slot()` starts all
counters at0 with per-metric limit i64MAX. `count_fit_trace_bounded(limit)` selects
an explicit nonnegative per-metric limit for bounded measurement and exhaustion
controls. A negative limit creates invalid status2; reset does not repair an
invalid configuration. No solver inputs, RNG or owners are stored in this slot.

`count_fit_trace_note(&!slot,metric)` increments one selected counter and returns
scalar status. IDs1..10 are:

| ID | Counter | Unit intended at the actual hook |
|---|---|---|
| 1 | fit_calls | CountGLM fit-engine invocations, including failed validation |
| 2 | irls_entries | IRLS loop bodies actually entered, including terminal convergence iteration |
| 3 | initial_rank_qr_calls | Calls of QR on the original design for the initial rank check |
| 4 | weighted_qr_calls | Calls of the weighted QR solver within IRLS |
| 5 | profile_point_evaluations | Entries to the scalar-alpha profile point evaluator |
| 6 | paired_rank_calls | Checked paired comparisons actually invoked for candidate ranking |
| 7 | full_logpmf_row_calls | Actual scalar full-PMF calls for profile points and paired row comparisons; excludes sampler proposal-likelihood checks |
| 8 | null_cold_searches | Cold null profile-search calls (observed and unseeded replica routes) |
| 9 | replica_null_seeded_searches | Seeded null profile-search calls on simulated replicates |
| 10 | alternative_seeded_searches | Seeded alternative profile-search calls |

These are attempted call/entry units, not completed factorizations, successful
fits, floating operations or quantities inferred from `iterations*n`. In
particular initial-rank and weighted QR are separate units. The full-PMF metric counts direct point-evaluator calls and a private scalar
counter hooked at each actual child PMF call in paired comparisons, including
failed rows. It is not computed as2*n. The counter does not cover proposal PMFs
inside a sampler or unrelated standalone callers.

Status0 means the recorded prefix is available. Status1 latches when the selected
counter has reached its configured limit, before the increment could overflow.
Status2 latches an invalid metric ID/control. A failed note changes no count.
All later notes retain the latched error. `count_fit_trace_reset` clears counts
and operational errors while preserving the limit/configuration. Scalar getters
return counts without constructing records; an invalid getter ID returns -1.
`count_fit_trace_snapshot` constructs a detached checkpoint with status, limit
and all ten values. Read status before treating a snapshot as complete work.

The note path has no allocation, array indexing, division or panic effect.
Only factories/checkpoint snapshots construct records. The paired row wrapper
does not create a TraceSlot: it reuses private i64 call counters. Aggregate
traced calls transfer those actual counts using checked count_fit_trace_add.
Batch transfer is checked at its boundary, so the limit is a retention bound,
not a promise to stop already performed aggregate math. An incomplete batch
latches status1 rather than presenting a wrapped count. Like the existing
numerical slots, it is not an authenticated capability against unsafe memory
mutation. A traced numerical call must expose trace exhaustion as unavailable;
it must not wrap the counter, panic or claim complete measurements.

## Additive traced numerical entries

The following checked entries append a mutable caller trace to their ordinary
arguments and accumulate without implicit reset:

- dynamic_count_fit_traced
- dynamic_dispersion_fit_traced
- dynamic_dispersion_fit_seeded_traced
- dynamic_count_profile_lrt_bootstrap_nested_traced

Legacy signatures and public fit/profile/bootstrap result layouts are unchanged.
Legacy wrappers own one constant local trace per invocation with tracing disabled;
no trace factory is added to an IRLS or likelihood-row loop. The same numerical
engines execute. A traced CountGLM failure uses unavailable status4; profile
point failures preserve stage7/trace status; bootstrap trace failure is unavailable
status4/stage10, retaining its RNG checkpoint and zero p/MC placeholders. A
bounded-zero bootstrap trace stops before random draws. Read the separate trace
status to distinguish incomplete measurement from ordinary numerical failure.

The unit fixture covers all IDs, bounded overflow, invalid controls, explicit
reset, detached snapshots and one million notes. The full B2 traced fixture uses
the original grouped dataset/options/seed, checks numerical result bits and RNG
against an untraced invocation, repeats the traced invocation to verify cumulative
counters, and exercises bounded-zero failure before RNG. Source is prepared and
wired; central typecheck/native/concordance/resource validation is still pending
for this instrumentation checkpoint. No measured counts or speedup are claimed
from the source alone.

## Reusing the trace when tracing is disabled

The additive `dynamic_count_fit_with_trace_control(..., enabled, trace)` entry
uses the same fully checked numerical engine as the legacy and traced fits.
With `enabled=false` it neither reads nor mutates the trace, including an
already unavailable trace. Buffer, shape, count, finite, numerical, iteration
and convergence checks remain active. It does not construct a trace slot.
`dp_evaluate` passes the profile's existing slot and its control flag to this
entry, eliminating the former trace constructor at each profile point. Legacy
public wrappers keep their signatures and their one local slot per wrapper call.
There is no unchecked/trusted-input entry or cached QR/weights.

The new control fixture retains the complete inputs and assertions from the
canonical CountGLM fixture, uses a configured-unavailable trace with tracing
disabled and checks that its status and counts stay unchanged after each fit.
Its numerical output is checked by the same independent CountGLM and increment
comparators. This candidate's focused execution and resource acceptance are
recorded separately from the frozen instrumentation baseline.

Focused Slurm16931 completed checks, native execution and markers for eight
fixtures, including the complete disabled-control and legacy CountGLM cases,
both B2 paths, trace controls and ranking metadata/extremes. Both CountGLM
outputs passed the independent CountGLM and256-case increment comparators;
both B2 outputs and both paired-likelihood outputs passed their comparators.
The source inventory remained unchanged. This closes selected functional
concordance for the control entry, not a claim about its performance. The
isolated B31 resource pair is recorded below.

Isolated factory pilot16932 used byte-identical nested B31 drivers and the same
compiler92d, comparing fast-local ranking before/after the trace-control entry.
The C++ analyzer found identical scientific output and reported90.74s versus
90.69s, RSS58852KiB versus58848KiB, ratio1.00055132870217223509. This single pair
does not demonstrate a useful performance improvement. The change is supported
as removal of a needless constructor from the callgraph, not as acceleration or
measured memory superiority. This pair does not use original full02 as baseline
and does not assess the cumulative2x target.

## Optional normalizer-cache trace extension (candidate)

The public snapshot and metric IDs1..10 remain unchanged. Additional named
getters report `normalizer_evaluations`, `normalizer_cache_hits` and
`normalizer_cache_bypasses`. Their hooks record actual events in the opt-in
cached full-PMF profile route. Evaluation means an actual top-level saturated-
normalizer call, including a failed call; a hit means a successful lookup; bypass
means the count is outside0..63. A bypass also performs a normalizer evaluation.
They are not inferred from row count, profile evaluations or the old PMF counter.
Other PMF consumers remain outside these cache-counter units.

The shared per-metric limit applies independently to the three new counters.
Reset clears them, negative additions or invalid event kinds latch status2, and
limit/overflow failures latch status1 without wrap. The original public snapshot
still describes its original ten metrics, and metric11 remains invalid. New
cache getters expose the recorded prefix under the same trace status. No trace
factory is introduced inside a scalar PMF or per-row loop. Selected functional acceptance for this extension is recorded below;
broader qualification and comparative resource evidence remain separate.

Slurm16936 passed the cache-specific counter reset, independent limits,
invalid-event and maximum-i64 overflow controls, together with the unchanged
trace unit and complete off/on B2 assertions. On its fixed B2 workload, the
new named getters recorded3264 normalizer evaluations,1344 hits and0 bypasses;
the original ten metrics remained288,307,288,307,288,235,6336,1,2,3. These are
recorded call/lookup events in the declared cached-profile scope. They are not
counts for sampler PMFs, uncached ranking fallbacks or an entire B199 campaign.
