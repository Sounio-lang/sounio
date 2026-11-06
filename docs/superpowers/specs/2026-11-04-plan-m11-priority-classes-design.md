# M11: priority classes for preemption

## Direction

M9/M10 preempt the smallest-gen decoding slot whenever the
table is full and the queue is non-empty. Every session is
equal: a background batch job can evict an interactive decode,
and a low-priority arrival can trigger the eviction of a
high-priority decoder. M11 adds priority classes: the victim
is the lowest-priority decoder (tie-break: smallest gen, the
M9/M10 rule), and a queued arrival only triggers preemption
when it outranks the victim. No struct layout change; pump-path fn signatures widen by one
parallel array (M10 pattern): priority travels in a
session-indexed parallel array, reusing the ss/sos session
identity M10 built.

## Slice

1. Session priority: sprio[8], session-indexed, default 0,
   higher value = more important (harder to evict). Set at
   push time per session; last write wins. Live: sv_begin
   parses optional request JSON field priority (default 0);
   plain test pushes stay 0.
2. Victim choice in svb_admit_all: among eligible slots
   (off >= 0, vvalid == 0, gen > 0) pick lowest
   (sprio[sos[s]], gen[s]) lexicographically. sos[s] < 0
   (plain occupant) reads as priority 0.
3. Trigger guard: preempt only if the head entry outranks
   the victim. Head priority = sprio[q.ss[0]] for live
   entries (ss >= 0), 0 for plain/victim entries (ss < 0;
   a staged victim head keeps the M10 restore path, which
   never preempts). If no victim is outranked, return with
   the queue held (same wait path as queue-full today).
4. Threading: sprio rides beside every call site that
   already carries sos/t_sos (engine fns, pump, turn,
   live server session array). No SvQueue/SvSlot field;
one extra parallel-array param on admit/fwd/pump/turn.

## Verification

- New tests/run-pass/serve_prio.sio: 3 sessions over
  4 slots with mixed priorities. High-priority decoder
  survives a low-priority burst (low-prio arrivals wait,
  no preempt bit); low-priority decoder is evicted for a
  high-priority arrival and recomputes bit-exact (dual-ref
  discipline from serve_preempt: main cache never reset).
- CPU 6/6 (existing 5 + prio). Live 5860 lane: burst with
  priority 1 arrivals against priority-0 decoders
  finishes 5/5 with preempt lines; inverted run (prio-0
  burst vs prio-1 decoders) shows zero preempt lines and
  all 5 served after drain. Receipt in plan-m11.
- Negative control: with all priorities equal, victim
  order is exactly smallest-gen (M9/M10 behavior kept).

## Out of scope

Starvation bounds / aging, priority inheritance across
restore, more than one preempt per fwd call, cross-node
spill, non-pump path behavior.
