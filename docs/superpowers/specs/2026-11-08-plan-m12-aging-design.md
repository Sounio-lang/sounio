# M12: a hold bound against priority starvation

## Direction

M11 added priority classes: the victim is the lowest-priority
decoder and a queued head only preempts when it outranks the
victim. That creates unbounded starvation: a low-priority head
waits forever while higher-priority decoders hold the table.
The first M12 cut tried linear aging (effective priority =
base + waited / DIV), but the trace killed it: with DIV = 1 a
gap-1 waiter escalates in one pass and breaks the M11 negative
control (mode 1: a priority-0 arrival must keep waiting while
priority-1 decoders drain), while a gap-2 waiter needs exactly
that speed to beat the drain. No single linear rate serves
both; the scenarios are structurally identical but demand
opposite outcomes. M12 v2 is an absolute bound instead: a
queued head held `SVB_MAX_HOLD` passes preempts regardless of
class. Simple, predictable ("every waiter is admitted within
K passes of reaching head"), no tuning of rate against gaps.
No struct layout change; the wait array rides the M10/M11
parallel-array pattern beside `sprio`/`sos`,
session-indexed.

## Slice

1. Wait array: `swait[8]`, session-indexed, starts 0.
   `svb_admit_all` ages every session holding a queued entry
   once per pass at entry (capped at `SVB_AGE_MAX` = 2^20, no
   overflow). Admitted heads reset to 0 before `svb_qshift`,
   so entry-aging nets to aging only passes spent queued, and
   each session counts once per pass however many entries it
   holds. Plain entries (`ss < 0`) never age.
2. Guard: M11 logic unchanged (victim = lowest base
   `(priority, gen)`; head that does not outrank waits),
   except a head held `SVB_MAX_HOLD` (32) passes preempts
   anyway. Staged-victim heads keep the M10 restore path,
   which never preempts; plain heads keep the M10 wait path.
   Bounds analysis: mode-1's waiter drains by ~turn 12
   (decoders capped at rmax 6) << 32, so the M11 control is
   untouched; mode-3's waiter binds the bound at turn 32 <<
   its decoders' 60-gen horizon (probe: no early EOS in 240
   decode steps on the tiny model), so the escalation is
   deterministic.
3. Threading: `swait` rides beside every call site that
   already carries `sprio` (`svb_admit_all`, `svb_pump_fwd`,
   `svb_pump`, `sv_pump_turn`, live server array, all five
   run-pass callers). `svb_qpurge`/`sv_sess_drop` reset the
   reaped session; a fresh live push resets its slot
   ("last write wins" extended from `sprio`). No new effects.
4. Constants live with the `SVB_` block: `SVB_MAX_HOLD`,
   `SVB_AGE_MAX`.

## Verification

- `tests/run-pass/serve_prio.sio` mode 3: 4 long
  (rmax 60) priority-2 decoders fill the table; a priority-0
  waiter arrives at turn 1 and forced-preempts at the bound.
  Assertion is causal, not timing: the waiter must first
  appear in a slot in the same turn that slot's preempt bit
  fires (free-slot admission sets no bit, so pre-M12 code
  fails deterministically). All five finish bit-exact under
  the dual-ref discipline; modes 0-2 unchanged (their heads
  always outrank-or-tie, so the bound never binds).
- CPU: existing lanes plus mode 3. Live 5860 lane: pending.
- Side effect of threading: `serve_dbg.sio` (broken since
  M11, argument count) compiles again.

## Out of scope

Mid-queue promotion (the bound helps the head only; a waiter
behind sustained higher-priority arrivals still advances one
slot per pass), priority inheritance across restore (reset on
admit is the documented rule), more than one preempt per fwd
call, cross-node spill, non-pump path behavior.
