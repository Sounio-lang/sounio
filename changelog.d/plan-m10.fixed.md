# M10: turn load 1+2 - harden max-4 + session/slot decoupling via pq

## Direction

M9 proved preemption + recompute for 4 sessions pinned 1:1 to
4 slots. M10 scope 1+2: harden the max-4 bound under real load
and decouple sessions from slots through the pending queue, so
one live server holds 8 sessions over 4 pump slots without
aliasing a queued victim prompt.

## Slice

SvQueue gains ss (owning session, -1 plain); qpush takes sess;
qpurge drops a session entries and frees staged victim regions.
admit_all grows bank/ids/sos live branches: fresh-live admits
copy ids[ss] into the freed bank lane, restores replay spill as
in M9. blist is a slot scan (bn <= 4 structural); pump and turn
translate slot->session for every session array via
sess_of_slot (pdone/rslot stay slot-indexed); unbind on
finish/fail. Session arrays grow 4->8, SV_MAX_SESS 8, pump
slots stay 4. sv_begin parses only, then pushes
(off=-1, nid, pmax, ss) to pq, 429+reap when full.
Disconnect reaps pq entries plus bound slots.

Two engine root causes fixed in tree: refill overshoot bound
(xlen - k > 0, the spilled extra emission is the last output)
and replay off-by-one (xlen = q.n + q.xl - 1, nid + scnt - 1),
the latter validated by negative control (revert -> FAIL,
re-apply -> PASS).

Live bank-lane aliasing, found on GPU not CPU: a fresh-live
admit rewrote a free slot bank lane while a preempted victim
still queued, so the victim replayed another session prompt
(prompt-4 deterministic fork under 5-way burst). Fix is
svb_qvictim_first restore-priority rotation (loop top and
post-preempt, no signature changes): a staged victim always
admits ahead of queued fresh arrivals, so no fresh bank write
lands between a victim staging and its restore.

## Receipts 2026-11-04

- CPU: SERVEPREEMPT PASS via direct souc run (harness 30 s
  timeout is marginal for this test, not a failure).
  Rotation no-ops for ss=-1 test traffic, so pump/batch/paged
  are unaffected (5/5 green pre-rotation).
- RTX4000 (5860) job 22049 EXIT=0, m10e.elf, SmolLM2-135M,
  max_tokens 8: PREEMPT_LINES_PD1=4, PREEMPT_LINES_PD0=6;
  solo 5/5 byte-IDENTICAL pd0 vs pd1; burst-vs-solo
  text+usage 5/5 MATCH on both lanes (prompt 4 emits its own
  text again - the pre-fix fork is gone). Whole-body DIFF is
  envelope only (id/created differ per request by design).
- live-server device wiring (M5) and paged GPU attention
  unchanged and green through the same run.

## Notes

- Lockstep is defined as text+usage MATCH; envelope
  id/created always differ and are excluded.
- One preempt per fwd call (M9); rotation drains staged
  victims one per admit iteration, fresh arrivals wait.
- Out of scope: priority classes, cross-node spill,
  non-pump path behavior (signatures only).
