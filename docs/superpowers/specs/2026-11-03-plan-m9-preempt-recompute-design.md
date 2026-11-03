# Plan M9: preemption + recompute in the serve pump — design

## Why

paged_grow caps at 64 blocks; paged_append past that drops the
token silently and logits continue over truncated KV. Any
sequence past 64*B tokens degrades with no signal. Own it:
visible overflow, eviction with a victim policy, and exact
recompute from a spill log.

## Current

4 slots, queue cap SVB_QCAP, admit-all on free slots, slots run
to EOS/maxgen. No eviction, no spill, no overflow signal.
Legacy single/batch forward paths keep silent-drop (gap, out
of scope); the signal lands in the pump path only.

## M9 changes

1. Signal: paged_full(pk) predicate in paged_kv.sio (no
   signature change, probes untouched). runs_impl checks per
   row; overflow marks the slot pdone=3 (truncated) with
   receipt. Admission guard: nid + maxgen past capacity is
   refused at admit with the same code.
2. Preemption: admit with zero free slots and a waiting queue
   evicts the active slot with smallest gen (cheapest to
   rebuild). Victim tables freed, state saved, requeued head.
3. Recompute: per-slot spill (generated tokens, 1024 each in
   a separate array) written at sample time. Readmitted slots
   replay prompt (bank) + spill (silent rows, greedy sample
   checked against the log, mismatch is a hard fail), then
   decode resumes with saved last. No struct change: parallel
   spill/xlen arrays beside slots.
4. Test: 1100-token prompt with decoders running forces
   overflow; 5th arrival over 4 busy slots forces preemption.
   Dual lockstep vs non-preempted reference, byte-identical
   streams.

## Verification

CPU 4/4 existing + new preempt test PASS; RTX4000 relay
same; live serve untouched (flag paths unchanged).

Out of scope: legacy single/batch overflow signal, priority
classes, cross-node spill.

## Build diary 2026-11-03 (as built)

- Signal rides on meta[3], not a paged_full predicate: build_rows
  sets bit s when a prefill/decode row hits cap (64 blocks), and
  the turn maps it to pdone=3 (receipt truncated). No admit-time
  nid+maxgen guard; no paged_kv.sio signature change.
- Victim eligibility adds vvalid==0 (a slot whose vspill region
  is reserved for a queued victim cannot be victimized) and
  gen>0. One preempt per fwd call. Queue-full (>= SVB_QCAP)
  refuses preemption; all-prefill occupancy waits.
- Recompute rows are normal engine rows (not silent): the test
  replays them against fresh reference caches and compares every
  logit (bit-exact), with the seed row detected by rowtok>=0 and
  the pass restart detected by rpos < fed.
- Whole-path svb_pump also records the spill log (same stanza
  as adv); otherwise preemption through the whole path would
  lose generated tokens on recompute.
- Test slot identity tracks the bank head token per slot
  (sloth); the old slotseq-only tracking misattributes rows
  after a slot changes occupant.

## Bugs caught by build+run

- meta[3]=0 was placed after admit_all+build_rows, wiping the
  overflow and victim bits. Moved before admit_all. Without
  this fix no M9 signal ever reaches the caller.
- svb_pump (whole path) never recorded spill tokens; added the
  recording loop after svb_sample_rows.

## Receipts

- CPU: SERVEPUMP, BATCHCONT, PAGEDRUNS, PAGEDGPU,
  SERVEPREEMPT (new) all PASS via run_sio_test_suite
  (SOUNIO_MADAROS_AVAILABLE=1).
- Preempt trace: victims slot0/1/2 over turns 3/4/5
  (m3=256/512/1024), all five sequences finish, E decodes
  exactly its 3 maxgen tokens, queue drains to 0.
- GPU lane (RTX4000) + live smoke: pending, SLURM worker
  state with clusterops.
