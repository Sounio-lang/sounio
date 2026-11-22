# M9: preemption with victim policy + recompute via spill log

## Direction

The serve pump ran 4 slots to EOS/maxgen with no eviction and
no overflow signal: any sequence past table capacity degraded
silently. M9 adds a visible overflow signal, smallest-gen
victim preemption with a spill log, and bit-exact recompute on
re-admit. Parallel arrays beside the slot structs; no struct
layout change.

## Slice

meta[3] carries the signal (bit s: row hit cap; bit s+8:
slot s preempted this turn). admit_all evicts the smallest-gen
decoding slot when full, stashes its spill log in vspill,
truncates its tables, and requeues the descriptor (vs/xl/vl/gn
in SvQueue). build_rows replays bank prompt + spill on
re-admit. Turn maps overflow to pdone=3, receipt truncated.
New serve_preempt.sio: 5 sequences over 4 slots, every row
replayed token-for-token against sequential single forwards.

## Receipts 2026-11-03

- CPU 5/5 PASS: SERVEPUMP, BATCHCONT, PAGEDRUNS, PAGEDGPU,
  SERVEPREEMPT via run_sio_test_suite.
- Preempt trace: victims slot0/1/2 on turns 3/4/5, all five
  finish, queue drains, E decodes exactly maxgen=3.
- Two engine bugs caught by build+run and fixed: meta[3]
  cleared after (instead of before) admit/build, wiping the
  signal; whole-path svb_pump never recorded the spill log.
- Test bugs fixed: stale rowtok across turns; slot identity
  not tracked across occupant change.
- RTX4000 (5860) lane 2026-11-03: all five PASS
  (SERVEPUMP, BATCHCONT, PAGEDRUNS, PAGEDGPU-DEV+PAGEDGPU,
  SERVEPREEMPT). Bundle staged via srun stdin (base64+tar;
  node /workspace is not the pod volume, /tmp is per-job
  private, so extract+run share one allocation). paged_gpu
  FPE was a missing micro-paged fixture, restaged green.
- Live smoke (pod CPU, SmolLM2-135M): pagedev 0 vs 1
  byte-identical (Paris completion, 6/3/9, finish=length).
  Turn wiring and receipts verified end to end. Both smoke
  servers stopped; pre-existing 8090 untouched.

## Notes

- One preempt per fwd call; queue-full or all-prefill
  occupancy waits instead of preempting.
- vvalid guards slots whose vspill region is reserved for a
  queued victim.
