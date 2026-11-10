<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-11-01-plan-m5-serve-device-design
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-11-01-plan-m5-serve-device-design
-->

# Plan M5: device paged attention in the live server — design

## Why

M4 proved the device kernel runs (proof counters, control exact)
and does NOT contract separate mul/add (FUSION-OFF, jobs
20991/20992). But the serve pump still computes attention on host:
`LL_PAGED_ATTN_GPU` is a compile-time const 0, and the full
`paged_attn_f64` kernel differs from the host reference in values
(M4 exit 40). M5 wires the device path into the live server behind
a runtime flag and closes the value gap to the bit-exact bar.

Out of scope: default-on cutover (flag stays opt-in), multi-GPU,
preemption/recompute, sampler changes, HTTP/auth changes.

## Current (what the code does today)

`svb_pump_fwd` (serve.sio) routes `use_gpu && rt.ready` to
`llama_forward_batch_runs_rt`, which checks the const and always
takes the host impl (`dev = 0`). `ll_runs_layer` (llama.sio)
computes every attention row on host, then overwrites `att` with
the device kernel only when `pagedev != 0`. So the device kernel
is proven in tests but unreachable from serve.

## M5 changes

1. Runtime flag, default off (env or serve arg, M3 precedent):
   flag on plus `rt.ready` selects the device paged path for pump
   turns; flag off keeps today's host behavior byte for byte.
2. Root-cause the bis_v4 value diff first. Ranked suspects after
   the M4 finding (FMA contraction excluded): the device exp
   sequence vs `math::pure::exp`, scale multiply placement,
   table/lens index staging. Fix kernel or host reference until
   the M4 matrix plus `paged_gpu` L1/L2 are bit-exact. No silent
   tolerance widening.
3. Lockstep demo: the scripted pump scenario runs twice, host
   flag-off vs device flag-on; receipt claims equal greedy token
   streams AND equal row logits bit-exact.
4. Serve fallback contract (unchanged from M4): any device
   anomaly in a turn (alloc/load/launch/sync/proof) falls back to
   the host row for that turn and marks the served receipt line;
   the exactness receipt only covers fully-device turns.

## Verification

New run-pass test `paged_serve_device.sio`: CPU arm always
(flag off, host path, design constants); GPU arm on RTX4000
(flag on, device path, bit-exact vs host arm in the same
process). Markers: `PAGEDEV-EXACT PASS` plus `gpu_rows=N`.
No regression: M2 9/9, SERVEPUMP, PAGEDGPU, MATRIX green in the
same tree.

## Receipts

- `PAGEDEV-EXACT PASS` on CPU (device skipped) and on RTX4000
  with counter proof N > 0 and zero differing logits.
- Bis_v4 value diff closed with a named cause (kernel or
  reference fix, not tolerance).
- Serve keeps host default; device only under the flag.

## Open questions (decide before implementing)

- Flag transport: env var or serve arg? (Recommend arg, M3 precedent.)
- Demo dual-forward (host AND device, compare, 2x cost) vs
  serve wholesale switch with fallback? (Recommend dual in the
  test, switch in serve.)
