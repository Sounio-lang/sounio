<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-11-01-plan-m4-paged-gpu-attn-design
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-11-01-plan-m4-paged-gpu-attn-design
-->

# Plan M4 (SOTA): true paged GPU attention — design

## Why

Every GPU lane so far proves `rt.ready` and bit-exact CPU
fallback, but `rt_gemm_rect` rounds GPU rows down to multiples
of 16 — every run-pass GPU arm (chunk 2, tiny rows) has executed
100% on the host. No device compute has ever been proven. M4
runs the paged attention itself on the device: one fused kernel
over the block tables, with a device-side execution counter as
proof (stronger than `rt.ready`).

Out of scope: tensor-core tiling, CUDA graphs, KV upload
avoidance (blocks re-upload per layer per turn in M4), device
free (the runtime never frees; M4 follows the same
test-scale-only discipline), perf targets.

## Current

`llama_forward_batch_runs[_rt]` runs QKV/output GEMMs on the
device when rows allow, but the attention dot/softmax/accumulate
loop is host code over `paged_get` in both arms. The attn
kernels (`attn_rope/softmax/av`) exist only as `.cu` sources for
the quantized sidecar flow; no `attn_*.ptx` is committed and no
nvcc exists on any controlled host, so they can never load.

## M4 — fused paged attention kernel

New file `stdlib/hetero/kernels/paged_attn_f32.ptx` (hand-written;
no nvcc exists — the matching `.cu` is committed as the readable
reference and states this). One thread per (row, head), 2D grid
`rows x heads`, f32 throughout:

- inputs: q rows `[nrows, nheads, dh]`, one flat K block pool
  plus one flat V block pool (f32, host-packed per layer per
  turn from the paged tables), per-row int block tables into the
  pools, per-row lens, `scale`, `B`, `rep` (GQA), dims.
- pass 1: max over positions (dots recomputed, q in regs).
- pass 2: exp-sum plus weighted V accumulate (dots recomputed).
- writes out `[nrows, nheads, dh]` plus one `atom.global.inc`
  on a device counter per block (execution proof).

`expf` via `ex2.approx` (`e^x = ex2(x * log2e)`); the f32-vs-f64
gap is covered by the tolerance rule below, not by matching
host `exp` bit-for-bit.

Host side (new public fn in `llm::attn_gpu`, e.g.
`attn_paged_rows_gpu`): convert q/k/v f64 to f32 with the
existing helper, upload pools + tables + lens, launch with
`ce_launch_2d_shmem`, download out + counter, convert back.
Any failure (alloc/launch/status) returns nonzero and the caller
keeps the host loop — same fallback contract as `ll_proj`.

## Verification

New run-pass test `paged_gpu.sio`: scripted 2-seq pump
scenario (chunk 2, mid arrival, slot reuse) through the device
path vs the host `runs` path, same model, lockstep rows.
Markers: `PAGEDGPU PASS`, `gpu_rows=N` with N equal to
rows-times-heads (counter proof), greedy token streams equal.
CPU arm always; GPU arm when `rt.ready` (guarded, like M2).

## Receipts

- PAGEDGPU PASS on CPU (device path skipped, host fallback)
  and on RTX4000 with counter proof N > 0.
- Tolerance rule (decided below): same greedy tokens
  (argmax lockstep) plus a bounded diff; exact f32 bit-equality
  is NOT claimed.
- No regression: M2 9/9 + SERVEPUMP CPU green in the same tree.
- Serve keeps calling the host path; device wiring into the
  live server is M5.

## Open questions (decide before implementing)

- Tolerance rule: argmax-lockstep + 1e-6-scale diff bound
  (attn-demo precedent) vs exact-match attempt?
- Serve wiring now (behind flag) or M5?
