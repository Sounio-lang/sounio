<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-11-02-plan-m6-paged-scratch-design
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-11-02-plan-m6-paged-scratch-design
-->

# Plan M6: per-head scratch in the paged device kernel — design

## Why

The `paged_attn_f64` kernel is already parallel: grid
(nrows, nheads), one thread per block, row = ctaid.x and head =
ctaid.y. But the score scratch has no head dimension:
`sco[row * maxblocks * B + t]` is shared by every head of the
row, with no cross-block sync possible. All M4/M5 green lanes
(0/48, 3/3 PASS) rest on benign block scheduling, not on
correctness. M6 owns the cause: give each (row, head) lane its
own scratch.

Out of scope: blockDim > 1, shared-memory staging, default-on
cutover, multi-GPU, sampler or HTTP changes.

## Current (what the code does today)

PTX base: `rd_sb = sco + row * sstride`, shared across the
row's heads. Host allocates `dc` as `n * maxb * bb` f64
(llama.sio `ll_attn_paged_rows_gpu`). The `.cu` C oracle loops
single-thread, so the shared layout is correct there and only
there. Proof words are already per (row, head) and stay.

## M6 changes

1. Scratch layout `sco[row, head, block]`: base =
   `(row * nheads + head) * sstride`. Same element op order,
   same exp sequence, same proof scheme.
2. Host `dc` alloc becomes `n * heads * maxb * bb` f64. Kernel
   entry signature unchanged (same `sco_ptr` param).
3. Mirror the layout in the `.cu` oracle (comments, index
   math) so it stays the op-for-op reference.
4. Fix the stale header comments (PTX + CU) to the new dims.

## Verification

Same bars as M5, no new harness: CPU SERVEPUMP/BATCHCONT/
PAGEDRUNS PASS; RTX4000 Ada 3/3 PASS with live CUDA context;
serve vivo pagedev 0 vs 1 byte-identical. Any mismatch stops
the lane and bisects by (row, head).
