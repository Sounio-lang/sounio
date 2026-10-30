<!-- docs:meta
topic_id: repo.docs.design.sm89-global-thread-index
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.design.sm89-global-thread-index
-->

# sm_89 global thread index

Status: design note for the experimental implementation in `926f1e9078`. Section 75 of `docs/design/heterogeneous-gpu-framework.md` records the earlier `TID_SUM=155392` against the unchanged requirement of 165600. The later single-block regression job 20147 passed with `TID_SUM=165600` and `BAD_TIDS=0`; it does not erase or rewrite the earlier observation. The multi-block results are recorded below.

## Current images

`sm89_own_build` in `stdlib/hetero/silicon/nvidia/elf.sio` is one block. Slot 5 is `S2R R4, SR_TID.X`. Slot 9 doubles R4 and waits on mask 5. Slot 10 makes the byte offset. The final store writes R4. `sm89_gemv_image` refuses `n>1024`. Launch is `grid=1, block=n`.

`sm89_grid_image` (`elf.sio` 549–594) is the implemented multi-block image. It copies that body, inserts a 13-instruction prefix, and grows the container to 48 instructions, 3112 bytes, register count 22, EXIT at slot 47. `si_gemv_grid_sm89` accepts `1..65535` by `1..4096`. `sg_launch_sm89` launches `block=128`, `grid=ceil(n/128)`.

The packed layout is unchanged: `output[n], tid[n], input[max(n,k)], weights[k][n]`, one pointer at `c[0][0x160]`.

## Alternative A — one grid, block 128

This is the image in `926f1e9078`.

`gid = ctaid.x * 128 + tid.x`. There is no measured `SR_NTID`, general `IMAD`, `SHL`, `LEA`, or `ISCADD`. `IMAD.MOV` (`24 74`) is a move. `IADD3.X` is only a row in the atlas table (`atlas.sio:17`); this body never emits it. Block 576 with `grid>1` has no multiply. Block 128 is the power of two already used by `stdlib/hetero/kernels/gemv_f32.ptx`. That PTX is sm_50. It is the algorithm, not an sm_89 encoding.

Prefix, after the copied slots 0–5:

1. `S2R R18, SR_CTAID.X`. The measured selector is byte 9 = `0x25`, produced here by adding 1024 to the TID high word `0x00321e0000002100`.
2. Seven `IADD3 R18, R18, R18` at stall 5. That is `*128`. The first adds a wait on scoreboard 1 on top of the `IADD3` mask 4, so the issued mask is `SB1|SB2`. The later six keep mask 4.
3. `IADD3 R4, R4, R18`, waiting on scoreboard 0. R4 becomes the gid. The copied body then treats R4 as it treated the thread index. The discussed reuse of the dead R17, which would have kept the register count at 20, was not taken.
4. `ISETP.LT.U32.AND P0, PT, R4, n`, then the measured `PLOP3`.
5. `@P0 BRA +16` skips one unconditional `BRA`. Live threads fall into the copied body. Tail threads take `BRA +448` to the single EXIT at slot 47. The `+448` is `(47-19)*16`.

The K-loop `@P0 BRA -144` is copied intact. Nothing is inserted between its `LDG` and that branch. `P0` is consumed by the tail branch before the loop's `ISETP` overwrites it. The tail branch is before the first `LDG`, so a gid `>= n` does not form an out-of-range weight address. The compare immediate is the same field that held 576 in the one-block loop. `n` up to 65535 fits in those two bytes, and the 65535 case passed in job 20215.

If `@P0 BRA +16` misses in a future test, one alternative to investigate is placing the check after the body, entering it with an unconditional positive `BRA`, and returning with a negative `@P0 BRA`, the direction job 19925 stored. That layout is not implemented or validated in `sm89_grid_image`.

`examples/silicon_grid_probe.sio` checks those words on the host and, under Slurm, checks dots, `tid[col]==col`, inputs and canaries. `tests/gpu/silicon_sm89_grid.py` also compares emitted images against CPU and PTX oracles.

On the NVIDIA L4, Slurm job 20215 completed 0:0: two repetitions of 32 checked-launch cases and 64 raw-image SASS/CPU/PTX cases passed, as did 68 legacy single-block cases. The checked launcher covered deterministic non-dyadic f32 inputs with an absolute tolerance of 1e-5 and relative tolerance of 1e-5 (5e-4 for K>64). The PTX comparisons used exact integer and dyadic inputs. N=65535 and K=4096 were tested separately, not together. The logs are in `/orangefs/training/hetero-probe/sio482-checked.h7gs8X/final-gate-20215.out`. These results validate the tested combinations on that device, not every scheduling combination or input.

## Alternative B — exact blocks, sequential launches

Not implemented. It is a possible alternative if A misses, not an automatic fallback.

Slice columns into `c<=1024`. For each slice, pack `output[c], tid[c], input[max(c,k)], weights[k][c]` and launch `sm89_gemv_image(c, k)` with `grid=1, block=c`. No second `S2R`, no tail, no new opcode. `n=576` stays one launch. `n=1025` is two.

Do not implement this by adding `base_col*4` to the single pointer. `R11` is both the buffer skip and the weight-row stride. An offset pointer misses the input and weight bases by `base_col*4`. Do not launch the current one-block image with `grid>1`: every block writes columns `0..c-1`.

## Scoreboard risks

The measured `S2R` control word is `0x00321e00`. Stall 15, yield 0, bits 46–48 = 0, bits 49–51 = 1, wait mask 3 (`SB0|SB1`). `examples/silicon_probe.sio` (325–334) extracts bits 46–48 as the destination barrier and requires 0. The first consumer of R4 must wait on mask 5 (`SB0|SB2`). Bits 49–51 are residue of that specimen. Do not rename them onto `atlas_sched`'s write-barrier field and then "fix" them.

Alternative A does not copy that control word for `SR_CTAID.X`. Slot 6 adds `1<<46` (`70368744177664`) as well as the selector. That sets the destination-barrier field to 1. The comment at `elf.sio:573` calls this independence from the TID read on scoreboard 0. It is not a measured residue. The first doubling waits on mask bit 1 (`9007199254740992`), on top of the `IADD3` mask 4, so its wait is `SB1|SB2`. Later doublings keep mask 4 and publish no scoreboard. The gid add waits on `SB0`.

That split is the intended schedule. Job 20215 exercised it successfully for the tested images and shapes. Its interpretation and generality remain risks:

- If bit 46 is not the destination barrier on this opcode, the `SB1` wait can hang, or both `S2R`s still publish on scoreboard 0. The second then retires the barrier before R4 is consumed.
- The CTAID word still carries the specimen wait mask 3. One `S2R` with that mask did not hang in job 19962; the retuned pair also ran in job 20215. Neither result establishes a universal scheduling rule.
- In the earlier measured `IADD3` slot, stall 5 committed and stall 3 did not. The doubling chain uses stall 5; this observation is not a general rule for all instructions.
- The copied body still contains the `ace2667` store split: first `STG` publishes scoreboard 0, the address add between the stores waits on mask 5, and the stores use R16 and R6. In the earlier single-block observation the hole was threads 128..191 (`T128=0`, `T191=0`, `D128=0`), and 10208 is the sum of those indices. Job 20147 subsequently passed that single-block regression. A future grid miss would still require diagnosis; do not rewrite 165600 to match it.

`@P0 BRA +16` joins two measured fields: predicated byte 1 = `0x09`, and the positive displacement whose high word job 19868 exercised for an unconditional `+16`. `BRA +448` uses the same displacement field at a width job 19868 did not store. The structural probe checks the bytes; job 20215 also returned the expected global indices and intact canaries for the tested shapes. This supports these words in this kernel, not arbitrary branch encodings.

## Address limit

No instruction in either image propagates a 32-bit carry. `MOV R2, c[0][0x160]` and `MOV R3, c[0][0x164]` split the pointer. Later adds touch the low half. R7 holds the high half and is not added. Section 74 records that a 32-bit step was enough for eighty bytes. That is not a carry proof.

The last live thread increments its weight pointer after the last load. The address that must stay in the same 4 GiB region is `base + 2*n*4 + span*4 + gid*4 + k*n*4` for `gid=n-1`. The large term is `k*n*4`, not the block index. Live threads do not widen it relative to a one-block launch of the same `n`.

`sg_layout_ok` (`stdlib/hetero/silicon/gemv.sio` 23–31) refuses a payload that does not fit, an unaligned base, an i64 overflow, and a low-half sum `(base & 0xffffffff) + (payload + 4*n - 4)` that exceeds `0xffffffff`. Capacity is not ownership. `sg_launch_sm89` queries `cuMemGetAddressRange_v2` and refuses unless the current device reports attributes 75 and 76 as 8 and 9. Rejected launches must not write; the probe compares the buffer before the accepted launch.

Tail threads are the extra address risk. For `n=576`, grid 5 covers gids 576..639. Those loads would pass the end of `weights[k][n]` even when the allocation does not cross 4 GiB. The branch to EXIT is before the `LDG` for that reason. The Python oracle checks the rounded final pointer only for the live `n`, not for the rounded grid. A tail that falls through is outside that check.

## What not to do

Do not encode `SR_NTID`, `ISETP.GE`, `@!P0`, or a predicate on `STG`, `LDG`, or `FFMA` without separate evidence. Do not add a second kernel parameter without changing and validating the measured container ABI. Do not change `sm89_own_image()` to the grid prefix: it remains the single-block regression oracle. Keep the expected sum 165600 unchanged.
