# Final slow refusals and checkpoint-cost discriminator

The final results of the two original SOUNIO_LOOP_REGIONS=1 runs are failures:

| Job | Compiler | Native overtake status | Job elapsed | Batch MaxRSS |
| --- | --- | ---: | --- | ---: |
| 19524 | raw11332 | timeout124 at3000s | 50m28s | 3227128KiB |
| 19533 | raw8abe | timeout124 at3000s | 50m27s | 3227260KiB |

Both original slow programs check and compile0 after the linear-pair fix. The other three original programs execute0 with ALL_PASS. Both long-job collectors finish1 and their final input/compiler/source/witness guards pass. No slow success marker is produced. The historical fast custody records described these jobs as running; this supplement records their actual final outcomes without rewriting those historical snapshots.

The original workload, seeds, gate coefficients, assertions and 3000-second deadline are unchanged. The source header describes 62,914,560,000 wide draws. These are CPU simulation/sampling programs, not physical-QPU results. The original slow gate remains OPEN.

## Literal loop versus hoisted p00

Jobs19568 and19571 qualify a separate small discriminator on raw8abe. Both kernels take an already-computed BasisProb. The original LCG/counting loop is preserved byte-exactly; the candidate changes only a p00 field projection into a scalar binding outside that loop. Tape replay and the original complete slow program are outside this discriminator's scope. No qpu/chsh library change is made.

The C++23 reference uses uint64 arithmetic for the exact modulo-2^64 LCG, including the source's zero-seed normalization. It computes exact-boundary seeds through the multiplicative inverse and self-checks those generated states. Fourteen cases cover zero, negative zero, one, exact quarter/half/three-quarter thresholds, adjacent representable values, the upper32-bit boundary, NaN in p00/p01 and three distinct seeds with10M draws each. Every literal/hoisted count agrees exactly with the C++ reference.

The three benchmark cases run in alternating order within the same process. All results are retained; there is no selected favorable retry. Timings below are direct C++ reports, in nanoseconds:

| Mode | Case | Literal | Hoist | Literal/hoist |
| --- | --- | ---: | ---: | ---: |
| 1 | bench0 | 789927771 | 792642007 | 0.996575710 |
| 1 | bench1 | 793003973 | 792452059 | 1.00069646 |
| 1 | bench2 | 790818148 | 795527598 | 0.994080092 |
| 0 | bench0 | 495317949 | 488399611 | 1.01416532 |
| 0 | bench1 | 495883787 | 488289367 | 1.01555311 |
| 0 | bench2 | 493705855 | 489209924 | 1.00919019 |

C++ reports aggregate literal/hoist ratios0.997113455 in mode1 and1.01296726 in mode0. The standalone p00 hoist does not explain or close the slow timeout. It is not adopted on that performance claim.

Both jobs complete0 (19568:10s/MaxRSS1544556KiB;19571:8s/1544376KiB). All source/generated-input/compiler/ELF/SDK/library/trace guards pass. The exact scripts are run-19568.sbatch and run-19571.sbatch.

## Observed final-IR trace boundary

The NV2 trace's exact field_get census is four literal projections versus three hoisted projections; p00 changes from two to one. In mode1 the literal kernel is function8 with265 instructions:

- i66 calls __rgn_enter, flags64076, yielding token register37.
- The hot body has no IR allocation or ordinary call.
- i135 is the scalar p00 field_get, flags64064.
- i255 calls __rgn_exit_scalar, flags64077, source1=37 and source2=130 (the generated scalar zero).
- The ShotCounts allocation is i259, after the loop's last checkpoint exit.

The hoisted function9 has263 instructions, no field_get in its hot body, and its matching checkpoint exit at i253. Mode0 yields262/260 instructions and no checkpoint call in either kernel. The lowerer/codegen is not edited here.

This is a preserved TRACE, not a complete serialized IrModule. NV2 labels some instructions "other" and omits label_id, complete args, literal/name and binop/unop details needed to reconstruct CFG and all uses. It must not be advertised as an arena dump. A new final-IR optimizer component must obtain and qualify a complete module through its real producer route before integration.

Any future checkpoint elision must inspect the whole completed function/loop and every exit. Deciding from a prefix at loop_region_close would be unsafe when break/continue precedes a later allocating branch. Unknown operations, ordinary calls, aggregates or uncertain scalar provenance must keep the checkpoint. Allocation/call/escape/nested-loop/break/continue controls remain required. The root/compiler lanes own that review.

## Single-file source projection for the optimizer component

wide_counts_projection.sio removes imports/timing and declares the same four-field BasisProb/ShotCounts layouts. Both core functions remain byte-identical to the template, as recorded in WIDE_PROJECTION_LEDGER.json. Its main invokes the same benchmark0 inputs and prints counts; no invented timing is emitted.

Projection SHA256 is f2192e251936bdf331ff3a84e21cf0821de0e904a4b2044b0e55d71a99042931. Slurm19580 completes0/4s, MaxRSS485220KiB: real raw8abe check/compile/native and independent C++ LCG counts pass. This qualifies the projected single-file input only. It does not certify the optimizer's producer route or replace the imported benchmark's complete IR.

A separate control,19576, runs the unchanged original complete slow workload with only loop-region mode0 differing, still under3000s. At00:28:40UTC on2026-10-29 it remains RUNNING. Its later result cannot close the failed mode1 gate.
