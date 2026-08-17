<!-- docs:meta
topic_id: repo.docs.audit.eisa-lowering-stack-investigation-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.eisa-lowering-stack-investigation-2026-08-17
-->

# EISA lowering stack investigation

Date: 2026-08-17

Status: measured; driver classified

Main baseline: `82ac550a01f4` (#1763), including `667930a5bd` (#1760)

## Decision

The EISA lowering failure is a real stack overflow. Its primary structural
driver is multi-megabyte expression-lowering frames multiplied by recursive
AST descent. The bump allocator's approximately 900 KiB per-function RSS cost
is separate heap pressure; it is not the cause of the 448 MiB stack boundary.

The next corrective step is to shrink the frames in
`lower_expr_non_binary_ref` and `lower_call_expr_ref`. Raising the stack to
1 GiB would postpone the same failure. Bump-allocation reclamation remains
important for total RSS and function-count scaling, but EISA does not put it on
the critical path for this stack overflow.

## Provenance

The instrumented compiler was built from source at `6f2c4e2461`. The measured
reproducer was `/tmp/ws_f_bridge_translate_min.sio`, which reaches
`bridge_translate_v1_stub` through the imported-module native path.

The lowering source is unchanged between the measured compiler and this
report's main baseline:

```text
git diff --stat 6f2c4e2461..82ac550a01 -- self-hosted/ir/lower.sio
# no output
```

`667930a5bd` changes only `bin/madaros` and
`scripts/ci/madaros_launcher_exit_status_gate.sh`. It raises the operational
reserve to 512 MiB and guards it against drift; it does not change the frames
measured below. #1763 changes CI and harness files only.

## Stack-boundary measurement

With a 384 MiB stack (`ulimit -s 393216`), GDB reported:

```text
[stack]  0x7fffe7fff000 .. 0x7ffffffff000  size 0x18000000
rsp      0x7fffe7e70b68
rbp      0x7fffe8b0f8e0
```

The stack mapping is exactly 402,653,184 bytes. At SIGSEGV, `rsp` was
1,631,384 bytes below its lower boundary. This directly establishes stack
exhaustion rather than an allocator failure or an RSS-only limit.

With a 448 MiB stack (`ulimit -s 458752`), the same compile passed. Polling the
`[stack]` section of `/proc/$pid/smaps` for the duration of the compile gave:

```text
rc=0
samples=700
max_stack_rss_kb=435004
max_stack_referenced_kb=435004
```

The passing compile physically touched 435,004 KiB, or approximately
424.8 MiB, in the stack mapping itself.

## Frame-size measurement

At the failing instruction:

```text
rbp - rsp = 13,233,528 bytes
```

The compiler's own large-frame diagnostic for the active function was:

```text
lower_call_expr_ref: stack frame = 13,233,520 bytes
```

The generated prologue independently agrees:

```asm
sub $0xc9ed70, %rsp
```

`0xc9ed70` is 13,233,520 bytes. The eight-byte difference from `rbp - rsp` is
the saved return address. One activation of `lower_call_expr_ref` therefore
uses approximately 12.62 MiB.

The other dominant diagnostic is:

```text
lower_expr_non_binary_ref: stack frame = 11,941,664 bytes
```

## Dynamic-depth measurement

Walking saved frame pointers at SIGSEGV found 102 active frames spanning
390,991,008 bytes. The dominant frame deltas were:

```text
26 x 11,941,704 bytes
26 x    244,824 bytes
14 x    979,192 bytes
12 x  2,633,080 bytes
11 x    244,856 bytes
 1 x 13,233,560 bytes
```

The repeated 11,941,704-byte delta matches the statically reported
11,941,664-byte `lower_expr_non_binary_ref` frame plus frame-chain overhead.
Those 26 activations alone account for 310,484,304 bytes, approximately
296 MiB. The terminal `lower_call_expr_ref`, recursive dispatch frames, and
the remainder of the call chain account for the observed approximately
425 MiB high-water mark.

The classification is therefore not simply "deep recursion." It is moderate
recursive expression descent made pathological by 11.94 MiB and 13.23 MiB
frames.

## Allocation-per-function control

`SOUNIO_LOWER_RSS_TRACE=1` around the EISA function reported:

```text
bridge_translate_v1_stub entry RSS:       814,492 KiB
bridge_translate_v1_stub after-block RSS: 925,112 KiB
observed RSS increase:                    110,620 KiB
```

Process RSS alone cannot classify this increase because it combines newly
touched stack pages and bump-allocated heap. The independent `smaps`
measurement resolves that ambiguity: up to 435,004 KiB belonged specifically
to `[stack]`, while the frame-pointer chain accounts directly for the live
stack consumption.

The known flat allocator cost is still material. At 236 lowered IR functions,
900 KiB per function predicts approximately 207 MiB of persistent heap/RSS.
That can coexist with the approximately 425 MiB live stack. Reclaiming it would
reduce total memory, but cannot shrink an 11.94 MiB activation or prevent `rsp`
from crossing the stack mapping boundary.

## Driver classification

| Candidate | Measured result | Classification |
| --- | ---: | --- |
| Recursion depth | 102 active frames; 26 repetitions of the dominant lowering cycle | Strong multiplier |
| Frame size | 11.94 MiB and 13.23 MiB dominant frames | Primary structural driver |
| Allocation per function | approximately 900 KiB/function; approximately 207 MiB at 236 functions | Separate heap/RSS pressure |

## Headroom implication

Reducing only the repeated 11.94 MiB frame to 64 KiB would change its measured
26-activation contribution from approximately 296 MiB to approximately
1.6 MiB. That is the available order of magnitude: frame surgery can recover
hundreds of MiB without approaching the environment's approximately 1.95 GiB
effective stack ceiling.

The 512 MiB reserve is an operational containment fix. It is sufficient for
the measured EISA path, but it should not be treated as the architectural fix
or used as the basis for another automatic doubling.

## Function-count capacity experiment

The flat-allocation hypothesis was tested separately from EISA with generated
single-module programs. Every generated function was made reachable through a
call chain; a dead-function control was also run and confirmed that cross-module
DCE reduces unreachable declarations before lowering. The independent variable
below is therefore merged-IR function count, not source-file size.

All runs used the same source-built Madaros, a 512 MiB stack, and an 8 GiB
per-process virtual-memory limit. `/proc/$pid/status` and
`/proc/$pid/smaps` were sampled continuously.

| Merged IR functions | Peak RSS KiB | Peak stack RSS KiB | Result |
| ---: | ---: | ---: | --- |
| 1 | 427,388 | 15,836 | pass |
| 33 | 447,552 | 29,992 | pass |
| 65 | 453,668 | 29,996 | pass |
| 129 | 465,360 | 29,992 | pass |
| 257 | 488,368 | 29,992 | pass |
| 513 | 535,576 | 29,992 | pass |
| 1,025 | 630,084 | 29,992 | pass |
| 2,049 | 819,080 | 29,992 | pass |
| 4,097 | 1,197,372 | 29,988 | pass |
| 8,001 | 1,917,832 | 32,024 | pass |
| 8,192 | 1,927,920 | 29,992 | pass |
| 8,193 requested | 548,380 | 21,156 | refused before merge |

The 33-through-8,001 least-squares fit is:

```text
peak_rss_KiB = 441,226.386 + 184.541687 * merged_functions
```

This workload therefore costs approximately 184.54 KiB per trivial reachable
function, not 900 KiB independent of function size. The known approximately
900 KiB observation is a useful heavy-function envelope, but it is not a flat
law of the current compiler.

The exact current function-count ceiling is architectural:

```text
IR_MAX_FUNCS = 8192
```

The 8,192-function generated program compiled successfully. Adding one more
reachable function produced the intended refusal:

```text
IR lowering failed during merge: too many functions:
shared IR module capacity exceeded (max 8192 slots)
```

At the measured trivial-function slope, the regression predicts 1.863 GiB RSS
at 8,192 functions, agreeing with the observed 1.839 GiB. At the conservative
900 KiB/function envelope, 8,192 functions project to approximately 7.45 GiB
including the measured intercept. In this configuration the IR slot limit is
the exact program-size ceiling; monotonic Box growth becomes the limiting
resource first only for heavier functions or after `IR_MAX_FUNCS` is raised.

## What reclamation would cost

This is not a greenfield garbage-collector project. The emitted runtime already
implements chunked Box-arena regions:

- `__arena_mark()` returns the current Box-arena cursor.
- `__arena_reset(mark)` walks chunk back-links, `munmap`s younger chunks, and
  restores the cursor.
- The imported-module pipeline already uses this protocol around dependency
  lowering.
- Its survivor protocol scans persistent IR call arguments into fixed BSS
  scratch, resets the arena, then rebuilds the `IrRegList` chains.

The current persistent `IrFunction` is flat except for instruction
`call_args: Option<Box<IrRegList>>`. Instruction bodies live in the flat IR
instruction arena. AST boxes are allocated before body lowering and must remain
outside a per-function reset window. `Lowerer.env` and expression-list scratch
are function-local and must be dead or cleared at the reset boundary.

The aggregate-storage pool is deliberately separate from the Box arena and has
no reset intrinsic. It contains aggregate backing blocks that may be aliased by
merged-module values. Per-function Box reclamation will not reclaim that pool;
its growth needs a separate lifetime/escape census before any reset is safe.

### Smallest safe implementation

1. Add allocation accounting around each function: Box-arena cursor delta,
   aggregate-pool cursor delta, surviving call-argument sites, and peak RSS.
   The required observation intrinsics already exist.
2. Take an arena mark after persistent AST/module state is established and
   immediately before lowering a function body.
3. After `lowerer_flush_current_func_mut`, snapshot call-argument survivors for
   that function from the authoritative flat argument pool.
4. Clear function-local `Lowerer` pointers, reset to the mark, and rebuild only
   the surviving argument chains. Refuse or skip reset loudly if the survivor
   census exceeds capacity; never truncate.
5. Gate equivalence against reset-disabled control runs using IR fingerprints,
   native output hashes, execution witnesses, and arena-unmap counters.

An even cleaner follow-on is to make the flat `IR_A_ARG_*` pool the sole
persistent representation and ensure stored `IrInstr.call_args` is always
`None`. That removes the only currently identified Box survivor rather than
continually re-boxing it, but requires a consumer census before changing the
representation contract.

### Estimated delivery shape

This is a bounded compiler change, not a runtime rewrite:

| Phase | Main work | Risk |
| --- | --- | --- |
| A: census | Per-function Box/pool deltas and survivor/refusal metrics | Low |
| B: opt-in region | Mark/reset around one lowered function plus existing-style re-box | Medium-high: dangling survivor |
| C: equivalence gate | Reset on/off IR, ELF, execution and negative-refusal comparison | Medium |
| D: default | Enable region reset; retain opt-out and loud skipped-reset counter | Medium |
| E: flatten survivor | Remove persistent `call_args` Box representation if consumer census permits | Medium, optional |

The difficult part is proof of the lifetime boundary, not allocator machinery.
The repo already contains the reset primitive, chunk unmapping, counters, a
module-scoped survivor algorithm, and explicit historical failure modes
(silent nested stores, argument-chain loss, survivor-capacity overflow).

Reclamation should therefore be scoped as one instrumented census change and
one opt-in compiler vertical before defaulting. It does not need tracing GC,
reference counting, destructors, or a language-wide ownership retrofit.
