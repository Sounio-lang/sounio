# Experimental loop-checkpoint elision

Status: implemented and component-qualified; not integrated into the compiler.
The frontend patch is a review artifact. Complete imported quantum IR capture,
target emission, scientific before/after concordance and integrated regressions
remain open. This is not a benchmark result or a change to the default mode.

## Contract

`ir::loop_checkpoint_elision` operates on the final mutable `IrModule`.
It recognizes only loop-region calls 64076–64079. Function-frame calls
64066/64067 and protected Worker operations 64501/64502 remain unchanged.
The module wrapper visits the existing function table without changing its
pointer, function IDs, counts or metadata. Counts in either audit struct are
diagnostics; applying always proves the current function again.

A successful proof covers the entire function: all instructions, call targets,
flat argument-pool operands, register definitions/uses, labels, control-flow
edges, reachable stack states and all exits of every token. It never decides
from the first matching exit or a trace prefix. An effect in an alternate arm
after an earlier exit still disqualifies the enclosing region. Nested regions
must independently satisfy the same proof; an unsafe inner region contaminates
its active ancestors. A safe inner region may be removed when a later allocation
makes only its outer region unsafe.

The token is uniquely defined by its recognized entry and used only as argument
zero of its matching region calls. Close results and the integer-zero scalar
argument cannot have observers. The close argument must be a uniquely defined,
non-float `LoadImm 0`; a float marker or another use refuses. Every source must
be defined on every incoming path. Returns with active tokens, inconsistent
stack joins, reentering an active token and unreachable region calls refuse.

Keep-local is recognized and checked, but any region using it is preserved:
this MVP has no aggregate-root provenance proof. The leave operation is eligible
only with the same token/Unit restrictions. Unknown instructions, arities,
metadata, flags or control flow preserve the region or the whole function.

## Exact scalar admission and native-emitter audit

This proof is restricted to x86_64 Linux native-v2, compile strategy zero,
no sret, ref/f128 parameters or Knowledge return profile. Backend or target
extensions require a new audit. The audited primary source is
`self-hosted/native/codegen_x86_linux.sio` SHA-256
`d3380a0826be9547204e8cc53d35c45e6fd1e010f421172d64c80a1e6507e01d`
(root fa485cc07e5596573eb99807184ccb47878fa198). The audit reads
`nc_emit_core_binop`, `nc_emit_core_float_binop_into`, and the native-v2
load/conversion/copy/unary dispatch. These admitted paths use direct integer
instructions, SSE or temporary stack moves, with no runtime allocator/helper.
Absence of an explicit IrAlloc alone is not the criterion.

| Instruction | Exact admitted profile inside a region |
| --- | --- |
| LoadImm, LoadFloat | flags 0; LoadImm -4242 refused everywhere in the function |
| LoadBool | flags 0 and value exactly 0 or 1 |
| Copy, IntToFloat, FloatToInt | flags 0 |
| BinOp, flags 0 | Add/Sub/Mul/Div/Rem; Eq/Ne/Lt/Le/Gt/Ge; And/Or; BitAnd/BitOr/BitXor; Shl/Shr; UDiv/URem/UShr |
| BinOp, flags 2, 4 or 6 | Add/Sub/Mul/Div and the six comparisons only |
| UnaryOp | flags 0, Neg/Not/BitNot only |
| Nop | flags 0 or numeric marker 64064 |
| Label, Jump, BranchTrue, BranchFalse | flags 0 only |

The -4242 exclusion is global because the emitter uses that LoadImm as an I/O
adjacency sentinel. Replacing a neighboring call by LoadImm 0 must not activate
a previously inactive sentinel outside the candidate region.

FieldGet is preserved even with marker 64064. LoadString, field/index access,
arrays, pointer loads/stores, refs/deref, calls, allocations, globals, Phi, wide,
epistemic and unknown operations are not admitted inside a candidate.
Protected Worker calls are never neutralized. A by-value field read before
entry can remain outside the region; no source motion is performed by this pass.

## Helper identity and real placeholder ABI

Each marked call must match both its resolved function ID/name and its
instruction name, the exact flag, canonical unused columns and full slot args.
Arity is 0 for enter, 2 for scalar close and keep-local, and 1 for leave.
The metadata of these synthesized empty builtin placeholders is exactly zero
params, zero registers/instructions/labels, strategy zero and the remaining
canonical empty-function descriptor fields checked by the implementation.
Call arity and placeholder metadata are separate facts.

The original raw03 Worker collision was inspected by qualified ptrace job19630:
`__rgn_enter` and `__rgn_exit_scalar` really have param_count 0, not 2.
This supports the exact-zero profile; it does not prove the still-uncaptured
wide_counts arena. A fabricated param_count 2 is a negative component control.

## Arena transaction and alias protection

Only the selected function region is cloned. Every live slot is copied with
`ir_arena_store_from`, retaining the original flat argument-pool base.
All scalar columns, exact f64 bits, name IDs, arg count and arg-base identity are
compared before mutation. The old region and any alias in another function
remain untouched, including pool entries of calls with more than two arguments.
The clone is not published if the proof, clone/copy checks or arena contract fail.

Only proven entry/exit call slots are neutralized: LoadImm 0 for a defined
destination, otherwise Nop. Register definitions remain present, with no
compaction, label remap, function-table replacement or arena reset. Publication
uses the same function metadata with its new private region handle. The compiler
scratch maps are private, bounded and sequential; this implementation is not
reentrant or a parallel pass. Limits are 1024 instructions/labels, 512 registers
and 64 call args/parameters. Unsupported larger functions are preserved.

## Integration proposal, not yet applied

`tests/fixtures/wave4_loop_checkpoint_elision/frontend_preseal_hook.patch`
targets frontend SHA
`9d33ff64767e34109c111999bb6e81b2d42001c89d606063d636e386526ce646`
(root16ffddfbfe037fd083cc891b3f4fe23caaee604d). Candidate frontend SHA is
`601a338f91a544590f4c964a18cc043ff82ac3ec4cd69fd9359ba377787ecb54`.
The patch is not a semantic qualification of that frontend.

Both single and merged routes call the module pass after cleanup, immediately
before the first function-region seal, independent of -O. The target resolver
already used for emission is moved before that call; only x86_64 Linux with
`SOUNIO_LOOP_REGIONS=1` enters the pass. No new ordinary/Worker guess is needed:
this stage precedes Worker clone preparation and final proof/certificates.
Worker binding/intent metadata do not rely on instruction indices; the pass
retains IDs/table/layout and non-target args. Later Worker proof must still run.
That proof does not substitute for this transformation's semantic proof.
No mutation is proposed for the native prepared stage, which is post-seal.

## Component qualification and required next gate

Job19635 ran the published gate using exact module/fixture sources:
modern raw8abe checks, legacy9cf produces the component ELF, and that ELF runs
the actual arena/function-table implementation and pass. All stages exited0.
The fixture contains 22 CFG/effect cases, 66 targeted attacks, four control-flag
attacks, three float-flag positives, alias/3arg-pool transaction checks and module
idempotence/bounds checks: 97 controls and one final marker.

These are synthetic real arena objects, not a reconstructed quantum arena.
Native execution qualifies the pass component; it does not yet qualify emission
or execution of transformed scientific kernels. The next integrated raw must
capture complete source-stage Schema2 IR, apply the exact pass, emit and compare
original quantum outputs with C++23, and run the existing 51 checker/21 native
loans, logical/short-circuit and constant regressions. Worker final proof and
unchanged protected ops/metadata remain mandatory.

The imported and single-file load_multimodule_ir driver in job19595 failed
before producing an IrModule (unresolved source locals / imported E039).
Modern compilation of the component itself in job19584 checked0 then failed139
while lowering ir.sio. Neither is a semantic rejection or a captured arena.
The qualified component therefore uses a separately identified legacy producer.
The complete Schema2 diagnostic is already delivered independently; old NV2
summary lines are never treated as a complete arena.

The scientific projection and imported template copied beside the component
are immutable artifacts from authorized commit
b5e6e549a31db0423ff2cb16365a3b32a20a0f04. Their provenance ledger distinguishes
the projection from the imported source. No quantum stdlib source, expected
corpus, timeout, benchmark mode or compiler lowerer is changed here.
