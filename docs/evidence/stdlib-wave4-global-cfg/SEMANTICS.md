# Certified read/definition and CFG scope

This component checks virtual-register must-definition, not pointee initialization,
memory safety, source typing, call-target resolution or numerical correctness.

| Opcode | Virtual reads | Virtual definition |
| --- | --- | --- |
|0 LoadImm,1 LoadFloat,4 LoadBool,5 LoadString,6 StaticBlob,20 Alloc | none | dst |
|2 IntToFloat,3 FloatToInt,7 Copy,15 FieldGet,19 ArrayLen | src1 | dst |
|8 BinOp | src1,src2 | dst |
|9 UnaryOp Neg/Not/BitNot only | src1 | dst |
|10 Call | complete captured call_args | dst only if nonnegative |
|11 Return with real operand | src1 | none; no successors |
|12 Jump | none | none; target only |
|13 BranchTrue,14 BranchFalse | src1 | none; target and fallthrough |
|16 FieldSet | src1,src2 | none |
|17 IndexGet | src1,src2 | dst |
|18 IndexSet | src1,src2 and signed imm_i64 register | none |
|21 Label,22 Nop/float marker | none | none |

Call src1/src2 are compatibility caches, not additional hidden arguments.
Unit calls may omit dst. A Return with src1=-1 is coverage-refused rather than
treated as a valid no-read return: ir_function_regs_are_valid requires a real
operand, and native Return loads src1. This does NOT claim every source Unit
return is a language bug; its producer convention remains outside this profile.

Ref/RefMut UnaryOp computes a stack-slot address by LEA and does not read the
slot value. They are coverage-refused until address provenance/alias effects are
modeled separately; they are never counted as ordinary value reads. Deref,
including byte-width flags, is also outside this profile. The native LoadImm
sentinel-4242 writes the previous instruction's destination, so it is refused
instead of fabricating a definition of its own dst. Phi incoming semantics are
not encoded in this auditor; Phi refuses coverage.

Flags are opcode-scoped and closed:
- LoadImm:0 or64068 BSS-address marker; sentinel refused independently.
- BinOp:0/2/4/6 scalar operand float flags. BinaryOp0..17 and21..23 only.
- Unary Neg:0/2; Not/BitNot:0.
- Call:0,64064,64066,64067,64076..64079, with ordinary complete captured args.
- FieldGet/IndexGet/Nop:0 or64064. Nop marker never defines dst.
- Alloc:0,64065 measure or64069 Box; single dst, no implicit vreg source.
- Remaining supported scalar opcodes:0 only.

Every other opcode/flag/UnaryOp/BinaryOp shape explicitly refuses coverage,
including IrWideCmp(op100), vectors and other wide operations. No claim that
all IR opcodes are supported follows from the41 controls.

All descriptors are censused and printed before supported coverage analysis.
Strict-parser FN indices, instruction rows and global totals join to the decoded
module census; each function's decoded instruction count and final audited/data/
empty classification are checked. Unsupported coverage never produces an accepted
global summary. Actual baseline per-function rows and parser summaries are retained
in19800/results/{wide,scalar}.audit.stdout.

Resource admission precedes bitset/CFG construction: executable reg_count and
instruction count <=16384, labels <=16384, per-function bitset storage <=8388608
uint64 words. Source capture <=64MiB and each line <=128KiB before parsing.
Fixed-point work is capped at67108864 bitword operations. Crossing a bound is
resource coverage-refusal2, never empty pass. Huge reg/label metadata controls
refuse without allocation;1025 real virtual slots remain supported.
