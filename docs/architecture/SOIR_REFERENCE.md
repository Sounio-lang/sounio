<!-- docs:meta
topic_id: repo.docs.architecture.soir-reference
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.soir-reference
-->

# SOIR v1 Format Reference Card

**Version**: 1
**Last Updated**: 2026-02-13
**Format**: Sounio Intermediate Representation (SOIR)

## Quick Format Overview

```
SOIR File = Header + Body

Header (8 bytes):
  Magic: "SOIR" (4 bytes)
  Version: 1 (1 byte)
  Reserved: 0x00 (3 bytes)

Body:
  fn_count: i64
  functions: IrFunction[]
  string_count: i64
  strings: Name[]
```

## Binary Layout Diagram

```
Offset    Size    Field              Description
────────────────────────────────────────────────────────────
0x0000    4       magic              "SOIR" (0x53 0x4F 0x49 0x52)
0x0004    1       version            Format version (0x01)
0x0005    3       reserved           Reserved (0x00 0x00 0x00)
0x0008    8       fn_count           Number of functions (i64 LE)
0x0010    ?       functions[]        Array of IrFunction
?         8       string_count       Number of strings (i64 LE)
?         ?       strings[]          Array of Name
────────────────────────────────────────────────────────────
LE = Little Endian
```

## IrFunction Layout

```
Offset    Size    Field              Description
────────────────────────────────────────────────────────────
+0x00     136     name               Function name (Name type)
+0x88     8       instr_count        Number of instructions
+0x90     8       reg_count          Number of virtual registers
+0x98     8       label_count        Number of labels
+0xA0     8       param_count        Number of parameters
+0xA8     512     param_regs[64]     Parameter register array (64 × i64)
+0x2A8    ?       instrs[]           Array of IrInstr (instr_count × 237 bytes)
────────────────────────────────────────────────────────────
Total header: 664 bytes + (instr_count × 237 bytes)
```

## IrInstr Layout (237 bytes fixed)

```
Offset    Size    Field              Description
────────────────────────────────────────────────────────────
+0x00     1       op                 Opcode (IrOpcode enum, 1 byte)
+0x01     7       padding            Alignment padding
+0x08     8       dst                Destination register (i64)
+0x10     8       src1               Source register 1 (i64)
+0x18     8       src2               Source register 2 (i64)
+0x20     8       imm_i64            Immediate integer value (i64)
+0x28     8       imm_f64            Immediate float value (f64)
+0x30     8       label_id           Label identifier (i64)
+0x38     8       fn_id              Function identifier (i64)
+0x40     8       field_idx          Field index (i64)
+0x48     1       bin_op             Binary operator (BinaryOp enum)
+0x49     7       padding            Alignment padding
+0x50     1       un_op              Unary operator (UnaryOp enum)
+0x51     7       padding            Alignment padding
+0x58     136     name               Name buffer (Name type)
+0xE0     8       arg_count          Argument count (i64)
────────────────────────────────────────────────────────────
Total: 237 bytes per instruction
```

## Name Type Layout

```
Offset    Size    Field              Description
────────────────────────────────────────────────────────────
+0x00     8       len                String length (i64)
+0x08     128     buf[128]           Character buffer (i8 array)
────────────────────────────────────────────────────────────
Total: 136 bytes
```

## Opcode Reference Table

| Code | Opcode | Operands | Description |
|------|--------|----------|-------------|
| 0 | IrLoadImm | dst, imm_i64 | Load integer immediate: `dst = imm_i64` |
| 1 | IrLoadFloat | dst, imm_f64 | Load float immediate: `dst = imm_f64` |
| 2 | IrLoadBool | dst, imm_i64 | Load boolean: `dst = (imm_i64 != 0)` |
| 3 | IrLoadString | dst, imm_i64, name | Load string literal: `dst = &strings[imm_i64]` |
| 4 | IrCopy | dst, src1 | Copy register: `dst = src1` |
| 5 | IrBinOp | dst, src1, bin_op, src2 | Binary operation: `dst = src1 op src2` |
| 6 | IrUnaryOp | dst, un_op, src1 | Unary operation: `dst = op src1` |
| 7 | IrCall | dst, fn_id, name, arg_count | Function call: `dst = call fn_id(args...)` |
| 8 | IrReturn | src1 | Return from function: `return src1` |
| 9 | IrJump | label_id | Unconditional jump: `goto label_id` |
| 10 | IrBranchTrue | src1, label_id | Branch if true: `if src1 goto label_id` |
| 11 | IrBranchFalse | src1, label_id | Branch if false: `if !src1 goto label_id` |
| 12 | IrFieldGet | dst, src1, field_idx, name | Get struct field: `dst = src1.field` |
| 13 | IrFieldSet | src1, field_idx, src2, name | Set struct field: `src1.field = src2` |
| 14 | IrIndexGet | dst, src1, src2 | Get array element: `dst = src1[src2]` |
| 15 | IrIndexSet | src1, src2, imm_i64 | Set array element: `src1[src2] = imm_i64` |
| 16 | IrAlloc | dst, imm_i64 | Heap allocation: `dst = alloc(imm_i64)` |
| 17 | IrLabel | label_id | Control flow label (no-op) |
| 18 | IrNop | - | No operation |
| 19 | IrPhi | dst | SSA phi node (future use) |

## BinaryOp Reference Table

| Code | BinaryOp | Symbol | Description |
|------|----------|--------|-------------|
| 0 | OpAdd | + | Addition |
| 1 | OpSub | - | Subtraction |
| 2 | OpMul | * | Multiplication |
| 3 | OpDiv | / | Division |
| 4 | OpRem | % | Remainder (modulo) |
| 5 | OpEq | == | Equality |
| 6 | OpNe | != | Inequality |
| 7 | OpLt | < | Less than |
| 8 | OpLe | <= | Less than or equal |
| 9 | OpGt | > | Greater than |
| 10 | OpGe | >= | Greater than or equal |
| 11 | OpAnd | && | Logical AND |
| 12 | OpOr | \|\| | Logical OR |
| 13 | OpBitAnd | & | Bitwise AND |
| 14 | OpBitOr | \| | Bitwise OR |
| 15 | OpBitXor | ^ | Bitwise XOR |
| 16 | OpShl | << | Shift left |
| 17 | OpShr | >> | Shift right |
| 18 | OpConcat | ++ | Array/string concatenation |
| 19 | OpRange | .. | Range (exclusive end) |
| 20 | OpRangeInclusive | ..= | Range (inclusive end) |

## UnaryOp Reference Table

| Code | UnaryOp | Symbol | Description |
|------|---------|--------|-------------|
| 0 | OpNeg | - | Negation |
| 1 | OpNot | ! | Logical NOT |
| 2 | OpRef | & | Borrow (shared reference) |
| 3 | OpRefMut | &! | Borrow (exclusive reference) |
| 4 | OpDeref | * | Dereference |

## Size Limits and Constants

```sio
IR_MAX_FUNCS:    64      // Maximum functions per module
IR_MAX_STRINGS:  256     // Maximum string literals per module
IR_MAX_INSTRS:   2048    // Maximum instructions per function
IR_MAX_PARAMS:   64      // Maximum parameters per function
SOIR_MAX_SIZE:   131072  // Maximum module size (128 KB)
IR_INVALID_REG:  -1      // Sentinel for invalid register
IR_INVALID_LABEL: -1     // Sentinel for invalid label
IR_INVALID_FN:   -1      // Sentinel for invalid function
```

## Annotated Example: Simple Function

### Source Code

```sio
fn add(x: i64, y: i64) -> i64 {
    x + y
}
```

### IR Representation

```
IrFunction {
  name: "add" (len=3)
  instr_count: 4
  reg_count: 3
  label_count: 0
  param_count: 2
  param_regs: [0, 1, -1, -1, ...]

  instrs: [
    0: IrLoadImm  dst=v0  imm_i64=0      // v0 = 0 (param x)
    1: IrLoadImm  dst=v1  imm_i64=0      // v1 = 0 (param y)
    2: IrBinOp    dst=v2  src1=v0  bin_op=OpAdd  src2=v1  // v2 = v0 + v1
    3: IrReturn   src1=v2                // return v2
  ]
}
```

### Hexdump (Partial)

```
Offset    Hex                                          ASCII
────────────────────────────────────────────────────────────────────
0x0000    53 4F 49 52 01 00 00 00                      SOIR....
0x0008    01 00 00 00 00 00 00 00                      ........  (fn_count=1)
0x0010    03 00 00 00 00 00 00 00                      ........  (name.len=3)
0x0018    61 64 64 00 00 00 00 00 ... (120 zeros)     add.....  (name.buf)
0x0088    04 00 00 00 00 00 00 00                      ........  (instr_count=4)
0x0090    03 00 00 00 00 00 00 00                      ........  (reg_count=3)
0x0098    00 00 00 00 00 00 00 00                      ........  (label_count=0)
0x00A0    02 00 00 00 00 00 00 00                      ........  (param_count=2)
0x00A8    00 00 00 00 00 00 00 00                      ........  (param_regs[0]=0)
0x00B0    01 00 00 00 00 00 00 00                      ........  (param_regs[1]=1)
0x00B8    FF FF FF FF FF FF FF FF ... (504 more)      ........  (rest=-1)
0x02A8    00 00 00 00 00 00 00 00                      ........  (instr[0].op=0)
...
```

## Validation Rules

### Module-Level Constraints

1. `fn_count` must be in range [0, IR_MAX_FUNCS]
2. `string_count` must be in range [0, IR_MAX_STRINGS]
3. Total file size must be ≤ SOIR_MAX_SIZE (128 KB)
4. All function names must be unique

### Function-Level Constraints

1. `instr_count` must be in range [0, IR_MAX_INSTRS]
2. `reg_count` must be ≥ highest vreg used
3. `label_count` must be ≥ highest label_id used
4. `param_count` must be in range [0, IR_MAX_PARAMS]
5. All `param_regs[i]` for i < param_count must be valid (≥ 0)
6. All `param_regs[i]` for i ≥ param_count must be IR_INVALID_REG (-1)

### Instruction-Level Constraints

1. All vreg references (dst, src1, src2) must be < reg_count or IR_INVALID_REG
2. All label_id references must be < label_count or IR_INVALID_LABEL
3. All fn_id references must be < fn_count or IR_INVALID_FN
4. IrLoadString: imm_i64 (string index) must be < string_count
5. IrFieldGet/Set: field_idx must be ≥ 0
6. IrAlloc: imm_i64 (size) must be > 0

## Endianness and Alignment

- All multi-byte integers: **little-endian**
- All floats: **IEEE 754 binary64 (little-endian)**
- Alignment: 8-byte boundaries (enforced by 7-byte padding after 1-byte fields)
- Name buffers: byte array (no endianness)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1 | 2026-02-13 | Initial SOIR v1 format specification |

## Compatibility Matrix

| SOIR Version | Compiler Version | VM Version | Status |
|--------------|------------------|------------|--------|
| 1 | 0.5.0+ | poseidon 0.1+ | Current |

## Tools

| Command | Description |
|---------|-------------|
| `sounio-verify inspect file.soir` | Disassemble SOIR to human-readable format |
| `sounio-verify serialize file.sio output.soir` | Compile Sounio to SOIR |
| `sounio-verify normalize file.soir` | Show normalized IR |
| `sounio-verify compare file1.soir file2.soir` | Compare two SOIR files |
| `sounio-verify validate file.soir` | Validate SOIR constraints |

## Further Reading

- [RUSTLESS_CUTOVER.md](RUSTLESS_CUTOVER.md) - Complete documentation
- [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md) - Daily workflow guide
- `self-hosted/ir/serialize.sio` - Reference implementation
