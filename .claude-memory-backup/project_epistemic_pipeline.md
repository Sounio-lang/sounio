---
name: Epistemic computing pipeline status
description: Progress toward running real scientific programs with Knowledge<T> — what's done, what's next
type: project
---

## Epistemic Computing Pipeline (2026-03-13)

### Completed (Sprint 115/116/118, commit 08587b5d)

1. **sqrt builtin** (Sprint 115): Hardware SQRTSD instruction, fully wired as native builtin. Gate: 11 PASS, 0 FAIL.

2. **Knowledge<f64> struct layout** (Sprint 116): 3-field struct (value, variance, confidence) registered in StructLayoutTable. `measure()` now emits IrAlloc + IrFieldSet×3 (variance = uncertainty²). Gate: 15 PASS, 0 FAIL.

3. **print_f64 builtin** (Sprint 118): Multiply-and-truncate algorithm with 6-digit fractional output. 1e6 constant in .rodata. Gate: 12 PASS, 0 FAIL.

Tests T400-T405; total=405.

### Remaining for MVP scientific program

4. **GUM arithmetic** (Sprint 117 — next): knowledge_add/sub/mul/div as stdlib functions using existing f64 ops. Depends on 115+116 (done). Design: regular functions, no new IR opcodes.

5. **exp/log/sin/cos/pow**: Minimax polynomial approximations as native builtins (same pattern as sqrt). Each ~100-200 bytes of x86-64.

6. **Heap allocation**: Currently IrAlloc uses fixed-size bump allocator. Needed for dynamic collections beyond Knowledge structs.

7. **FFI/extern calls**: Sprint 108 extern keyword parsed, codegen incomplete. Needed for BLAS/LAPACK.

8. **Unit enforcement**: Parser accepts units, type checker does zero dimensional analysis.

### Architecture decisions

- **Soft-float over dynamic linking**: All math builtins are self-contained x86-64 sequences. No PLT/GOT, no runtime deps beyond kernel syscalls.
- **Knowledge as regular struct**: Uses existing IrAlloc + IrFieldGet/Set. No custom IR opcodes.
- **Explicit functions over operator overloading**: MVP uses `knowledge_add(a, b)` not `a + b`.

**Why:** Keeps the static ELF model intact and avoids touching the type checker for operator dispatch. Operator overloading can be added later in the lowerer.

**How to apply:** When implementing GUM arithmetic (Sprint 117), write it as regular Sounio functions that extract fields, do f64 math, and construct new Knowledge structs.
