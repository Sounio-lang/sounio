# Bootstrap Gap Analysis — AST-Direct Compiler

**Target**: `render_native_compile_driver_lean.sio` compiles ITSELF to a native ELF binary,
eliminating the Cranelift JIT dependency.

**Date**: 2026-03-14

## Current Status

The AST-direct compiler (`lean_work.sio`, ~2700 lines) can compile `triangle_basic.sio`
pixel-perfect (Sprint 116, 12/12 PASS). It handles: let/var, if/else, while, fn+calls,
return, struct construction/field access, arrays, references, deref-index, f64, casts,
&&/||, string print literals.

## Missing Features for Self-Compilation

### Phase 1 — Bitwise Operators (447 uses)

**Priority: HIGHEST** — used in every emit_byte/machine code emission.

| Operator | Uses | x86-64 |
|----------|------|--------|
| `>>` (shr) | 216 | `SHR RAX, CL` |
| `&` (and) | 224 | `AND RAX, RCX` |
| `\|` (or) | 3 | `OR RAX, RCX` |
| `<<` (shl) | 4 | `SHL RAX, CL` |

**Impl**: Add cases in `ast_compile_expr` ExprBinary handler. ~20 lines.
Shift uses `CL` register (lowest byte of RCX). AND/OR use reg-reg.

### Phase 2 — Match Expressions (59 uses)

**Priority: HIGH** — core dispatch mechanism for AST/opcode handling.

Patterns used in lean_work.sio:
1. **Enum tag match** — `match expr.kind { ExprKind::ExprIntLit => { ... } }` (most common)
2. **Option match** — `match *opt { Some(v) => { ... }, _ => { ... } }` (~10 uses)
3. **Wildcard** — `_ => { ... }` (fallthrough)

**Impl**: Compile as if-else chain:
- Load discriminant from match subject
- For each arm: `CMP RAX, <tag>; JNE next_arm; <body>; JMP end`
- Wildcard arm = final else
- Need: enum tag values for ExprKind, StmtKind, IrOpcode, BinOp, UnaryOp, Option

**Complexity**: Medium. Needs enum variant → integer tag mapping in AstCompileCtx.
~150-200 lines.

### Phase 3 — String Values & Comparisons (163 uses)

**Priority: HIGH** — used for function/struct name lookup.

Current: only `print("literal")` inlined to syscall.
Needed:
- String variables (store pointer+length or static address)
- String equality `name == "vertex"` → inline memcmp or byte-by-byte
- String field access on AST nodes (e.g., `s.name`, `f.name`)

**Impl strategy**: Strings as `(ptr, len)` pairs in two stack slots.
Comparison: known-length inline `cmp` for short strings, `repe cmpsb` for general case.
~200-300 lines.

### Phase 4 — Module System (35 imports)

**Priority: MEDIUM** — but can be BYPASSED for bootstrap.

**Bypass strategy**: Create a single-file bootstrap compiler with all needed type
definitions inlined (no `use` imports). The souc JIT already resolves these modules;
the native compiler just needs the type layouts at compile time.

Two approaches:
- **A. Inline types**: Concatenate needed struct/enum definitions into a single file.
  The bootstrap compiler only needs struct layouts + enum tags, not full module resolution.
- **B. Precompiled headers**: Emit struct/enum metadata as a binary blob that the native
  compiler reads at startup (like a .pch file).

Approach A is simpler and sufficient for first bootstrap.

### Phase 5 — Remaining Small Features

| Feature | Uses | Notes |
|---------|------|-------|
| `with` effects | 39 | Parse and ignore (effects are for type checking, not codegen) |
| Method syntax | ~20 | Rewrite as function calls in a desugar pass |
| Nested generic types | ~10 | Only need Option<Box<T>> pattern |

## Bootstrap Strategy

```
Stage 0 (current):  JIT runs lean_work.sio → compiles triangle_basic.sio → ELF
Stage 1 (target):   JIT runs lean_work.sio → compiles lean_bootstrap.sio → ELF
Stage 2:            lean_bootstrap.elf → compiles lean_bootstrap.sio → ELF (self-hosting!)
Stage 3:            lean_bootstrap.elf → compiles full self-hosted compiler → souc-native
```

Stage 1 requires Phases 1-3 above.
Stage 2 requires Phase 4 (or bypass via single-file).
Stage 3 is the endgame.

## Completed Phases

### Phase 1 — Bitwise Operators: DONE
Added `&`, `|`, `^`, `<<`, `>>` to integer binary op handler. 15 lines. Verified 6/6 operations.

### Phase 2 — Match Expressions: DONE
Added ExprMatch handler supporting:
- PatWildcard (_) — unconditional
- PatBinding (v) — binds scrutinee to name
- PatEnum Some(x) — nullable pointer test (non-zero)
- PatEnum None — nullable pointer test (zero)
- Multi-arm JMP-to-end patching (up to 16 arms)
~100 lines. Verified wildcard + binding patterns.

### Phase 3 — String Values: DEFERRED
Reassessed: `string` type only used in 3 functions (CLI plumbing).
Core compilation uses `Name` struct (`[i8; 128]` + `len`) which is already handled.
Can be bypassed for first bootstrap by refactoring CLI to avoid `string`.

## Remaining Work

### Phase 4 — Single-File Bootstrap (Path A)

**Analysis** (2026-03-14): Only 7 of 35 imports are actually used.
24 imports are completely dead. The real dependencies are:

| Module | Used Items |
|--------|-----------|
| parser::ast | 14 types (Expr, Block, Name, ItemKind, ExprKind, etc.) |
| parser::mod | parse_program_preloaded |
| parser::parser | parser_last_error_count |
| native::encode | CodeBuffer + code_buffer_new + emit_byte |
| ir::ir | ir_empty_name, ir_name_eq, ir_name_is_some, ir_name_is_none |
| io::file_write | io_write_native_binary |
| stdlib | print, print_int, arg_count, get_arg, lex, read_file |

The parser+lexer source is ~6000 lines using only basic constructs
(if, while, let, var, fn, return — no match, no for, no break).

**Strategy**: Create `bootstrap.sio` — single file containing:
1. Inlined AST type definitions (~500 lines)
2. Inlined lexer (~700 lines)
3. Inlined parser (~5000 lines)
4. Inlined CodeBuffer + emit_byte (~200 lines)
5. Inlined ir_empty_name + helpers (~50 lines)
6. The lean compilation engine (~2700 lines)
Total: ~9000 lines, self-contained, no imports.

Then: JIT compiles lean driver → compiles bootstrap.sio → bootstrap.elf
Then: bootstrap.elf compiles bootstrap.sio → bootstrap2.elf (SELF-HOSTING!)

### Remaining Feature Gaps for Single-File Bootstrap

| Feature | Parser Uses | Lean Driver Uses | Status |
|---------|------------|-----------------|--------|
| let/var | 22 | 354 | DONE |
| if/else | 25 | 165 | DONE |
| while | 3 | 36 | DONE |
| fn + calls | 28 | 31 | DONE |
| return | 2 | 24 | DONE |
| struct def/access | 1 | 518 | DONE |
| arrays | ~5 | 42 | DONE |
| bitwise ops | ~10 | 447 | DONE |
| match | 0 | 59 | DONE |
| for | 1 | 0 | TODO |
| string ops | ~20 | 3 fns | TODO |
| `as i8`/usize casts | ~50 | ~100 | DONE |

## Updated Effort Estimate

| Phase | Feature | Lines | Sprints |
|-------|---------|-------|---------|
| 1 | Bitwise ops | 15 | DONE |
| 2 | Match expressions | 100 | DONE |
| 3 | String ops (deferred) | — | — |
| 4a | for loops + break | ~40 | 1 |
| 4b | String builtins (str_len, etc.) | ~60 | 1 |
| 4c | Single-file assembly | ~9000 | 2-3 |
| 4d | Self-hosting verification | — | 1 |
| **Remaining** | | **~9100** | **5-6** |
