# Bootstrap Navigation

## self-hosted/main.sio Pipeline Stages

```
Source file (.sio)
    │
    ├─ run_lex_pipeline()           → token stream
    │
    ├─ run_parse_pipeline()         → AST (N nodes)
    │       └─ parse_program()
    │
    ├─ run_resolve_pipeline()       → resolved symbols
    │       └─ resolve_program()
    │
    ├─ run_check_pipeline()         → type-checked AST / "Check OK: 0 errors"
    │       └─ typecheck_program()
    │
    ├─ run_ir_dump_pipeline()       → "ir-dump: SOIR v2 binary, N bytes, M functions"
    │       └─ lower_program_to_ir()
    │
    ├─ run_ir_roundtrip_pipeline()  → "ir-roundtrip: OK"
    │       └─ lower → serialize → deserialize → compare
    │
    ├─ run_disasm_pipeline()        → human-readable IR instructions
    │       └─ lower → ir_disasm()
    │
    ├─ run_compile_pipeline()       → SOIR bytecode (.soir)
    │       └─ lower → compile_ir()
    │
    └─ run_compile_pipeline_native()  → ELF binary
            └─ lower_program_to_ir() → compile_native() → write ELF
```

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|---------------|
| Lexer | `self-hosted/lexer/` | Tokenize source → TokenKind stream |
| Parser | `self-hosted/parser/` | Tokens → AST (ExprKind, StmtKind, DeclKind) |
| Resolver | `self-hosted/resolve/` | Symbol table, name → index mapping |
| Type checker | `self-hosted/check/` | Hindley-Milner inference, effect propagation |
| IR lowerer | `self-hosted/ir/lower.sio` | AST → IrModule (IrFunction[], IrInstr[]) |
| IR optimizer | `self-hosted/ir/` | inline, layout, opt_cleanup, opt_strategy |
| Native codegen | `self-hosted/native/codegen.sio` | IrModule → x86-64 machine code |
| Register allocator | `self-hosted/native/regalloc.sio` | Live-interval → physical register assignment |
| Frame builder | `self-hosted/native/frame.sio` | Stack frame layout, prologue/epilogue |
| Encoder | `self-hosted/native/encode.sio` | x86-64 instruction encoding |
| ELF emitter | `self-hosted/native/codegen.sio` | Sections, relocations, ELF header |

## Compiler (Rust) vs Self-hosted Paths

| Aspect | Rust compiler/ | Self-hosted self-hosted/ |
|--------|---------------|--------------------------|
| Entry | `cargo run -- check/run` | `$SOUC run self-hosted/main.sio --` |
| Pipeline | AST → HIR → HLIR → Codegen | AST → IrModule → ELF |
| Optimization | HLIR passes + LLVM | PGO pipeline (sprints 38–52) |
| Gate | `tests/run-pass/*.sio` | `scripts/sprint56–58_*.sh` |
| Modifies | `compiler/` codebase | `self-hosted/` codebase |

## Render Corpus (Sprint 56 validation targets)

```
examples/render/triangle_basic.sio       ← primary bootstrap target
examples/render/triangle_ppm.sio
examples/render/cube_wireframe.sio
examples/render/uncertainty_ppm.sio
examples/render/uncertainty_field.sio
examples/render/causal_dag.sio
examples/render/quaternion_rotation.sio
```

All 7 must pass `--check` with "Check OK: 0 errors" before Sprint 57 IR gates.

## Common Failure Patterns

| Symptom | Likely Module | Fix |
|---------|--------------|-----|
| "Undefined variable X" | resolver | Add X to symbol table in resolve/ |
| "Type mismatch" | check | Add case to typecheck dispatch in check/ |
| "ir-dump: 0 functions" | lower | Missing AST node handler in lower.sio |
| "ir-roundtrip: FAIL" | ir/ir.sio | Serialize/deserialize mismatch for opcode |
| "SEGFAULT" on bootstrap binary | codegen/regalloc | Stack frame or register clobber bug |

## Self-test

The self-hosted compiler has a built-in self-test mode:
```bash
$SOUC run self-hosted/compiler/main.sio -- --self-test
```
Self-tests T1–T27 cover: IR construction, PGO strategies, inlining, layout, const_fold, DCE, regalloc.
Add new self-tests (T28+) when implementing new passes.
