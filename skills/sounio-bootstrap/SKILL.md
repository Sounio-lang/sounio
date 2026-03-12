---
name: sounio-bootstrap
description: "Work on the self-hosted Sounio compiler bootstrap path: self-hosted/main.sio pipeline modes (--lex, --parse, --check, --ir-dump, --native-compile), frontend corpus gates, and bootstrap binary milestones."
---

# Sounio Self-hosted Bootstrap

## Overview

`self-hosted/main.sio` is the entry point for the self-hosted Sounio compiler. It runs the full pipeline — lex → parse → resolve → check → lower → native compile — as a pure `.sio` program executed by the souc JIT binary.

The self-hosted pipeline is separate from the Rust `compiler/` pipeline. Changes here affect the self-hosted path only.

## Workflow

### 1) Use the right mode for your investigation

Invoke as: `$SOUC run self-hosted/main.sio -- <mode> <args>`

| Mode | Output | Use for |
|------|--------|---------|
| `--lex <file>` | Token stream | Lexer bugs, new keyword support |
| `--parse <file>` | "Parse OK: N nodes" | Parser coverage |
| `--resolve <file>` | Resolution log | Symbol resolution bugs |
| `--check <file>` | "Check OK: 0 errors" or errors | Type-check gaps |
| `--ir-dump <file>` | "ir-dump: SOIR v2 binary, N bytes, M functions, K strings" | IR lowering |
| `--ir-roundtrip <file>` | "ir-roundtrip: OK" | IR serialize/deserialize fidelity |
| `--disasm <file>` | Human-readable IR | Lowering correctness |
| `--compile <file>` | SOIR bytecode | Bytecode output |
| `--native-compile <file> -o <out>` | ELF binary | Full pipeline → runnable binary |

```bash
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
$SOUC run self-hosted/main.sio -- --check examples/render/triangle_basic.sio
# Expected: "Check OK: 0 errors"
```

### 2) Fix frontend gaps conservatively

When a corpus file fails `--check`, find the minimal fix in the appropriate module:

- **Type-check failures**: fix in `self-hosted/check/` — never change language semantics, only fill missing dispatch cases
- **Resolve failures**: fix in `self-hosted/resolve/` — add missing symbol table lookups
- **Lower failures**: fix in `self-hosted/ir/lower.sio` — add missing AST node → IR instruction mappings
- **Codegen failures**: fix in `self-hosted/native/codegen.sio` — add missing IR opcode handling

One fix per commit. Run the corpus gate after each fix to confirm no regression.

### 3) Validate against the render corpus

The render corpus is the primary correctness target (7 files):

```bash
bash scripts/sprint56_frontend_corpus_gate.sh
# All 7 render examples must produce "Check OK: 0 errors"

bash scripts/sprint57_ir_fidelity_gate.sh
# IR dump must show >= min function counts per file

bash scripts/sprint58_bootstrap_gate.sh
# Bootstrap binary must produce pixel-correct 128×128 PPM
```

Minimum expected function counts:

| File | Min functions |
|------|--------------|
| `triangle_basic.sio` | 4 |
| `cube_wireframe.sio` | 5 |
| `uncertainty_field.sio` | 4 |
| `causal_dag.sio` | 5 |
| `quaternion_rotation.sio` | 5 |

### 4) Bootstrap binary milestone

The primary bootstrap claim: `examples/render/triangle_basic.sio` compiled via self-hosted frontend → runnable ELF → pixel-correct 128×128 PPM.

```bash
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
$SOUC run self-hosted/main.sio -- --native-compile examples/render/triangle_basic.sio -o /tmp/tri_sh
# Expected: "Native compilation successful"
/tmp/tri_sh > /tmp/tri_sh.ppm && head -3 /tmp/tri_sh.ppm
# Expected: P3 / 128 128 / 255
```

"Pixel-correct" means the PPM header dimensions match. Pixel-level exactness is out of scope for Sprint 58.

## References

- Entry point: `self-hosted/main.sio` — all pipeline modes dispatched here
- Frontend modules: `self-hosted/lexer/`, `self-hosted/parser/`, `self-hosted/check/`, `self-hosted/resolve/`
- IR lowering: `self-hosted/ir/lower.sio`
- Native backend: `self-hosted/native/codegen.sio`, `self-hosted/native/regalloc.sio`, `self-hosted/native/frame.sio`, `self-hosted/native/encode.sio`
- Gates: `scripts/sprint56_frontend_corpus_gate.sh`, `scripts/sprint57_ir_fidelity_gate.sh`, `scripts/sprint58_bootstrap_gate.sh`
- Navigation: `skills/sounio-bootstrap/references/bootstrap-navigation.md`
