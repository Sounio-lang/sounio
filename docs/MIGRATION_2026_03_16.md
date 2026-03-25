<!-- docs:meta
topic_id: repo.docs.migration-2026-03-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.migration-2026-03-16
-->

# Sounio Migration Guide — VM to devPOD (2026-03-16)

## Repository State

- **Branch**: `main` at `b0f1bea7`
- **Sprints completed**: 225 (1003+ optimizer tests, all FAIL=0)
- **Self-hosted compiler**: 47 functions, ~110KB native ELF

## Active WIP: Boot2g Self-Compilation

**Goal**: Pure Sounio compiling itself to bare-metal x86-64. No C, no JIT.

**Chain**: `stage0.c -> boot2g_v1.elf -> boot2g_v2.elf -> (should be fixed-point)`

**Status**: v1 works. v2 has codegen bug -- tokenizer produces fewer tokens, wrong call-site code.

**Bug**: `&&`/`||` short-circuit in `compile_expr_prec` (boot2g.sio ~lines 642-670).
stage0.c uses explicit false-path (`XOR rax,rax`), boot2g relies on rax=0 propagation.
v2-compiled programs have wrong stack slot loads for function call arguments.

**Reproduce**:
```bash
gcc -o /tmp/stage0 bootstrap/stage0.c
/tmp/stage0 bootstrap/boot2g.sio /tmp/v1.elf && chmod +x /tmp/v1.elf
/tmp/v1.elf bootstrap/boot2g.sio /tmp/v2.elf && chmod +x /tmp/v2.elf
# v1 prints 7, v2 prints wrong:
echo 'fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { print_int(add(3, 4)); print("\n"); 0 }' > /tmp/t.sio
/tmp/v1.elf /tmp/t.sio /tmp/t1.elf && chmod +x /tmp/t1.elf && /tmp/t1.elf
/tmp/v2.elf /tmp/t.sio /tmp/t2.elf && chmod +x /tmp/t2.elf && /tmp/t2.elf
```

**Key files**: `bootstrap/boot2g.sio`, `bootstrap/stage0.c`

## Completed: Epistemic Nuclear Decay Demo

`tests/run-pass/epistemic_nuclear_decay.sio` -- Mo-99/Tc-99m, Bateman+RK4+GUM, 20KB ELF, 6/6 PASS

## Build (no Rust needed)

```bash
SOUC=./bin/souc
$SOUC check self-hosted/ir/ir.sio
$SOUC run self-hosted/compiler/main.sio -- --self-test
```

## Architecture

Native backend: `AST -> ir/lower.sio -> IrModule -> native/codegen.sio -> x86-64 ELF`
4-reg linear-scan regalloc, PGO inlining+layout, 222 algebraic rules, e-graph saturation

## Known Issues

- **JIT OOM**: native-compile grows to 14-35GB RSS (Cranelift). Use `--check`/`--ir-dump`.
- **JIT &! bug**: exclusive ref mutations invisible to caller. Return by value.
- **E017**: second+ `(*ref)[idx]` in same fn body dispatches wrong. One per function.

## Claude Memory

Backed up to `.claude-memory-backup/`. Restore to `~/.claude/projects/<hash>/memory/` on devPOD.
