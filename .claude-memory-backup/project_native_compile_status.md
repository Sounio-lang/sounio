---
name: native_compile_pipeline_status
description: Lean driver — Sprint 111 gate 9/9 PASS; ExprCast/Field/Index added; produces correct ELFs for primes/FizzBuzz/Ackermann/isqrt
type: project
---

# Self-hosted Native Compile Pipeline Status

## MILESTONE: Sprint 112 — Array Indexing (2026-03-13, commits fcf52823 + 2038efc0)

Array indexing fully working in lean driver:
- `var arr: [i64; N] = [val; N]` — allocates N consecutive stack slots, initializes all
- `arr[i]` read — ExprIndex: compile index, compute [rbp - 8*(base+1)] - index*8, load
- `arr[i] = v` write — StmtAssign ExprIndex target: push val, compute addr, pop rdx, store
- Frame size: 8192 bytes (up from 512, supports up to ~1024 slots)
- Linter also added short-circuit `&&` (OpAnd) and `||` (OpOr) in same commit

Gate: `scripts/sprint112_array_gate.sh` — 5/5 PASS
Tests: sieve_200 (46 primes), array_sum=55, bubble_sort (min=0 max=9), prefix_9=55, dp_fib30=832040

## MILESTONE: Sprint 111 — 9/9 Gate PASS (2026-03-13, commit f0a9d010)

AST-direct lean driver (`render_native_compile_driver_lean.sio`) — Sprint 111 gate:

| Test | Expected | Status |
|------|----------|--------|
| hello_world | "The answer is: 42" | PASS |
| primes_100 | prime_count=25 | PASS |
| fizzbuzz | FizzBuzz at n=15 | PASS |
| fibonacci | fib(10)=55 | PASS |
| gcd_lcm | lcm(12,18)=36 | PASS |
| collatz | collatz(27)=111 | PASS |
| hanoi | hanoi(10)=1023 | PASS |
| ackermann | A(3,4)=125 | PASS |
| isqrt | isqrt(10000)=100 | PASS |

**Gate script**: `scripts/sprint111_lean_driver_gate.sh`

## Lean Driver Capabilities (Sprint 112)

**Supported:**
- ExprIntLit, ExprBoolTrue/False, ExprStringLit (inline print)
- ExprIdent (local variables, up to 64)
- ExprBinary (all ops: +,-,*,/,%, ==,!=,<,<=,>,>=)
- ExprUnary (neg, not)
- ExprCall (up to 4 args; special: print(str), print_int(n))
- ExprIf (if/else chains, nested)
- ExprWhile
- ExprReturn, ExprBlock
- ExprCast (no-op for int casts — Sprint 111)
- ExprFieldAccess (CodeBuffer.len@65536, Name.len@128 — Sprint 111)
- ExprIndex (WORKING — dynamic index into local [i64; N] arrays — Sprint 112)
- ExprArrayLit repeat form [val; N] (WORKING — initializes N consecutive slots)
- Short-circuit `&&` / `||` (WORKING — Sprint 131 linter addition)

**I/O builtins:**
- `print_int(n)` — 130-byte precompiled helper
- `print("literal")` — inline jmp-over + sys_write + RIP-relative LEA
- `println("literal")` — same + newline

**Output:** 4–5KB ELF binaries, no sections, single LOAD segment

**Import surface:** 8 modules (down from 24)

## Path to Self-Compilation (Sprint 112+)

Blocked by: struct-aware stack allocation for large value types.
- `CodeBuffer { bytes: [i8; 65536], len: i64 }` — 65544 bytes, can't be a stack slot
- Need heap allocation OR pointer-passing convention for large structs
- Alternative: write a stripped-down version of lean driver using only i64 values

## Sprint 111 SOTA comparison

- LLVM: handles full C with all types
- Sounio lean driver: handles integer programs, recursion, arithmetic, string literals
- Lean driver is deterministic, no JIT instability, produces correct ELFs in ~5s
- Next frontier: array indexing (ExprIndex with struct-aware layout)

## Full Pipeline Status

`native_compile_driver.sio` (full codegen path):
- 9/9 basic programs PASS (exit-code tests)
- OOMs on programs with branches (JIT ceiling)
- Not viable for bootstrap

## Key Files

- `self-hosted/compiler/render_native_compile_driver_lean.sio` — lean AST-direct driver
- `scripts/sprint111_lean_driver_gate.sh` — 9/9 gate
- `/tmp/lean_driver_final_v4.sio` — backup before Sprint 111 (v4)
- `/tmp/lean_v5.sio` — current working v5 backup
