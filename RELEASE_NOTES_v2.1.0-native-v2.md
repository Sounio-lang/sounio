# Sounio v2.1.0-native-v2 — first source → native-ELF by the modular compiler

The elected modular self-hosted compiler (`main.sio` → `souc` v0.80.0) now compiles
real `.sio` SOURCE to a runnable native x86-64 ELF via `--native-v2-compile`, with
the program's `main()` return value as the process exit code.

    souc-mc --native-v2-compile examples/native_v2_source/fib.sio -o /tmp/fib
    /tmp/fib; echo $?     # => 55

## Verified capability matrix — 17/17 (tests/native_v2_capgate/)
literals (i64/i32) · arithmetic `+ - * / %` + precedence · comparisons · if/else ·
`let` · `var` + mutation · `while` · `for … in a..b` · function calls (1–4 args) ·
recursion (fib(10)=55) · structs (field access) · arrays (index) · bool · floats
(f64 SSE, `(a*b) as i64`). Run `tests/native_v2_capgate/run.sh <souc-mc.elf>`.
A feature counts as supported ONLY if its program PASSes the gate (anti-overclaim).

## The arc (how the wall fell)
1. **SRET-chain fix** (`b4cce3a51`): the source→ELF wall was a chain of frame-sensitive
   SRET large-struct-return miscompiles in the c634b38f bootstrap. Crossed by
   eliminating by-value `IrModule` returns/copies (box/in-place routing) — no rebootstrap.
2. **Correct values** (`4b0f7c3e5`): pinned & fixed the box-deref-store / whole-element /
   explicit-deref miscompiles so `fn main()->i64{N}` returns N.
3. **Breadth**: arith (mul/mod), IrLoadBool, for-loops, 3–4 arg calls, IrFieldGet/Set,
   IrIndexGet/Set, f64 SSE — each gated by the capability suite.
4. **17/17** (`bdbab4e30`): f64-var casts via a runtime-flag conditional `IrFloatToInt`.

## Known limitations (honest)
- **Float-through-aggregate**: a f64 value loaded from a struct field, array element, or
  function return, then cast to int, is NOT yet converted (the IrFieldGet/IndexGet/IrCall
  handlers don't set the runtime float flag) — it returns the raw IEEE-754 bits. The 17
  gate programs don't exercise this path. Fix = set the float flag on those loads (needs
  field/return type info) or port compile-time f64-type tracking (feat/mc-v2-opcodes f755ff8a7).
- Struct field indices use a first-byte hash (collides for same-initial field names);
  proper fix = declaration-order layout via struct-type tracking.
- Scope: single-file programs, the tested subset above. Strings, nested generics, traits,
  and the broader stdlib are out of scope for this milestone.

## Bootstrap note
Built by the legacy `bin/souc` (c634b38f lineage). Retiring that bootstrap requires the
modular compiler to self-host through this same native-v2 backend — the next milestone.
