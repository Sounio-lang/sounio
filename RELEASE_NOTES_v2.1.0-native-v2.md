# Sounio v2.1.0-native-v2 — single-file source → native-ELF by the modular compiler

The modular self-hosted compiler (`self-hosted/compiler/main.sio`, which self-identifies
as **Madares v0.80.0**) compiles a real **single-file** `.sio` SOURCE to a runnable native
x86-64 ELF via `--native-v2-compile`, with the program's `main()` return value as the
process exit code. Build the compiler with the shipped bootstrap (`bin/souc`, a static
`mini_native` ELF) first:

    ulimit -s 1048576
    ./bin/souc self-hosted/compiler/main.sio /tmp/mc.elf && chmod +x /tmp/mc.elf
    /tmp/mc.elf --native-v2-compile examples/native_v2_source/fib.sio /tmp/fib
    chmod +x /tmp/fib && /tmp/fib; echo $?     # => 55

## Verified capability matrix — 32/32 (tests/native_v2_capgate/), re-verified 2026-06-07
literals (i64/i32) · arithmetic `+ - * / %` + precedence · comparisons · if/else ·
`let` · `var` + mutation · `while` · `for … in a..b` and array literals · function
calls (1–6 args) · recursion (fib(10)=55) · structs (field access) · arrays (index) ·
bool · floats (f64 SSE, `(a*b) as i64`) · i128 wide-int arithmetic incl. multi-limb
div/mod. Run `tests/native_v2_capgate/run.sh <mc.elf>`.
A feature counts as supported ONLY if its program PASSes the gate (anti-overclaim).

Independently verified this release: linear types reject double-consume (use-once);
multi-module link + run (`native_v2_multimodule_gate` 9/9, one documented import-
typecheck-bypass class); cross-compile to macOS arm64 Mach-O and host x86_64 ELF via
`bin/souc … --target aarch64-macos`; backend field-store soundness (40/40, 1 tracked
field-hash residual, see below).

## The arc (how the wall fell)
1. **SRET-chain fix** (`b4cce3a51`): the source→ELF wall was a chain of frame-sensitive
   SRET large-struct-return miscompiles in the c634b38f bootstrap. Crossed by
   eliminating by-value `IrModule` returns/copies (box/in-place routing) — no rebootstrap.
2. **Correct values** (`4b0f7c3e5`): pinned & fixed the box-deref-store / whole-element /
   explicit-deref miscompiles so `fn main()->i64{N}` returns N.
3. **Breadth**: arith (mul/mod), IrLoadBool, for-loops, 3–4 arg calls, IrFieldGet/Set,
   IrIndexGet/Set, f64 SSE — each gated by the capability suite.
4. **f64-var casts** (`bdbab4e30`): runtime-flag conditional `IrFloatToInt` (reached the
   first full capgate green at 17/17).
5. **Extension to 32/32**: 5–6 arg calls, `for … in` array literals, and i128 wide-int
   arithmetic incl. multi-limb div/mod — each gated by the capability suite.

## Known limitations (honest)
- **Scope is single-file.** `--native-v2-compile` compiles one source file. Multi-module
  programs are linked + run via a separate path (`native_v2_multimodule_gate`), which still
  has a tracked import-typecheck-bypass class (ill-typed cross-module programs can slip).
- **No self-compilation.** The modular `main.sio` compiler does NOT yet compile itself to
  a native ELF via this backend. Only the legacy `lean_single.sio` lane reaches a
  bit-identical bootstrap fixed point. mc self-hosting is the next milestone.
- **Int literals need an explicit cast.** `fn main() -> i32 { 0 }` is rejected (`found i64`);
  write `{ 0 as i32 }`. Literal coercion to `i32` is in progress.
- **Closures / generics / nested control checks.** A lambda passed as a `fn` argument fails
  typecheck and the backend does not reliably compile lambdas — closures are NOT supported
  end-to-end. Generics, nested `if let`, and `while let` currently false-reject.
- **Epistemic ctor is prototype.** `Knowledge(15.0, ε=0.92, prov=...)` and `ε >= 0.82`
  confidence gates do not typecheck/enforce on mc; single-module `--check` does resolve-skip,
  so these forms parse but are not validated.
- **Field-hash residual (disclosed).** Struct field slots use a first-byte hash that can
  collide for fields sharing an initial byte. `native_v2_backend_soundness_gate` is 40/40
  with exactly 1 tracked residual (`C_known_residual_bucket_collision`). Proper fix =
  declaration-order layout via struct-type tracking.
- **Float-through-aggregate**: a f64 value loaded from a struct field, array element, or
  function return, then cast to int, is not yet converted — it returns the raw IEEE-754
  bits. The gate programs don't exercise this path.
- Strings, nested generics, traits, and the broader stdlib are out of scope for this
  milestone. Roughly 252 of ~860 example programs are currently green.

## Bootstrap note
Built by the shipped bootstrap binary `bin/souc` — a static `mini_native` ELF using the
raw `souc <source.sio> <output> [flags]` interface (NOT a launcher with subcommands).
Retiring that bootstrap requires the modular compiler to self-host through this same
native-v2 backend — the next milestone.
