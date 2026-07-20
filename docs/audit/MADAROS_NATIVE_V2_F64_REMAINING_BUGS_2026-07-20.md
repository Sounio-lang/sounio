<!-- docs:meta
topic_id: repo.docs.audit.madaros-native-v2-f64-remaining-bugs-2026-07-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-native-v2-f64-remaining-bugs-2026-07-20
-->

# Madaros native-v2 f64 codegen — remaining defects (2026-07-20)

**Toolchain:** `./bin/souc` → Madaros v0.80.0 (native-v2 multi-module path,
`codegen_x86_linux`).
**Context:** migrating the SciPy↔Sounio parity harness (special/stats/linalg)
off the `lean_single` seed onto the official Madaros engine. Five native-v2 f64
defects were found and **fixed** (see "Fixed" below); the SPECIAL vertical (48
functions) is now fully green on Madaros. Two defects remain, each a distinct
deep native-v2 issue. All diagnostics use `f64_to_bits` + `print_int` (NOT
`print_f64`, which has its own negative-value display bug — see note).

## Fixed in the same work (for context)

1. `f64_to_bits`/`bits_to_f64` had no native codegen emitter → calls defaulted to
   fn 0 (the runtime entry) and segfaulted. Registered as builtins 29/30 with an
   identity emitter (`mov rax, rdi`). — `codegen.sio`, `codegen_x86_linux.sio`,
   `native_compile_driver.sio`.
2. Visibility (E175): the special functions in bessel/airy/zeta/elliptic/
   hypergeometric/orthopoly were all private; made the 33 API functions `pub`.
3. Module-level `let X: f64 = <lit>` (main file) recorded only `ExprIntLit`
   (int_val=0 for floats) → read 0. Now records `f64_to_bits(float_val)`. —
   `parser/items.sio`.
4. `let t = <var>` aliased the source local's register for immutable bindings, so
   a later in-place mutation of the source `var` corrupted the binding
   (`let t=a; a=a+1; use t` returned the new `a`). Now always snapshots a scalar
   bare-ident binding into a fresh register. — `ir/lower.sio` `lower_let_stmt_ref`.
   Fixes AGM-style loops (elliptic) and value capture in general.

## Defect A — imported-module f64 constants read 0 (multi-module)

A module-level `let K: f64 = <lit>` **in an imported module** reads 0 at use
sites, even though the identical declaration in the *main* file now works (fix #3).

Minimal repro:
```sounio
// main file: `use stats::densities::*` then print f64_to_bits(DE_LN_SQRT_2PI)
// -> Madaros: 0   |  lean_single: 0.9189385332046727
```
Root cause: global initializers are recorded in a process-global side-table
(`GLOBAL_VAR_INIT_*`) at parse time and read at lower time. The table was reset on
every per-module parse, so the last-parsed module wiped earlier ones.

Status: **partial fix landed** — `preflight_multimodule_frontend` now resets the
table once and sets `GLOBAL_VAR_INIT_SUPPRESS_RESET=1` so main + imports
accumulate. This does **not** land the fix because the native-v2 compile path
(`compile_multimodule_native_advanced` → `module_frontend::load_multimodule_ir` →
`load_module_file` → `sourcefile_parse_program_loaded`) parses through a path that
does not run under the preflight bracket, so the table the lowering reads is still
per-module. A robust fix likely carries the init value on the parsed AST/Program
(so it travels with the module to lowering) instead of a reset-prone global table,
or brackets every native-path load loop. Blocks: `lognormal_pdf` (stats).

## Defect B — passing a global array by `&!` ref computes a wrong address

Passing **any module-level global array** by `&!` reference segfaults; direct
indexing of the same global works, and passing a **local** array by ref to the
same callee works.

Minimal repro:
```sounio
var GG: [f64; 4] = [0.0; 4]
fn wr(a: &![f64; 4]) with Mut { a[0] = 5.0 }
fn main() -> i32 with IO, Mut, Div, Panic { wr(&!GG) print_int(f64_to_bits(GG[0])) 0 }
// -> rc=139 (SIGSEGV). Same with a LOCAL array: rc=0.
```
Root cause: the ref path (`ir/lower.sio` ~10398, `IR_STRATEGY_BSS_GLOBAL` branch)
emits `ir_load_imm(BSS_BASE_LINUX + functions[fn_id].param_count)` as the array's
address. `addr(&!GG)` returns a plausible-looking high address, but writing through
it faults — so `param_count` is stale/wrong relative to the final BSS layout that
direct access and local-array refs use. Fix: source the global's BSS offset from
the same tracking that direct access uses, or reconcile `param_count` with the
final layout. Blocks: `eigen` (linalg — the emitter passes module-level
`[f64;65536]` matrices by ref).

## Note — `print_f64` mangles negatives

`print_f64(-2.0)` prints `-0.000000`. Off the parity path (emitters use
`f64_to_bits`), but worth a separate fix. Do not diagnose f64 values through
`print_f64` — always `f64_to_bits`.

## Migration status

SPECIAL: green on Madaros. STATS: 18/19 (blocked on Defect A). LINALG: blocked on
Defect B. After the compiler fixes ship in the prebuilt (madaros-prebuilt-refresh),
the SPECIAL gate can drop `SOUNIO_SOUC_ENGINE=lean_single`; STATS/LINALG stay on
lean_single until A and B land.
