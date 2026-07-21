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

## Defect B — passing a global array by `&!` ref — **FIXED (wave10e, 2026-07-21)**

Passing **any module-level global array** by `&!` reference used to silently
miss BSS (or SIGSEGV on older layouts); direct indexing of the same global
worked, and passing a **local** array by ref to the same callee worked.

Minimal repro:
```sounio
var GG: [f64; 4] = [0.0; 4]
fn wr(a: &![f64; 4]) with Mut { a[0] = 5.0 }
fn main() -> i32 with IO, Mut, Div, Panic { wr(&!GG) print_int(f64_to_bits(GG[0])) 0 }
// was: GG[0] bits stayed 0 (write at BSS+object_header). Now: 4617315517961601024.
```
Root cause (two cooperating bugs):
1. **OpRef of BSS aggregate** LEA'd a stack slot holding the plain BSS address;
   native-v2 `ref_array_*` then does `load[slot] → resolve raw → +object_header`,
   so stores landed at `BSS+header` while `GG[i]` (raw IndexGet) still read
   `BSS+0`.
2. **`var p = &!arr` did not mark `is_ref`**, so IndexSet through the alias used
   the GC-handle array path instead of `ref_array`.

Fix (`self-hosted/ir/lower.sio` + `IR_NATIVE_V2_OBJECT_HEADER_SIZE` in `ir.sio`):
- OpRef/OpRefMut of a BSS aggregate stashes `(bss_addr - object_header)` so the
  `+header` step lands on the true BSS base.
- let/var bindings from a ref type or from an `&`/`&!` RHS call `bind_local_ref`.

Gate: `scripts/ci/madaros_global_array_ref_gate.sh` /
`tests/run-pass/global_array_ref_mut.sio` (cross-fn f64/i64, local+global alias,
direct store control). Unblocks linalg emitters that pass module-level matrices
by ref.

## Note — `print_f64` mangles negatives

`print_f64(-2.0)` prints `-0.000000`. Off the parity path (emitters use
`f64_to_bits`), but worth a separate fix. Do not diagnose f64 values through
`print_f64` — always `f64_to_bits`.

## Migration status

SPECIAL: green on Madaros. STATS: 18/19 (blocked on Defect A). LINALG: Defect B
(global `&!` array ref) is fixed wave10e — remaining linalg blockers are
independent of that ref path. After the compiler fixes ship in the prebuilt
(madaros-prebuilt-refresh), the SPECIAL gate can drop
`SOUNIO_SOUC_ENGINE=lean_single`; STATS stays on lean_single until A lands.
