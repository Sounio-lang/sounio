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

## Defect A — imported-module f64 constants read 0 (multi-module) — **FIXED (Wave11e, 2026-07-21)**

A module-level `let K: f64 = <lit>` **in an imported module** used to read 0 at use
sites, even though the identical declaration in the *main* file worked (fix #3).

Minimal repro (pre-fix):
```sounio
// main file: `use stats::densities::*` then print f64_to_bits(DE_LN_SQRT_2PI)
// -> Madaros: 0   |  lean_single: 0.9189385332046727
// lognormal_pdf(1,0,1) -> 1.0 (const wiped → exp(0)) instead of ≈0.3989422804014327
```
Root cause: global initializers are recorded in a process-global side-table
(`GLOBAL_VAR_INIT_*`) at parse time and read at lower time. The table was reset on
every per-module parse, so the last-parsed module wiped earlier ones.

**Fix (Wave11e):** the live Madaros multi-mod paths
`module_frontend_compile_imported_to_file` and `load_multimodule_ir_to_box_traced`
now bracket load+lower with `ast_reset_global_var_inits()` once and
`GLOBAL_VAR_INIT_SUPPRESS_RESET=1`, restoring suppress=0 on every exit. (The older
`module_loader::preflight_multimodule_frontend` already had this; the modular
frontend path that actually emits ELFs did not.)

Gate: `scripts/ci/madaros_imported_f64_const_gate.sh`
→ `MADAROS_IMPORTED_F64_CONST_GATE_OK` (minimal leaf+pad witness + lognormal science).

Residual / future: carrying init words on the AST/Program (no process-global table)
would be more robust under capacity pressure (`GLOBAL_VAR_INIT_*` is still 128 slots)
and re-parse-heavy probe paths; not required for the measured science vertical.

## Defect A′ — multi-mod BSS offset collision (two modules with BSS scalars) — **FIXED (wave11, 2026-07-21)**

When **two imported modules** each own a module-level f64/i64 BSS global, each is
lowered with **module-local** BSS offset 0. Multi-mod merge used to append functions
**without** relocating `param_count` / `IrLoadGlobal`/`IrStoreGlobal` immediates by
`acc.bss_total_size`, so both slots and both loads shared `BSS_BASE+0`. Last init wins:

```
// a.sio: let A_CONST: f64 = 1.5
// b.sio: let B_CONST: f64 = 2.5
// was: both get_*_bits() → 2.5 bits | now: 1.5 and 2.5 distinct
```

**Fix (wave11 Agent D, PR #1382):** `ir_merge_place_and_remap_function` /
`ir_merge_modules_into` add `bss_offset_delta = dst.bss_total_size` to BSS slot
offsets and global load/store immediates (and absolute `BSS_BASE+off` `IrLoadImm`),
then `dst.bss_total_size += src.bss_total_size`. Complements Wave11e Defect A
suppress (`GLOBAL_VAR_INIT` accumulate) — both are required for multi-import science
with per-module constants.

Gate: `tests/run-pass/imported_module_f64_const.sio` (A then B, both nonzero and
distinct) via `scripts/ci/madaros_imported_f64_const_gate.sh`.

Status: **CLOSED wave11** (source fix). **Prebuilt refreshed Wave12e (2026-07-21)** —
`bin/madaros-linux-x86_64` now carries the remap; default `bin/souc` passes
`scripts/ci/madaros_imported_f64_const_gate.sh` and the Wave12 tip-green lock
(`scripts/dev/madaros_wave12_tip_green_gate.sh`, gate `imported_f64`).

Residual closed Wave13 (2026-07-21): bare `use m::{CONST}` Ident of a global
**from main** — seed now preseeds external BSS after own items
(`lowerer_preseed_external_bss_globals_mut`); merge DEDUPs BSS by name and
resolves `IrLoadGlobal` by name. Gate arm:
`tests/run-pass/imported_module_f64_const_bare_ident.sio` via
`scripts/ci/madaros_imported_f64_const_gate.sh`.
Audit: `docs/audit/MADAROS_WAVE13_BARE_CROSSMOD_F64_IDENT_2026-07-21.md`.

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

## Note — `print_f64` mangles negatives — **CLOSED (Wave15 B, 2026-07-22)**

Was: `print_f64` of any negative printed `-0.000000` (sign kept, magnitude
zeroed). Root cause and fix landed earlier as #1286 / #890
(`emit_builtin_print_f64` reloads abs bits from `xmm0` after the `'-'` write —
rdi held stdout fd). Residual closeout: dedicated `print_f64` witness + gate.

- Witness: `tests/run-pass/print_f64_negative.sio` — cases `-0.0`, `-2.0`,
  `-0.5`, positive `+2.0`; `f64_to_bits` oracle; `print_int` control.
- Gate: `scripts/ci/madaros_print_f64_negative_gate.sh` →
  `MADAROS_PRINT_F64_NEGATIVE_GATE_OK` (also re-runs
  `tests/run-pass/println_f64_negative.sio`).
- Audit: `docs/audit/MADAROS_PRINT_NEGATIVE_F64_2026-07-14.md` (FIXED 2026-07-20).

Parity emitters may still prefer `f64_to_bits` for exact equality; display of
negatives is now trustworthy under default Madaros.

## Migration status

SPECIAL: green on Madaros. STATS: Defect A closed Wave11e (lognormal_pdf const);
Defect A′ multi-mod BSS remap closed wave11 (A_CONST/B_CONST distinct). LINALG:
Defect B (global `&!` array ref) is fixed wave10e — remaining linalg blockers are
independent of that ref path. After the compiler fixes ship in the prebuilt
(madaros-prebuilt-refresh), the SPECIAL/STATS gates can drop
`SOUNIO_SOUC_ENGINE=lean_single` where they still pin it.

## Defect A″ — into-acc f64 BSS arithmetic loses float mark (Wave15 D 2026-07-22) — **FIXED**

Distinct from A (const wiped → 0 → pdf=1.0) and A′ (offset collision). Const
**init and bits** were correct; same-module `DE + 1.0` / `0.0 - DE` inside an
imported module did `cvtsi2sd` on IEEE bits (~4.6e18), so
`lognormal_pdf(1,0,1)` returned `~1e-300`. Root: into-acc
`lowerer_from_acc_module` empty `global_types` + skip re-record when seed already
preseeded the BSS slot. Fix + gate:
`docs/audit/MADAROS_IMPORTED_F64_BSS_ARITH_2026-07-22.md`,
`scripts/madaros_imported_f64_bss_arith_gate.sh`.
