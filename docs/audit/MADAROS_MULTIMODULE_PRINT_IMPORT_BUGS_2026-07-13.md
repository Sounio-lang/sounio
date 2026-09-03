<!-- docs:meta
topic_id: repo.docs.audit.madaros-multimodule-print-import-bugs-2026-07-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-multimodule-print-import-bugs-2026-07-13
-->

# Madaros v0.80.0 — two multi-module defects (named import + print_f64)

**Date:** 2026-07-13
**Toolchain:** `./bin/souc` → Madaros v0.80.0
**Owner:** CODEX-2 (`self-hosted/compiler/main.sio`, `run_check_mode` / visibility-preflight pass)
**Discovery context:** hardening `epistemic::gum`; both defects are compiler-side, not stdlib.
Forensic dispatch per CLAUDE.md §8 — do not patch `self-hosted/` ad hoc.

## Defect 1 — single-symbol `use module::symbol` fails E137 even for `pub` symbols

`use epistemic::gum::gum_type_b_uniform` (a symbol that IS `pub fn` in `stdlib/epistemic/gum.sio`) fails:
```
error[E137] use of undeclared variable
  = help: declare the variable before use, or import it from another module
run_check_mode: verdict=1
```
The **wildcard** form of the same import works:
```sounio
use epistemic::gum::*          // resolves; program compiles to ELF and runs
```
So named single-symbol import resolution is broken while wildcard import is fine. Publishing helpers does
not change this (the failing symbol was already `pub`).

**Workaround (in use):** import stdlib modules with `use module::*`.

## Defect 2 — `print_f64` trips spurious E137 in any 2+-module (importing) program

A `main` file that has at least one `use ...` import and calls `print_f64(x)` fails the same
`E137`/visibility-preflight check. The overloaded `print(f64)` / `println(f64)` builtins print floats
correctly in the same importing program.

Evidence it is `print_f64`-specific, not the import:
- `use epistemic::gum::*` + `print_f64(gum_std_u(r))` → **E137**.
- `use epistemic::gum::*` + `print_int((uc*1e6) as i64)` → compiles + runs (`290401`).
- `use epistemic::gum::*` + `print(gum_std_u(r))` / `println(gum_value(r))` → compiles + runs
  (`0.290401`, `val=98.299999`).
- Green multi-module `stdlib/clinical/vancomycin_pbpk.sio` (imports `epistemic::knightian`) prints floats
  and runs — with **zero** `print_f64` calls (uses `println(f64)`).

**Workaround (in use):** in importing programs print floats with `print(f64)`/`println(f64)`, never
`print_f64`.

## Defect 3 — a user-defined helper fn in an importing program trips visibility-preflight

A `main` file with `use ...` **and** any second user-defined function fails:
```
run_check_mode: verdict=1   (note: "function arguments are checked against the declared parameter types")
```
The same logic **inlined into `main`** (no helper fn) compiles and runs. Verified: `use epistemic::gum::*`
+ `fn near(...)` + `main` → fail; identical program with the comparison inlined → runs.

**Workaround (in use):** in importing programs, inline all logic into `main` — no user helper functions.
This forces verbose drivers but works.

## Defect 4 (adjacent, pre-existing) — `gum.sio`'s embedded self-test segfaults at runtime

`stdlib/epistemic/gum.sio` is annotated `//@ run-pass` / `//@ expect-stdout: ALL PASS`, but compiled and
run standalone it **segfaults (exit 139)** after printing `─── stdlib/epistemic/gum tests ───\n  PASS [`.
This is **pre-existing** and independent of the GUM hardening: the pre-session file and the hardened file
produce **byte-identical output** (same md5) and both exit 139. So `expect-stdout: ALL PASS` is stale —
the file has not run clean on Madaros v0.80.0. The GUM *functions* are fine (an importing driver exercises
them and asserts correct ISO-GUM values — `tests/stdlib/epistemic/test_gum_stdlib.sio`); the crash is in
the file's own test `main`/print path. Not chased here (scope + likely codegen-side). The
`epistemic_gum_gate.sh` gate therefore verifies the module via the **importing driver**, not by running
`gum.sio`'s embedded self-test.

## Impact
Neither blocks the epistemic/GUM hardening (both have clean caller-side workarounds), but both are
papercuts that make stdlib modules feel unusable to newcomers (the natural `use mod::name` + `print_f64`
both fail). Fixing them would materially improve real-world usability of the whole stdlib.

## Closeout 2026-08-06

Shipped Madaros (post-#1627 tip promote) accepts the #862 triad:

- named `use epistemic::knowledge::ep_measured`
- user helper `fn twice(x: f64) -> f64`
- `print_f64(...)`

Gate: `scripts/ci/madaros_862_import_print_gate.sh` → `MADAROS_862_GATE_OK`.
Fixture: `tests/epistemic_trust/madaros_862_import_print_accept.sio`.

Root already in checker source (`checker_collect_runtime_builtins_inplace` binds
`print_f64`; path-form named import binds last segment) — promote made it live
in `bin/madaros-linux-x86_64`. Issue #862 closable.

## Repro (self-contained)
```sounio
// FAILS (E137): single-symbol import
use epistemic::gum::gum_type_b_uniform
// FAILS (E137): print_f64 in an importing main
use epistemic::gum::*
fn main() -> i32 with IO, Mut, Div, Panic {
    print_f64(gum_std_u(gum_combine2(98.3, gum_type_a(0.0707,5), gum_type_b_uniform(0.5))))
    return 0
}
// WORKS:
use epistemic::gum::*
fn main() -> i32 with IO, Mut, Div, Panic {
    let r = gum_combine2(98.3, gum_type_a(0.0707,5), gum_type_b_uniform(0.5))
    print("u_c="); println(gum_std_u(r))     // 0.290401
    return 0
}
```
