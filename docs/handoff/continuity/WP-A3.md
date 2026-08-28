<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-a3
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-a3
-->

# WP-A3 — Madaros: AST specializer in the MULTI-module lane [Opus] (dep: WP-A0; parallel with A1/A2)

## Problem
The phase-1 specializer (`self-hosted/check/specializer.sio`, hooked in `compiler/{main,module_frontend}.sio`) runs ONLY in single-module lanes, by name-safety design. `tests/run-pass/cd_exact_generic_i64.sio` imports the generic engine via `use algebra::cayley_dickson_exact::{...}` — the imported-compile path never specializes, so the checker still sees `CDElementExact<F>` templates and fails: `error[E008] expected CDElementExact / found CDElementExact__T` (+ E011s on `er_*` methods, which are WP-A2's).

## Design constraints (READ FIRST)
- `self-hosted/check/specializer.sio` header documents: the structural type-param recovery heuristic (single-uppercase-letter `TypeNamed` in params/return that isn't a locally declared struct/enum — parser drops `<T>` headers, so params are recovered structurally), the same-name single-instantiation replacement scheme (a 2nd distinct instantiation of one template POISONS it back to baseline — this preserves the E010 compile-fail guard), and FOUR backend miscompile classes the pass's own code must avoid (global aggregate arrays; global Option<Box<T>>; whole-struct stores through *mut; large-struct SRET self-assignment). Respect all of it — the pass compiles under the lean_single-built bundle and these are real.
- The multi-module hazard the single-module restriction was guarding: imported type NAMES (e.g. a real imported struct named `F` or `T`) could be misread as type params. Two acceptable designs — pick ONE and justify in the PR:
  (a) run the pass AFTER module merge (where the full item list including imported structs/enums is available) so the "locally declared struct/enum" exclusion covers imported names too;
  (b) restrict specialization to fns whose own declaring module marks them generic (recover the param list in the module where the fn is declared, then specialize at the merged level).
- Hook points: `compiler/module_frontend.sio` — the phase-1 hooks sit in `preflight_multimodule_frontend` (~5492/5508) and `load_multimodule_ir_traced` (~5548/5581) single-module branches; the multi-module path runs `imported_compile` / `module_frontend_check_items_with_source_context` over the merged item list. Find where the merged list exists before type checking and insert `specialize_generics(items)` there.

## Witnesses
W1: `MADAROS_RAW_BIN=<build> ./bin/madaros compile tests/run-pass/cd_exact_generic_i64.sio -o /tmp/x.elf` → NO `E008 ... CDElementExact__T` and no other mono-class errors. (E011 on `er_*` methods may remain until WP-A2 lands — that is OK for this WP; E035 may remain until WP-A1 — also OK. Record residual classes verbatim in the scoreboard.)
W2: a small 2-module witness you author: module `m1.sio` with `fn wrap<T>(v: T) -> W<T>` + `struct W<T>{val: T}`, main file `use`s it and calls `wrap::<i64>(9)` → compiles, runs, rc=9 via `.val`.
W3 (no-misfire guard): a 2-module witness where the IMPORTED module declares a real `struct F { x: i64 }` used by the main file — must compile and run unchanged (proves imported single-letter type names aren't eaten as type params).

## Validation battery
- Single-module witnesses stay green: `turbofish.sio` 3/3, `generic_struct_return.sio` compile-clean, `generic_struct_basic/instantiate` unchanged, compile-fail `turbofish_type_arg_arity.sio` still rejected.
- Multi-module smoke: 8-10 existing run-pass tests that `use` stdlib modules and are green on your pre-change build — byte-identical after.
- Umbrella before/after — zero new reds.

## Done criteria
W1–W3 verified; battery green; PR merged; scoreboard + handoff updated.
