<!-- docs:meta
topic_id: repo.docs.audit.madaros-duplicate-main-collapse-dispatch-2026-09-27
authority: repo_only
audience: users
last_validated: 2026-09-27
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-duplicate-main-collapse-dispatch-2026-09-27
-->

# Madaros imported-module `main` takeover on specialized collapse — audit and remediation

**Date:** 2026-09-27
**Base:** `main` @ `9c5ffa236`. Line references are to that commit.
**Engines:** see "Engines measured" below. All engines were invoked through `./bin/souc`.
**Status:** compiler fix and exact-output regression gate implemented in PR #2734; exact-head CI is authoritative for acceptance.

## Engines measured

- Madaros built from `2e8b76d31` (`make build-madaros`, md5 `5764851f…`).
- The same build plus PR #2720 (md5 `123e3657…`).
- lean_single as committed.

Both Madaros builds give identical results on every row below. Between `2e8b76d31` and the base,
`self-hosted/` changed only in four tuple-let lowering fixes in `ir/lower.sio`
(`d6b68b3bd`..`816980d5f`), which do not touch entry selection.

The repro was then re-run on Madaros built from the base itself (`9c5ffa236`,
`make build-madaros`, md5 `f427f163…`). It gives the same result: `DEP_MAIN`, rc=0, one
`main` slot with `ic=6`. The control gives `7`, `USER_MAIN`.

## Why this was looked at

`stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio` has its own `fn main`, and
`tests/run-pass/darwin_venlafaxine_xr_pgx_smoke.sio` imports it and has another. The question
was which `main` the executable enters. That test is fine: it runs its own `main`. The general
case is not.

## Minimal reproduction (21 lines across four files)

`docs/audit/repro/dup_main_collapse/`:

```sounio
// lib/dm/dep.sio
pub fn ident<T>(x: T) -> T { x }
fn main() with IO { println("DEP_MAIN") }

// user_generic.sio
use dm::dep::*
fn main() with IO {
    let v = ident::<i64>(5)
    println("USER_MAIN")
}
```

`user_plain.sio` is the control. It imports `lib/dm/plain.sio`, which is the same library with a
non-generic `pub fn seven()` in place of `ident`.

```bash
R=docs/audit/repro/dup_main_collapse
SOUNIO_STDLIB_PATH=$PWD/$R/lib ./bin/souc run $R/user_generic.sio
SOUNIO_STDLIB_PATH=$PWD/$R/lib ./bin/souc run $R/user_plain.sio
```

To run lean_single, copy `lib/dm/` under `stdlib/` and run from the repo root. lean_single
resolves imports against the working directory.

| program | engine | lowering path | `main` slots in merged IR (`SOUNIO_DUMP_MERGED_CALLS=1`) | output | rc |
|---|---|---|---|---|---|
| `user_generic.sio` | Madaros | `specialized_collapse lower_count=1` | one, `ic=6` | **`DEP_MAIN`** | **0** |
| `user_plain.sio` | Madaros | `dep_mode into_acc` | one, `ic=10` | `7`, `USER_MAIN` | 0 |
| `user_generic.sio` | lean_single | | | `USER_MAIN` | 0 |
| `user_plain.sio` | lean_single | | | `7`, `USER_MAIN` | 0 |

The user's `main` never runs, and the program exits 0. In both rows the merged IR holds exactly
one `main`: the into-acc slot holds the user's 10-instruction body, and the collapse slot holds
the dependency's 6-instruction body.

Nine further variants all take the into-acc path and all run the user's `main` on both
engines:
- the dependency `main` small or large;
- declared before or after the other items;
- `pub` or private;
- the user's `main` small or large;
- the imported function called or not.

Only the collapse path picks the wrong one. In these probes, the collapse path was reached
through an explicit instantiation (`ident::<i64>(5)`). The inferred form `ident(5)` of an
imported generic is refused by Madaros at check time (see "Not established").

## Root cause

A function's identity in the merged IR is its **unqualified name**. `main` is the one name
allowed in every module: `compiler/private_fn_identity.sio:289` exempts it from
module-qualification ("every module may have one", `:296`). Each lowering path therefore needs
its own rule for the extra `main`s.

- **Into-acc path (correct).** `lower_dep_program_items_into_acc_with_externs`
  (`ir/lower.sio:25135`) states "Skips dep test `main` (seed owns entry)". The dependency's
  `main` binds by name to the seed's `main` slot, which already has a body, so the dedup body
  lowering never lowers it.
- **Specialized-collapse path (wrong).** `module_frontend_specialized_prepare` builds its
  lowering list with `module_frontend_merge_program_items` (`compiler/module_frontend.sio:6193`,
  used at `:6305`). That function concatenates **every** module's items, entry module first,
  into one list. Nothing in this path removes the dependency `main`s:
  - `spec_dce_unreachable_item_fns` (`check/specializer.sio:2689`) roots at the name hash of
    `main`, so it keeps every item called `main`.
  - The list is then lowered as a single freestanding module, where both `main` items map to
    one name-keyed slot. The body lowered last, the dependency's, is the one left in the slot.
  - The hollow-main guard (`module_frontend_module_has_main_body`, `:6334`) passes, because
    the surviving `main` has a body.
  - Codegen's `find_main_index` (`native/codegen_x86_linux.sio:7021`) and reachability's
    `reach_find_main` (`ir/reachability.sio:53`) each take the first function named `main`,
    and only one exists.

## Exposure in this repository

- 324 modules under `stdlib/` define a top-level `main`, among them `epistemic::knowledge`,
  `compress::huffman` and several `darwin_pbpk` modules.
- 441 files under `tests/`, `examples/` and `stdlib/` import at least one of them, 90 of them
  in `tests/run-pass/`.
- A program is exposed when it both imports such a module and instantiates a generic anywhere
  (any monomorphization in the merged program selects the collapse path).

**Current victims: none found among the run-pass importers.** Each of the 90 was run on
Madaros `5764851f`, one at a time with no harness timeout, recording the lowering path:

| outcome | count |
|---|---:|
| `//@ ignore` / `check-only` / `compile-fail` header, not built | 3 |
| into-acc path, i.e. the user's `main` | 73 |
| refused before lowering (check/privacy errors that predate this; path not reached) | 14 |
| **specialized-collapse path** | **0** |

Three tests missed their `expect-stdout-contains`, none because of this defect:
- `compress_huffman_fixed` takes the into-acc path and dies with SIGSEGV (rc=139).
- `darwin_venlafaxine_xr_pgx_smoke` hits the privacy refusal that PR #2723 fixes.
- `viz_headless` is refused before lowering.

So the defect is latent. It becomes live as soon as a program that imports one of these modules
also instantiates a generic. The 14 programs refused before lowering were not measured on this
axis. Neither were `examples/` and `tests/stdlib/` (181 more importers).

## Remediation implemented

In `module_frontend_specialized_prepare`, the **lowering** list (`specialized_raw`) is now
built from a merge that omits top-level `main` items of `programs[1..count)`. This mirrors
the rule the into-acc path already follows ("seed owns entry").

- Leave the **typecheck** merge (`specialized_tc`, `:6285`) as it is. Dependency `main`s keep
  being type-checked on both paths, so whether a program is accepted does not depend on which
  path it takes.
- With one `main` in the lowering list, the DCE root, the name-keyed slot, reachability and
  codegen all agree without further change.
- **Witness:** `scripts/ci/madaros_duplicate_main_collapse_gate.sh` compiles the
  isolated multi-module fixture under `tests/multimodule/duplicate_main_collapse/`
  with current-source Madaros. Its byte-for-byte stdout comparison requires
  `USER_MAIN` and proves `DEP_MAIN` is absent.
- **Acceptance:**
  - the repro prints `USER_MAIN` under Madaros;
  - every surveyed program on the collapse path keeps or gains its user output;
  - main-bearing programs on the into-acc path compile to byte-identical ELFs, using the
    method of `MADAROS_NO_MAIN_ENTRY_FN0_DISPATCH_2026-09-26.md`.

The typecheck merge might itself be unsound with two `main` items in one list. That has not been
measured; every probe returned typecheck verdict 0.

## Not established

- **Inferred calls to imported generics.** Madaros refuses `let v = ident(5)` and
  `let v: f64 = ident(2.5)`, where `ident` is an imported `pub fn ident<T>`, with E009/E001. It
  does so with or without a dependency `main`. lean_single accepts both. This is a separate
  checker gap and has not been reduced further.
- **A wrong value on the collapse path.** With no dependency `main`,
  `let v = ident::<i64>(5)` followed by `println(v)` prints `0.000000` under Madaros, against
  `5` under lean_single. That resembles the open print-dispatch issue #2354 ("scalar kind lost
  through inferred let"), but it has not been reduced and is not claimed to be the same defect.
