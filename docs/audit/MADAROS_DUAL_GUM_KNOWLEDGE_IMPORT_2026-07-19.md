<!-- docs:meta
topic_id: repo.docs.audit.madaros-dual-gum-knowledge-import-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-dual-gum-knowledge-import-2026-07-19
-->

# Madaros dual gum+knowledge import — check + native run

## Claim

> Under **default Madaros**, a single program that imports **both**
> `epistemic::knowledge` and `epistemic::gum`:
>
> 1. type-checks (`check: OK`);
> 2. **compiles and runs** natively with stdout `DUAL_GUM_KNOWLEDGE_OK`;
> 3. true private cross-module access still reports `error[E175]`.

## Symptom (pre-fix check path, origin/main tip including #1239)

```sio
use epistemic::knowledge::{Epistemic}
use epistemic::gum::{gum_type_a, gum_type_b, gum_combine2, gum_k95}
```

| Program shape | Madaros `check` |
|---|---|
| knowledge alone | OK |
| gum alone | OK |
| both (3 modules) | **51× E175** "function is private in its defining module" |
| lean_single dual | accepted (with unrelated field warnings on some APIs) |
| Root 2 multi-module methods (#1227) | OK (not regressed) |

#1239 (visibility preflight allow-list for `print_f64`/`sqrt`/… builtins) does
**not** fix the E175 class. That change is orthogonal (E137 on missing builtins).

## Root cause (check / #1245)

Multi-module visibility preflight
(`check_modules_verdict_boot4_with_visibility(..., enforce_visibility=true)` in
`self-hosted/compiler/main.sio:run_check_mode`) merges all modules into one
`FnSigTable`, stamping each signature with `defining_module_id`.

Lookup used `fn_sig_table_find`, which returns the **first** free function of a
given name. `epistemic/gum.sio` and `epistemic/knowledge.sio` both define
private helpers:

| Name | Role |
|---|---|
| `chk` | test assert helper |
| `near` | approximate equality |
| `test_combine` | self-test entry |

Closure load order is main → knowledge → gum. Gum's body calls to `chk`/`near`
therefore resolved to knowledge's private sigs (`defining_module_id=1`) while
`current_module_id=2` → false E175. Knowledge alone and gum alone never saw the
collision.

`file_path_to_module_path` keeps only the basename (`knowledge` / `gum`), so
module identity is carried by `defining_module_id`, not path equality alone.

### Fix (check)

1. `self-hosted/check/defs.sio` — add `fn_sig_table_find_prefer_module`:
   - prefer free fn defined in `prefer_module_id`
   - else any **non-private** free fn (cross-module import target)
   - else first free-fn match (legacy; visibility still rejects private cross-module)
2. `self-hosted/check/check.sio` — `checker_fn_sigs_find_inplace` calls the
   prefer-module lookup with `(*c).current_module_id`.

Does **not** set `scalar_kind=2` globally (D5). Does **not** disable
`enforce_visibility`. Does **not** rename stdlib helpers (would hide the
collision class for other dual pairs).

## Symptom (pre-fix run path, post-#1245)

Dual **check** green, but `compile` SEGV'd (exit 139) at:

```
imported_compile: lower_begin
lower_array: seed_begin
```

No ELF emitted. Gate treated run as best-effort WARN.

## Root cause (run / 2026-07-20)

Measured bisect under current-source Madaros (`artifacts/self-hosted/madaros`):

| Program | Compile | Run |
|---|---|---|
| gum alone (`gum_type_*` / `gum_k95`) | OK | `GUM_ALONE_OK` |
| knowledge alone (`Epistemic::measured` only) | OK | OK |
| knowledge alone (`e.val()`) | OK | OK |
| knowledge alone (`e.mean()`) | **SEGV** at `seed_begin` | — |
| knowledge alone (`e.this_method_does_not_exist()`) | **SEGV** at `seed_begin` | — |
| dual with `e.val()` + full gum path | OK | **`DUAL_GUM_KNOWLEDGE_OK`** |
| dual with free `ep_measured` / `ep_val` + gum | OK | **`DUAL_GUM_KNOWLEDGE_OK`** |
| dual witness as shipped (`e.mean()`) | **SEGV** at `seed_begin` | — |
| Root 2 multi-module method gate | OK | OK |

`Epistemic` has **no** `mean` method (canonical accessor: `val` / `ep_val`). The
dual witness called `e.mean()`. Multi-module **check still accepts unknown
methods** (`verdict=0` on `e.this_method_does_not_exist()`); native lower then
null-derefs during external method preseed (`lower_array: seed_begin`). That is
an independent residual (unknown-method not rejected at check / Root 2 hardening).

**Dual multi-module native codegen for real gum+knowledge APIs was already live.**
The SEGV was a false dual-import residual caused by a non-existent method name.

### Fix (run)

1. Witness: `e.mean()` → `e.val()`; drop `//@ check-only`.
2. Gate: dual compile+run with `DUAL_GUM_KNOWLEDGE_OK` is **required** (fail, not WARN).
3. Alone knowledge control also uses `e.val()`.

No change to prefer-module checker lookup. No D5/D1/write_file retouch.

## Negative control preserved

```bash
./bin/souc check tests/multimodule/visibility_fn_private_main.sio
# still: error[E175] function is private in its defining module
```

## Gate

```bash
# rebuild if modular checker source changed:
SOUNIO_BUILD_LOCK=/tmp/sounio-dual-import-build.lock \
  bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros

bash scripts/madaros_dual_import_gate.sh
# → MADAROS_DUAL_IMPORT_GATE_OK
# run twice; both must pass (check + native run)
```

Witness: `tests/run-pass/madaros_dual_gum_knowledge.sio`

## Measured (post-fix, current-source Madaros)

```
run_check_mode: about to check 3 modules
run_check_mode: verdict=0
check: OK

lower_array: final_fn_count 225
imported_compile: lower_done
Compilation successful!
DUAL_GUM_KNOWLEDGE_OK
```

Alone controls and `tests/run-pass/madaros_root2_multimodule_method.sio` remain
verdict=0. Private-fn negative control still E175.

## claims_not_made

- Exhaustive census of all same-named private helpers across stdlib
- Resolution of #854 (EISA E5 multi-module E175 graph) beyond the name-collision class
- ~~**Unknown method calls rejected at multi-module check**~~ — closed by `fix/madaros-unknown-method-check` (associated `Type::method` return typing + E011); gate `scripts/madaros_unknown_method_check_gate.sh`
- Compact emitter / remaining imported-module native defects (D1–D4 residual classes)
- lean_single fixed-point identity after the checker change
- Native run for every dual stdlib pair beyond gum+knowledge
