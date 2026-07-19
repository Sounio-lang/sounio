<!-- docs:meta
topic_id: repo.docs.audit.madaros-dual-gum-knowledge-import-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-dual-gum-knowledge-import-2026-07-19
-->

# Madaros dual gum+knowledge import (false E175) — 2026-07-19

## Claim

> Under **default Madaros**, a single program that imports **both**
> `epistemic::knowledge` and `epistemic::gum` type-checks (`check: OK`).
> True private cross-module access still reports `error[E175]`.

## Symptom (pre-fix, origin/main tip including #1239)

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
**not** fix this. That change is orthogonal (E137 on missing builtins).

## Root cause

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

## Fix (surgical)

1. `self-hosted/check/defs.sio` — add `fn_sig_table_find_prefer_module`:
   - prefer free fn defined in `prefer_module_id`
   - else any **non-private** free fn (cross-module import target)
   - else first free-fn match (legacy; visibility still rejects private cross-module)
2. `self-hosted/check/check.sio` — `checker_fn_sigs_find_inplace` calls the
   prefer-module lookup with `(*c).current_module_id`.

Does **not** set `scalar_kind=2` globally (D5). Does **not** disable
`enforce_visibility`. Does **not** rename stdlib helpers (would hide the
collision class for other dual pairs).

## Negative control preserved

```bash
./bin/souc check tests/multimodule/visibility_fn_private_main.sio
# still: error[E175] function is private in its defining module
```

## Gate

```bash
# rebuild if source changed:
SOUNIO_BUILD_LOCK=/tmp/sounio-dual-import-build.lock \
  bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros

bash scripts/madaros_dual_import_gate.sh
# → MADAROS_DUAL_IMPORT_GATE_OK
# run twice; both must pass
```

Witness: `tests/run-pass/madaros_dual_gum_knowledge.sio`

## Measured (post-fix, current-source Madaros)

```
run_check_mode: about to check 3 modules
run_check_mode: verdict=0
check: OK
```

Alone controls and `tests/run-pass/madaros_root2_multimodule_method.sio` remain
verdict=0. Private-fn negative control still E175.

## claims_not_made

- Native multi-module **run**/codegen correctness for every dual stdlib pair
- Exhaustive census of all same-named private helpers across stdlib
- Resolution of #854 (EISA E5 multi-module E175 graph) beyond the name-collision class
- Compact emitter / imported-module native defects (D1–D4)
- lean_single fixed-point identity after this checker change
