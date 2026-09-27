# Imported tuple-array callee through the real thin-link unit builder

Regression fixture for `self-hosted/compiler/module_loader.sio`'s
`thin_build_compiled_unit` and the `LOWER_FN_TUPLE_ARR_*` collection it drives
(`self-hosted/ir/lower.sio`). Driven by
`scripts/ci/madaros_thinlink_tuple_arr_import_gate.sh`, via `--native-compile`
(the real multi-module frontend), not a lowerer probe entry point.

| case    | what it pins |
|---------|--------------|
| `basic` | `main.sio` imports `make_pair` (returns `(i64, [f64; 2])`) from `tta_callee.sio` and does arithmetic on its destructured `[f64; 2]` slot. `tta_other.sio`, `use`d FIRST, declares an unrelated PRIVATE `make_pair` (never imported, never reachable from `main`) under the exact same bare name but a DIFFERENT tuple-array shape/mask. Before this review round's fix, the per-unit collector handed each imported module's WHOLE raw item list to the tuple-array collector, not just the one export that module's own import_map entry actually names -- so tta_other's colliding private `make_pair`, collected first (its `use` comes first in `main.sio`), would occupy the shared bare-name slot and the real export's own, different mask would lose the table's "already present" dedup check. `lower_fn_tuple_f64_arrays_collect_one_named` (ir/lower.sio) now matches only the requested export name, so the private, uncalled `make_pair` is never even visited. |

This does not replay the exact historical crash from the reset-erasure bug
`thin_build_compiled_unit`'s own comment describes (that required reverting
to a per-lowering-call reset, which this repo's current code no longer
does): it pins the current, real multi-module path end to end instead of
only a synthetic probe boundary, so a regression that reintroduces per-call
resetting, or that widens collection back to a whole imported module's item
list, has an actual compiled-and-run program to fail against.

A residual, narrower case remains out of scope: two DIFFERENT imports, from
two different modules, that are BOTH genuinely selected (both actually named
in some `use`) and happen to share one bare name. Closing that needs real
module-qualified keying of `LOWER_FN_TUPLE_ARR_*`, not just narrowing which
items get collected -- see `lower_fn_tuple_f64_arrays_collect_owner_priority`'s
own comment in `ir/lower.sio`.
