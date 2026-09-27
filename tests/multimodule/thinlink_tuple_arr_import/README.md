# Imported tuple-array callee through the real thin-link unit builder

Regression fixture for `self-hosted/compiler/module_loader.sio`'s
`thin_build_compiled_unit` and the `LOWER_FN_TUPLE_ARR_*` collection it drives
(`self-hosted/ir/lower.sio`). Driven by
`scripts/ci/madaros_thinlink_tuple_arr_import_gate.sh`, via `--native-compile`
(the real multi-module frontend), not a lowerer probe entry point.

| case    | what it pins |
|---------|--------------|
| `basic` | `main.sio` imports `make_pair` (returns `(f64, [f64; 3])`) from `tta_callee.sio`; the destructured array's elements must classify as f64, not integer. `tta_callee.sio` and `tta_other.sio` each also declare a private, non-exported `helper` under the same name but a different (tuple-array vs. plain `i64`) signature -- loaded as two separate compiled units of the same program, pinning that one unit's per-name `LOWER_FN_TUPLE_ARR_*` collection does not leak into, or get clobbered by, another unit's collection for a same-named function. |

This does not replay the exact historical crash from the reset-erasure bug
`thin_build_compiled_unit`'s own comment describes (that required reverting
to a per-lowering-call reset, which this repo's current code no longer
does): it pins the current, real multi-module path end to end instead of
only a synthetic probe boundary, so a regression that reintroduces per-call
resetting, or that miscollects an imported callee's mask, has an actual
compiled-and-run program to fail against.
