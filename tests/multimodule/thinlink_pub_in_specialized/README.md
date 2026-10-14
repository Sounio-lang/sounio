# `pub(in path)` visibility through the specialized multi-module checker

Regression fixture for `self-hosted/check/mod.sio`'s
`check_items_verdict_boot4_with_module_map`. Driven by
`scripts/ci/madaros_pub_in_specialized_gate.sh`, via `--native-compile` (the
real multi-module frontend), whenever a generic instantiation anywhere in
the program routes the whole typecheck through this specialized path
instead of the ordinary per-module checker.

Before this fix, `current_module_id` (used by every `defining_module_id`-keyed
private-visibility check) was stamped per item, but `current_module` (the
real `AstPath`, used only by `pub(in path)`'s
`ast_path_is_prefix(vis.path, current_module)`) stayed fixed at whatever the
caller passed -- always `empty_path()` at every call site. `empty_path()` can
never be a valid non-empty path's descendant, so a `pub(in path)` access
through this specialized path was **always** rejected, no matter how it was
written, whenever an unrelated generic instantiation anywhere routed the
program through it.

Note on module naming: this repo derives a file's module path from its own
file path (`module_loader.sio`'s `file_path_to_module_path`), so a real
package/directory-scoped `pub(in path)` restriction isn't expressible here --
each fixture in this directory is invoked with a bare, zero-slash filename
(`cd` into the fixture directory first) so that derivation lands on the bare
module name (`main`, `callee`) rather than embedding a parent-directory
segment, letting `pub(main)` name the accessing module directly.

| case            | what it pins |
|------------------|--------------|
| `basic`          | `callee.sio` declares `pub(main) fn restricted_value()`; `main.sio` (the module literally named `main`) calls it and must succeed -- REFUSED before this fix, for every caller, unconditionally. |
| `wrong_module`   | Same shape, but restricted to `pub(nobody)` -- a module nothing in this fixture is named. Must still be REFUSED after the fix, proving it's a real check and not `current_module` now always satisfying it. |
