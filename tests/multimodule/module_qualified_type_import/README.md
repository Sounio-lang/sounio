# A path-form TYPE import must still register its module-qualifier suffix

Regression fixture for `self-hosted/compiler/module_frontend.sio`'s
`resolve_import_file_path` (and its raw-scanner sibling), which populate
`lower_named_item_terminal_note` (`parser/ast.sio`) -- read back by
`self-hosted/check/check.sio`'s `checker_use_module_note_path`. Driven by
`scripts/ci/madaros_module_qualified_type_import_gate.sh`, via `--check`.

| case    | what it pins |
|---------|--------------|
| `basic` | `main.sio` imports `Widget` from `pkg_mod.sio` via a path-form (no braces) import: `use pkg_mod::Widget;`. This is syntactically identical to a function/value named import (`use a::b::f;`), so the named-item fallback in `resolve_import_file_path` used to record it into the same "this is a named item, not a module" registry a function import uses -- which made `checker_use_module_note_path` skip registering `"pkg_mod::Widget"` as a known use-suffix, so `pkg_mod::Widget::make()` (a real associated-function call) fell back to W044 instead of typechecking. The fix loads the confirmed parent file and checks whether the terminal is actually a struct/enum before noting it. |

**Why `--check` only, not compile-and-run:** confirmed directly that a
native-compiled binary for this fixture crashes (SIGILL, the body-less-
mangled-stub failure class this whole PR is about) even with the checker
fix applied. The checker now correctly accepts `pkg_mod::Widget::make()`,
but resolving it to the right callee ALSO depends on `self-hosted/ir/
lower.sio`'s `callee_path_module_stripped_name`, which has its own,
separate, pre-existing heuristic: when searching for how many leading path
segments form the module qualifier, it refuses to even consider a
candidate length whose LAST included segment looks like a type name
(uppercase, or a known struct/enum) -- a deliberate defensive rule
(`struct_layouts` is not guaranteed populated yet at this point, so a real
type's uppercase spelling is the fallback signal) documented for the
`af_mod::AfNum::quad` case, where `af_mod` alone is the real module and
`AfNum` comes from an unrelated import. That heuristic runs BEFORE the
registry check (`lower_use_module_known`) ever gets a chance to confirm
the LONGER candidate, so for `pkg_mod::Widget::make()` it forces the
shorter candidate (module = `pkg_mod` alone) even though only
`"pkg_mod::Widget"` (not bare `"pkg_mod"`) was ever actually registered as
a suffix -- stripping to the wrong split and mangling to a body-less
`pkg_mod_make` stub. This fixture pins the checker's half of the fix
honestly, rather than claiming a full run-to-completion pass the compiler
cannot yet deliver for this exact shape; fixing the lowerer's heuristic
(preferring the longest REGISTERED candidate over the uppercase guess) is
a separate, riskier change to an already heavily-tuned function, not
rushed here.
