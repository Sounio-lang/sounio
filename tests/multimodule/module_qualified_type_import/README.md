# A path-form TYPE import must still register its module-qualifier suffix

Regression fixture for `self-hosted/compiler/module_frontend.sio`'s
`resolve_import_file_path` (and its raw-scanner sibling), which populate
`lower_named_item_terminal_note` (`parser/ast.sio`) -- read back by
`self-hosted/check/check.sio`'s `checker_use_module_note_path` and
`self-hosted/ir/lower.sio`'s `callee_path_module_stripped_name`. Driven by
`scripts/ci/madaros_module_qualified_type_import_gate.sh`, compile-and-run.

| case    | what it pins |
|---------|--------------|
| `basic` | `main.sio` imports `Widget` from `pkg_mod.sio` via a path-form (no braces) import: `use pkg_mod::Widget;`. This is syntactically identical to a function/value named import (`use a::b::f;`), so the named-item fallback in `resolve_import_file_path` used to record it into the same "this is a named item, not a module" registry a function import uses -- which made `checker_use_module_note_path` skip registering `"pkg_mod::Widget"` as a known use-suffix, so `pkg_mod::Widget::make()` (a real associated-function call) fell back to W044 instead of typechecking. The fix (comment 4114124500) loads the confirmed parent file and checks whether the terminal is actually a struct/enum before noting it. |
| `oversized` (generated at gate run time, not checked in -- see the gate script) | Same shape as `basic`, but the module file is padded past 1 MiB before its `pub struct Widget` declaration, mirroring real files in this tree (`self-hosted/check/check.sio`, `self-hosted/ir/lower.sio`) that already exceed 1 MiB. Comment 4114530613: the byte scanner backing the check above was first (wrongly) capped at 1 MiB on the mistaken assumption that `read_file` itself has that limit -- it doesn't (`emit_builtin_read_file`, self-hosted/native/codegen_x86_linux.sio, mmaps a buffer sized to the file's real `st_size`); that wrong cap then made a *second* mistake, "return true (assume type) past the cap", which classified every large-file function/value import as a type too, without ever looking -- reopening the same "qualified call binds an unrelated same-named symbol" class of bug from the other direction. Fixed properly: scan the file's real, full size up to the lexer's own actual 16 MiB ceiling (`lex_file_to_globals`, self-hosted/lexer/mod.sio, hard-refuses anything larger, so no module this compiler can otherwise parse exceeds it), and fail closed (assume the terminal could be a type) only past that ceiling, where no real answer is possible because nothing else in the compiler could have parsed the file either. |
| `multiline` (generated at gate run time -- see the gate script) | Same shape again, but `pub struct` and `Widget` are split across a newline (`pub struct\nWidget {`). Copilot review "Handle newlines when scanning path-form type imports": the lexer treats LF/CR as ordinary whitespace, so this is a valid declaration, but the byte scanner's post-keyword whitespace loop skipped only space (32) and tab (9), missing it -- the same misclassification-into-function/value bug, just triggered by a line break instead of file size. Fixed by including LF (10) and CR (13) in that whitespace loop. |

**History -- two more bugs found resolving this fixture to a real
compile-and-run pass, both now fixed:**

1. (comment 4114245220) The struct/enum byte scanner backing the check
   above did not skip `//` line comments or `"..."` string literals, so
   text like `// struct Widget` in a comment was treated as a genuine
   declaration. Fixed by making the scanner comment/string-aware, matching
   `struct`/`enum` only as the reserved keyword itself.

2. (comment 4114245201) Once the checker accepted `pkg_mod::Widget::make()`,
   resolving it to the right *callee* still depended on `self-hosted/ir/
   lower.sio`'s `callee_path_module_stripped_name`. Its cand-search loop
   originally refused to even consider a qualifier candidate whose last
   included segment looked like a type (uppercase, or a known struct/enum)
   -- a heuristic that ran BEFORE the registry check
   (`lower_use_module_known`) ever got a chance to confirm the longer,
   genuinely-registered candidate (`"pkg_mod::Widget"`, not bare
   `"pkg_mod"`). Removing that premature veto surfaced a second, related
   bug: once the longer candidate (`k == 2`) legitimately won, the
   function's "one segment remains after the qualifier" branch checked
   `lower_impl_method_known` against `segments[0]` unconditionally (correct
   only when the qualifier is exactly one segment, `k == 1`) instead of the
   segment that actually sits next to the method name, `segments[k - 1]`
   (`"Widget"` here). Checking the wrong segment always missed the real
   impl method and fell through to a bare, unmangled `"make"`, which the
   native backend refuses as a body-less stub (`NATIVE_REFUSAL kind=
   empty_stub_ud2 ... reason=missing_lowered_body`, surfacing as a SIGILL
   at runtime). Both are fixed in place; `segments[k - 1]` generalizes the
   original `k == 1` special case for any qualifier length.

The `af_mod::AfNum::quad` fixture (`tests/run-pass/
madaros_module_qualified_call.sio`) -- where `af_mod` alone is the real
module and `AfNum` comes from an unrelated import, the case the original
uppercase heuristic existed to protect -- was re-verified to still pass
after all of the above.
