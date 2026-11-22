Madaros parser: the condition of `if`, `while`, `match` and `for` keeps
refusing a bare struct literal all the way to its end. The restriction flag
(`Parser.no_struct_lit`) used to be reset to `false` by any `if`, `match`,
`while` or `for` nested inside an enclosing condition, so in
`if (match jq { 0 => 1.0, _ => 2.0 }) * cf > 0.01 * vf {` the `vf {` was read as
a struct literal and the module was refused with only "module failed to
parse". The flag is now saved and restored at every condition site.

Struct literals stay legal inside the delimited parts of a condition, as in
Rust: parentheses, call and method arguments, index brackets, array literals,
blocks and match arms (`if nx(V { x: 3.0 }) > 1.0 {`, `if (V { x: 3.0 }).x > 1.0 {`
used to fail to parse).

A bare `Name { field: ...` written directly in a condition now gets one located
message, `struct literal not allowed here at line L:C`, with a hint to wrap the
literal in parentheses, instead of a cascade that ended at the `:`.

Tests: `tests/run-pass/parser_condition_struct_literal_restriction.sio`,
`tests/compile-fail/parser_struct_literal_in_condition_diagnostic.sio`.
