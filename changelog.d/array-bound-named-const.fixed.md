A named `const` is now a valid array-TYPE bound. Before this change an
identifier in the bound was read as its zero `int_val`, so
`const N: i64 = 4096; struct S { arr: [bool; N] }` typed the field as a
length-0 array and every use of it — including the constructor
`S { arr: [false; 4096] }` — reported a spurious mismatch.

The bound now resolves through the const-int evaluator
(`checker_eval_const_int_opt_expr_mut`), so an integer literal, a named const,
unary negation and integer binops all work. A top-level
`const NAME: i64 = <int>` — recorded by the parser in the `GLOBAL_VAR_INIT_*`
side table, not in the checker's const-int table — is now resolved by the
const-int lookup, so its value reaches both array-type paths
(`checker_lower_array_type_mut` and `lower_array_type`) regardless of the order
items are collected or lowered in. A non-constant bound keeps the previous
`int_val` read; a mutable `var` global is not treated as a constant.

Scope: this fixes the array-TYPE position. A repeat-literal count `[v; NAME]`
still requires an integer literal: the IR lowering
(`lower_array_literal_fixed_len_ref`) fills the array from the count
expression's literal `int_val`, so a const count there would type-check but fill
zero elements. Making that path constant-aware is tracked separately.

Regression: `tests/run-pass/array_bound_named_const.sio`.
