Fixed (ADR-015 row 9, both engines): the multi-character tokens `<-`, `+-` and `&!` were measured over the corpus (9,019 `.sio` files, strings and `//` comments removed): `<-` 1 occurrence in 1 file (inside the causal-graph `<->`), `+-` 2 occurrences in 2 files (`1.0 +- 0.1` in two compile-fail tests), `&!` 30,810 occurrences in 1,636 files (the mutable borrow `&!T`).

- `<-` has no use of its own, so it is no longer a token: `x<-1` is `x < -1` on both engines (Madaros used to fail to parse it). The `<-` of the causal-graph `<->` is kept.
- `+-` and `&!` stay, but the ambiguous spellings are refused (E283) with a hint to add a space: `a+-2` (plus a negative, or the uncertainty operator) and `a&!b` (and-not, or a mutable borrow). `1.0 +- 0.1` and `&!T` after a space, `(`, `,` or `:` are unchanged.

Probes `op_lt_neg`, `op_lt_neg_sp`, `op_plus_neg`, `op_plus_neg_sp`, `op_and_not`, `op_bool_and_not` now agree. Pins: `op_spacing_{madaros,lean_single}` (run-pass), `op_plus_neg_unspaced_*` and `op_and_not_unspaced_*` (compile-fail).
