# Deferred Checker Crash/False-Positive Cases

These are well-typed single-module probes that are intentionally not part of
the active crash gate yet.

Current status after the binary-tail and impl-item fixes:

- `02_closure_capture.sio`: **DEFERRED.** False E004/E009; untyped closure
  params default to `()` (TypeUnit) in `parser/exprs.sio` `parse_closure_param`.
  The only cheap "fix" — binding an untyped param to `TyUnknown` — is a C0
  soundness HOLE (TyUnknown is compatible with everything, so the body would
  type-check against any use) AND an R1 break (the backend has no concrete type
  to lower the body). Needs real closure-param-type inference; left rejected.
- `04_hof_fn_param.sio`: **FIXED** — moved to `valid/`. The `*mut` type-lowering
  (`checker_lower_type_expr_mut`) had no `TypeFn` arm, so a `fn(i64) -> i64`
  parameter type fell to the `_` error arm; and `types_compatible` only matched
  two `TyFn` by `fn_sig_id` equality (never true for a fn-type parameter vs a
  named-function argument). Fixed by `checker_lower_fn_type_mut` (TypeFn arm) +
  structural `checker_fn_arg_compatible_inplace`. Checker accepts AND
  `--native-v2-compile` lowers to the correct exit (42; distinct 20 for
  `apply(double, 10)`). Ill-typed twin (wrong arity) committed to
  `invalid/04_hof_fn_param_wrong_arity.sio`; wrong-param/wrong-return/non-fn
  twins also reject with E009.
- `05_hof_closure_param.sio`: **DEFERRED on R1.** The 04 checker fix would also
  make a closure-as-fn-arg CHECK OK, but the native-v2 backend cannot yet lower
  a closure passed as a fn-type argument to a correct exit (it silently
  miscompiles to 0 — empirically confirmed for both a closure literal
  `apply(|x| x*2, n)` and a closure bound to a local and passed by ident
  `let g = |x| x*2; apply(g, n)`). To avoid landing a checker change that
  unblocks an un-lowerable program, `checker_fn_arg_compatible_inplace` GUARDS
  the structural accept by the ARGUMENT's resolved FnSig: only a TOP-LEVEL
  function (non-empty sig name, lowerable via the IrLoadFnRef/fnptr path) is
  accepted; an anonymous closure sig (empty name) is rejected regardless of
  whether it arrives as a literal or via a local. Both forms stay honest
  false-rejects (R1 witnesses in `invalid/04_hof_closure_value_arg.sio`).
  Re-enable once the backend lowers closure args.
- `06_tuple_return.sio`: **FIXED** — moved to `valid/`. The `*mut` type-lowerer
  (`checker_lower_type_expr_mut`, check.sio) had no `TypeTuple` arm, so a
  `(i64, i64)` return/param type fell to the `_` error arm and the checker
  false-rejected every tuple-typed program. Added `checker_lower_tuple_type_mut`
  + `checker_lower_type_expr_list_mut` (mirroring the dead by-value
  `lower_tuple_type`/`lower_type_expr_list`) and a `TypeTuple` arm. Both
  `--check` OK and `--native-v2-compile` ELF runs to the correct exit (42 =
  10+32 for the return form; 42 = 40+2 for a tuple-param form). Ill-typed twins
  stay rejected for the right reason — `tuple_types_compatible` (compat.sio) is
  arity- and element-wise structural: wrong return arity / wrong return element
  type reject with E008, wrong-arity tuple argument rejects with E009. See
  `invalid/06_tuple_return_wrong_arity.sio`,
  `invalid/06_tuple_return_wrong_elemtype.sio`,
  `invalid/06_tuple_param_wrong_arity.sio`.
- `09_if_let_nested.sio`: false E006; `if let` condition is checked as an
  `Option` expression rather than pattern control flow.
- `10_while_let_option.sio`: still SIGSEGVs in the single-module check path.
- `13_method_value_receiver.sio`, `14_method_ref_receiver.sio`: FIXED — moved
  to `valid/`. The live `*mut` collect spine had a no-op `ItemImpl` branch in
  `checker_collect_item_inplace` (check.sio), so no method signatures were
  registered and `find_method_semantic` returned nothing -> false E011. Wired a
  `*mut` impl collector (`checker_collect_impl_def_inplace` ported from the dead
  by-value `collect_impl_def`/`collect_impl_method`). Both `--check` OK and
  `--native-v2-compile` ELF runs to exit 42. Ill-typed twins (nonexistent
  method, wrong arg type, wrong receiver type) stay rejected — see
  `invalid/13_method_*.sio`, `invalid/14_method_ref_nonexistent.sio`.
- `15_generic_fn_id.sio`, `16_generic_multi_param.sio`,
  `17_generic_struct_pair.sio`: DEFERRED — generic function/struct
  instantiation still produces false E008/E009/E004. Needs FnSig type-param
  storage + per-call turbofish binding + substitution into params/return
  (15/16: check.sio:~4195 `checker_check_call_args_inner_inplace` compares
  arg `i64` vs unresolved `TyNamed` `T`) and monomorphized-struct field
  registration at lowering (17: `checker_lower_named_type_mut` ignores
  `te.type_args`). A wildcard shortcut would over-accept `id::<i64>(true)` =
  C0 soundness hole, so these stay deferred until done properly.
- `18_option_box_match.sio`: false E005 for Box dereference in the checker.
