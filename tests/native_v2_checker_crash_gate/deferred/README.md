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
- `06_tuple_return.sio`: tuple return type lowering/compat remains rejected.
- `09_if_let_nested.sio`: false E006; `if let` condition is checked as an
  `Option` expression rather than pattern control flow.
- `10_while_let_option.sio`: still SIGSEGVs in the single-module check path.
- `13_method_value_receiver.sio`, `14_method_ref_receiver.sio`: no longer
  SIGSEGV after the impl-item in-place fix, but method lookup still reports
  false E011.
- `15_generic_fn_id.sio`, `16_generic_multi_param.sio`,
  `17_generic_struct_pair.sio`: generic function/struct instantiation still
  produces false E008/E009/E004.
- `18_option_box_match.sio`: false E005 for Box dereference in the checker.
