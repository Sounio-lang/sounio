# Deferred Checker Crash/False-Positive Cases

These are well-typed single-module probes that are intentionally not part of
the active crash gate yet.

Current status after the binary-tail and impl-item fixes:

- `02_closure_capture.sio`: false E004/E009; untyped closure params default to
  `()` in the checker capture path.
- `04_hof_fn_param.sio`, `05_hof_closure_param.sio`: higher-order function
  parameter typing is still rejected by the single-module checker.
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
