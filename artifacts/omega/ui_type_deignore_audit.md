# UI Type De-ignore Audit

- generated_at_utc: 2026-03-01T14:46:19Z
- souc_bin: /home/demetrios/work/sounio/souc
- ignored_files: 40
- safe_deignore_candidates: 0
- ready_count: 0
- needs_fix_count: 2
- still_blocked_count: 38

## Bucket Summary

- ready: 0
- needs-fix: 2
- still-blocked: 38

## Ready

## Needs Fix
- tests/ui/type/epistemic_call_boundary_invalid_provided_epsilon.sio (exit=1, pattern_hit=false, first_line=Error: P0003)
- tests/ui/type/epistemic_call_boundary_unknown_provided_epsilon.sio (exit=1, pattern_hit=false, first_line=Error: P0003)

## Still Blocked
- tests/ui/type/array_elem_mismatch.sio (BLOCKED: pinned binary does not enforce array element types; self-hosted checker has E020)
- tests/ui/type/array_index_type.sio (BLOCKED: pinned binary does not enforce integer array indices; self-hosted checker has E014)
- tests/ui/type/assign_to_immut.sio (BLOCKED: pinned binary does not enforce immutability; self-hosted checker has E003)
- tests/ui/type/bitwise_float.sio (BLOCKED: pinned binary does not enforce integer-only bitwise ops; self-hosted checker has E048)
- tests/ui/type/break_outside_loop.sio (BLOCKED: pinned binary does not enforce break in loops; self-hosted checker has E044)
- tests/ui/type/closure_arg_type.sio (BLOCKED: pinned binary does not enforce closure arg types; self-hosted checker validates via fn sig)
- tests/ui/type/comparison_type_mismatch.sio (BLOCKED: pinned binary does not enforce comparison type matching; self-hosted checker has E004)
- tests/ui/type/condition_not_bool.sio (BLOCKED: pinned binary does not enforce bool conditions; self-hosted checker has E006)
- tests/ui/type/continue_outside_loop.sio (BLOCKED: pinned binary does not enforce continue in loops; self-hosted checker has E045)
- tests/ui/type/division_by_zero_assign_const.sio (BLOCKED: pinned binary does not propagate const assignment denominator; self-hosted checker has E056)
- tests/ui/type/division_by_zero_const_expr.sio (BLOCKED: pinned binary does not fold const denominator; self-hosted checker has E056)
- tests/ui/type/division_by_zero_const_ident.sio (BLOCKED: pinned binary does not track const id denominator; self-hosted checker has E056)
- tests/ui/type/division_by_zero.sio (BLOCKED: pinned binary does not detect literal div-by-zero; self-hosted checker has E056)
- tests/ui/type/extra_args.sio (BLOCKED: pinned binary does not enforce call arity; self-hosted checker has E010)
- tests/ui/type/field_not_found.sio (BLOCKED: pinned binary does not enforce field existence; self-hosted checker has E012)
- tests/ui/type/if_branch_mismatch.sio (BLOCKED: pinned binary does not enforce branch type compatibility; self-hosted checker has E007)
- tests/ui/type/invalid_binary_op.sio (BLOCKED: pinned binary does not enforce binary op types; self-hosted checker has E004)
- tests/ui/type/invalid_unary_op.sio (BLOCKED: pinned binary does not enforce unary op types; self-hosted checker has E005)
- tests/ui/type/logical_not_bool.sio (BLOCKED: pinned binary does not enforce logical not on bool; self-hosted checker has E005)
- tests/ui/type/match_arm_mismatch.sio (BLOCKED: pinned binary does not enforce match arm types; self-hosted checker has E018)
- tests/ui/type/method_not_found.sio (BLOCKED: pinned binary does not enforce method existence; self-hosted checker has E011)
- tests/ui/type/modulo_by_zero_assign_const.sio (BLOCKED: pinned binary does not propagate const assignment divisor; self-hosted checker has E057)
- tests/ui/type/modulo_by_zero_const_expr.sio (BLOCKED: pinned binary does not fold const divisor; self-hosted checker has E057)
- tests/ui/type/modulo_by_zero_const_ident.sio (BLOCKED: pinned binary does not track const id divisor; self-hosted checker has E057)
- tests/ui/type/modulo_by_zero.sio (BLOCKED: pinned binary does not detect literal mod-by-zero; self-hosted checker has E057)
- tests/ui/type/modulo_float.sio (BLOCKED: pinned binary does not enforce integer-only modulo; self-hosted checker has E050)
- tests/ui/type/not_callable.sio (BLOCKED: pinned binary does not enforce callable check; self-hosted checker has E017)
- tests/ui/type/not_indexable.sio (BLOCKED: pinned binary does not enforce array indexing; self-hosted checker has E013)
- tests/ui/type/range_type_mismatch.sio (BLOCKED: pinned binary does not enforce range types; self-hosted checker has E055)
- tests/ui/type/refinement_violation.sio (BLOCKED: pinned binary cannot parse refinement type declarations; self-hosted checker has E042)
- tests/ui/type/shift_float.sio (BLOCKED: pinned binary does not enforce integer-only shift ops; self-hosted checker has E049)
- tests/ui/type/struct_extra_field.sio (BLOCKED: pinned binary does not enforce struct field count; self-hosted checker has E046)
- tests/ui/type/struct_field_type.sio (BLOCKED: pinned binary does not enforce struct field types; self-hosted checker has E016)
- tests/ui/type/struct_missing_field.sio (BLOCKED: pinned binary does not enforce struct field count; self-hosted checker has E046)
- tests/ui/type/tuple_index_oob.sio (BLOCKED: pinned binary does not enforce tuple index bounds; self-hosted checker has E047)
- tests/ui/type/unit_mismatch.sio (BLOCKED: pinned binary has no unit registry; self-hosted checker has E041)
- tests/ui/type/while_cond_not_bool.sio (BLOCKED: pinned binary does not enforce bool conditions; self-hosted checker has E006)
- tests/ui/type/wrong_arg_count.sio (BLOCKED: pinned binary does not enforce call arity; self-hosted checker has E010)

## Safe Candidates

## Blocked Samples
- tests/ui/type/array_elem_mismatch.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/array_elem_mismatch.sio)
- tests/ui/type/array_index_type.sio (exit=0, pattern_hit=false, first_line=All checks passed: tests/ui/type/array_index_type.sio)
- tests/ui/type/assign_to_immut.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/assign_to_immut.sio)
- tests/ui/type/bitwise_float.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/bitwise_float.sio)
- tests/ui/type/break_outside_loop.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/break_outside_loop.sio)
- tests/ui/type/closure_arg_type.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/closure_arg_type.sio)
- tests/ui/type/comparison_type_mismatch.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/comparison_type_mismatch.sio)
- tests/ui/type/condition_not_bool.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/condition_not_bool.sio)
- tests/ui/type/continue_outside_loop.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/continue_outside_loop.sio)
- tests/ui/type/division_by_zero_assign_const.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/division_by_zero_assign_const.sio)
- tests/ui/type/division_by_zero_const_expr.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/division_by_zero_const_expr.sio)
- tests/ui/type/division_by_zero_const_ident.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/division_by_zero_const_ident.sio)
- tests/ui/type/division_by_zero.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/division_by_zero.sio)
- tests/ui/type/epistemic_call_boundary_invalid_provided_epsilon.sio (exit=1, pattern_hit=false, first_line=Error: P0003)
- tests/ui/type/epistemic_call_boundary_unknown_provided_epsilon.sio (exit=1, pattern_hit=false, first_line=Error: P0003)
- tests/ui/type/extra_args.sio (exit=0, pattern_hit=false, first_line=All checks passed: tests/ui/type/extra_args.sio)
- tests/ui/type/field_not_found.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/field_not_found.sio)
- tests/ui/type/if_branch_mismatch.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/if_branch_mismatch.sio)
- tests/ui/type/invalid_binary_op.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/invalid_binary_op.sio)
- tests/ui/type/invalid_unary_op.sio (exit=0, pattern_hit=true, first_line=All checks passed: tests/ui/type/invalid_unary_op.sio)
