# UI Type Triage (Session 2026-02-27)

- souc_bin: `/home/demetrios/work/sounio/artifacts/omega/souc-bin/souc-linux-x86_64`
- total_files: 42
- active: 10 (pass=10, fail=0)
- ignored: 32 (ready=1, needs-fix=2, still-blocked=29)

## Ready De-ignore Candidates
- `tests/ui/type/generic_constraint.sio`

## Needs Expectation Fix
- `tests/ui/type/refinement_violation.sio`
- `tests/ui/type/unit_mismatch.sio`

## Still Blocked
- `tests/ui/type/array_elem_mismatch.sio`
- `tests/ui/type/array_index_type.sio`
- `tests/ui/type/assign_to_immut.sio`
- `tests/ui/type/bitwise_float.sio`
- `tests/ui/type/break_outside_loop.sio`
- `tests/ui/type/closure_arg_type.sio`
- `tests/ui/type/comparison_type_mismatch.sio`
- `tests/ui/type/condition_not_bool.sio`
- `tests/ui/type/continue_outside_loop.sio`
- `tests/ui/type/division_by_zero.sio`
- `tests/ui/type/extra_args.sio`
- `tests/ui/type/field_not_found.sio`
- `tests/ui/type/if_branch_mismatch.sio`
- `tests/ui/type/invalid_binary_op.sio`
- `tests/ui/type/invalid_unary_op.sio`
- `tests/ui/type/logical_not_bool.sio`
- `tests/ui/type/match_arm_mismatch.sio`
- `tests/ui/type/method_not_found.sio`
- `tests/ui/type/modulo_float.sio`
- `tests/ui/type/not_callable.sio`
- `tests/ui/type/not_indexable.sio`
- `tests/ui/type/range_type_mismatch.sio`
- `tests/ui/type/shift_float.sio`
- `tests/ui/type/struct_extra_field.sio`
- `tests/ui/type/struct_field_type.sio`
- `tests/ui/type/struct_missing_field.sio`
- `tests/ui/type/tuple_index_oob.sio`
- `tests/ui/type/while_cond_not_bool.sio`
- `tests/ui/type/wrong_arg_count.sio`

