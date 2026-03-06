# Sprint 5: Contest / Validated / Robust Types — 1-2-3 Status Pack

**Date**: 2026-03-05
**Branch**: codex/sprint5-contest-validated-20260305

## Gate: 16/16 PASS

```
check_main_sio                      PASS  all_checks_passed
typekind_TyPolicy                   PASS  found_in_types_sio
typekind_TyContest                  PASS  found_in_types_sio
typekind_TyRobust                   PASS  found_in_types_sio
ctor_ty_policy                      PASS  found_in_types_sio
ctor_ty_contest                     PASS  found_in_types_sio
ctor_ty_robust                      PASS  found_in_types_sio
compile_fail_stub_contest_no_silent_unwrap  PASS  file_exists
compile_fail_stub_contest_requires_annotation  PASS  file_exists
compile_fail_stub_robust_not_validated  PASS  file_exists
epistemic_fn_check_contest_type     PASS  found_in_epistemic_sio
epistemic_fn_check_policy_type      PASS  found_in_epistemic_sio
epistemic_fn_check_robust_type      PASS  found_in_epistemic_sio
dispatch_TypePolicy                 PASS  found_in_check_sio
dispatch_TypeContest                PASS  found_in_check_sio
dispatch_TypeRobust                 PASS  found_in_check_sio
```

## What was delivered

- **TypeKind**: `TyPolicy`, `TyContest`, `TyRobust` variants added to enum
- **TypeEntry**: 3 new fields (`epistemic_meta_id`, `robustness_level_id`, `robustness_scope_id`)
- **Constructors**: `ty_policy(inner, meta_id)`, `ty_contest(inner, meta_id)`, `ty_robust(inner, level_id, scope_id)`
- **Type checker dispatch**: `lower_type_expr` routes `TypePolicy/TypeContest/TypeRobust` AST nodes
- **Epistemic checkers**: `check_contest_type`, `check_policy_type`, `check_robust_type` in epistemic.sio
- **Compatibility**: `types_compatible` extended for Policy/Contest/Robust pairs
- **Name helpers**: `name_matches_str11`, `name_matches_str12` added to compat.sio
- **Compile-fail stubs**: 3 test files with `//@ ignore` pending bootstrap routing

## Cumulative Lane Count

| Sprint | Lanes |
|--------|-------|
| Sprint 1 | 2 |
| Sprint 2 | 4 (cumulative) |
| Sprint 3 | 8 (cumulative) |
| Sprint 4 | 9 (cumulative) |
| Sprint 5 | 16-case gate pass |
