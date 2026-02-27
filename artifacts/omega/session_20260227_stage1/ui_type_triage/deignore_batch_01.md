# UI Type De-ignore Batch 01 (Prepared)

Status: prepared, not applied.

Source evidence:
- `ui_type_triage.json`
- `ui_type_triage.md`

## Candidates

1. `tests/ui/type/mismatch_arg.sio`
   - triage: `ready`
   - expectation match: yes
   - semantic risk: low

2. `tests/ui/type/generic_constraint.sio`
   - triage: `ready`
   - expectation match: yes
   - semantic risk: medium (currently fails as `Undefined type`, not trait-bound enforcement)

## Safe apply command (only low-risk first)

```bash
sed -i '/^\\/\\/@ ignore$/d' tests/ui/type/mismatch_arg.sio
```

## Optional second wave (requires reviewer confirmation)

```bash
sed -i '/^\\/\\/@ ignore$/d' tests/ui/type/generic_constraint.sio
```
