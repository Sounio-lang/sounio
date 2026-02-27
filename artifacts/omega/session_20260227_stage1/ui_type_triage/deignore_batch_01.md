# UI Type De-ignore Batch 01

Status: applied.

Source evidence:
- `ui_type_triage.json`
- `ui_type_triage.md`

## Candidates

1. `tests/ui/type/mismatch_arg.sio` (applied)
   - triage: `ready`
   - expectation match: yes
   - semantic risk: low

2. `tests/ui/type/generic_constraint.sio` (applied)
   - triage: `ready`
   - expectation match: yes
   - semantic risk: medium (currently fails as `Undefined type`, not trait-bound enforcement)

## Applied command

```bash
sed -i '/^\\/\\/@ ignore$/d' tests/ui/type/mismatch_arg.sio
```

## Post-apply status

- `active`: 11 (pass=11, fail=0)
- `ignored`: 31 (ready=0, needs-fix=2, still-blocked=29)

```bash
# no remaining ready candidates in this batch
```
