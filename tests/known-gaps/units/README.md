# Known gaps in dimensional typing — reproductions

Each file states its EXPECTED outcome in its first line. Measured 2026-09-02 on both engines;
dispatch in `docs/audit/DIMENSIONAL_TYPING_GAP_2026-09-02.md`.

`derived_unit_dropped_by_inference.sio` LEFT this directory on 2026-09-05: the
gap it reproduced closed, so a file that must now be REFUSED moved to where the
refusals live, as `tests/compile-fail/unit_quotient_keeps_dimension.sio`. A
directory of known gaps should not hold a gap that is shut.

| file | engine | expected | what it shows |
|---|---|---|---|
| `direct_unit_mismatch_is_caught.sio` | both | fail | control: `mol + K` is rejected |
| ~~`derived_unit_dropped_by_inference.sio`~~ | both | **fail** (was pass) | CLOSED 2026-09-05 — moved to `tests/compile-fail/unit_quotient_keeps_dimension.sio` |
| ~~`unit_lost_at_call_boundary.sio`~~ | both | fail | CLOSED — moved to `tests/compile-fail/unit_lost_at_call_boundary.sio` |
| ~~`derived_unit_annotation_unparsed.sio`~~ | both | parses | CLOSED (#2388) — `^n` and parentheses parse; covered by `tests/run-pass/unit_derived_power_and_parens_issue_2388.sio`, and `mL`/`L` by `tests/run-pass/unit_volume_mL_L_issue_2388.sio`; the file is gone from this directory |
