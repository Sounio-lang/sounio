# Known gaps in dimensional typing — reproductions

Each file states its EXPECTED outcome in its first line. One of them is still
expected to **pass**; that is the defect it reproduces. It is kept out of
`tests/compile-fail/` for that reason. Measured 2026-09-02 on both engines;
dispatch in `docs/audit/DIMENSIONAL_TYPING_GAP_2026-09-02.md`.

`derived_unit_dropped_by_inference.sio` LEFT this directory on 2026-09-05: the
gap it reproduced closed, so a file that must now be REFUSED moved to where the
refusals live, as `tests/compile-fail/unit_quotient_keeps_dimension.sio`. A
directory of known gaps should not hold a gap that is shut.

| file | engine | expected | what it shows |
|---|---|---|---|
| `direct_unit_mismatch_is_caught.sio` | both | fail | control: `mol + K` is rejected |
| ~~`derived_unit_dropped_by_inference.sio`~~ | both | **fail** (was pass) | CLOSED 2026-09-05 — moved to `tests/compile-fail/unit_quotient_keeps_dimension.sio` |
| `unit_lost_at_call_boundary.sio` | lean_single | **pass** | a `K` value enters an `f64` parameter unchecked |
| `derived_unit_annotation_unparsed.sio` | both | parse fail | `mol/cm3`, `cal/mol` are not in the grammar |
