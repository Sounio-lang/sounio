# Known gaps in dimensional typing — reproductions

Each file states its EXPECTED outcome in its first line. Two of them are
expected to **pass**; that is the defect they reproduce. They are kept out of
`tests/compile-fail/` for that reason. Measured 2026-09-02 on both engines;
dispatch in `docs/audit/DIMENSIONAL_TYPING_GAP_2026-09-02.md`.

| file | engine | expected | what it shows |
|---|---|---|---|
| `direct_unit_mismatch_is_caught.sio` | both | fail | control: `mol + K` is rejected |
| `derived_unit_dropped_by_inference.sio` | lean_single | **pass** | `(mol/cm3) + K` accepted: quotient loses its dimension |
| `unit_lost_at_call_boundary.sio` | lean_single | **pass** | a `K` value enters an `f64` parameter unchecked |
| `derived_unit_annotation_unparsed.sio` | both | parse fail | `mol/cm3`, `cal/mol` are not in the grammar |
