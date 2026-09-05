# Known gaps in dimensional typing — reproductions

Each file states its EXPECTED outcome in its first line. One of them is still
expected to **pass**; that is the defect it reproduces. They are kept out of
`tests/compile-fail/` for that reason. Measured 2026-09-02 on both engines;
dispatch in `docs/audit/DIMENSIONAL_TYPING_GAP_2026-09-02.md`.

`derived_unit_dropped_by_inference.sio` was the second of those two and is now
REFUSED (2026-09-05) — the quotient keeps its dimension. Its row below records
the closure. It could move to `tests/compile-fail/`; it has not, because that
needs a `//@` annotation and shifts the corpus the LoRA export and the parity
gate both walk, which is a separate change from recording that the gap closed.

| file | engine | expected | what it shows |
|---|---|---|---|
| `direct_unit_mismatch_is_caught.sio` | both | fail | control: `mol + K` is rejected |
| `derived_unit_dropped_by_inference.sio` | both | **fail** (was pass) | CLOSED 2026-09-05: `(mol/cm3) + K` now `error: unit dimension mismatch` — quotient keeps its dimension |
| `unit_lost_at_call_boundary.sio` | lean_single | **pass** | a `K` value enters an `f64` parameter unchecked |
| `derived_unit_annotation_unparsed.sio` | both | parse fail | `mol/cm3`, `cal/mol` are not in the grammar |
