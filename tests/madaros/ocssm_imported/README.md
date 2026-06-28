# OCSSM Imported Witnesses

This directory preserves the original import-bearing OCSSM run-pass programs
that were moved out of `tests/run-pass/` when the native full-suite lane gained
self-contained mirrors.

The native full suite currently executes a lean_single-derived stage2 artifact
through `scripts/ci/souc-native-wrapper.sh`; that path does not resolve these
single-entry stdlib imports and reports E137. Keep these files here as the
Madaros/import-aware target surface.

`manifest.tsv` gates the currently green `bin/madaros check` surface for all
five import-bearing witnesses, including the SWDA and M5 loaders that use
`read_file`, `file_size`, and `read_i64`. This is a type-check/import gate
only: `bin/madaros run` still reaches the import-aware native lowering path and
segfaults at `lower_array`.
