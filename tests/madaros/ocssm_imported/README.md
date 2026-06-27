# OCSSM Imported Witnesses

This directory preserves the original import-bearing OCSSM run-pass programs
that were moved out of `tests/run-pass/` when the native full-suite lane gained
self-contained mirrors.

The native full suite currently executes a lean_single-derived stage2 artifact
through `scripts/ci/souc-native-wrapper.sh`; that path does not resolve these
single-entry stdlib imports and reports E137. Keep these files here as the
Madaros/import-aware target surface, then move them back under a wired
import-aware gate once `bin/madaros check/run` can resolve their stdlib module
dependencies in CI.
