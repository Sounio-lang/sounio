lean_single: a type error in an imported module now fails the build (exit 1,
`typecheck: failed`, no output file), as one in the main file already did.
`tc_mark_failed` and `note_limit_error` used to return early for functions from
imported modules, so the seed printed `error: ...`, exited 0 and wrote an ELF
holding whatever code the refused expression left: the witness in
`tests/compiler/lean_single_import_error_fatal/refused` printed
1547925990675276 instead of 1234. New gate
`lean_single_import_error_fatal_gate.sh` (with a clean-import positive control).

The seed is refreshed to the fixed point of the new source (g1 == g2). On the
3474 non-compile-fail full-suite files, 68 printed an error with exit 0 under the
previous seed; all 68 are now refused, and the output of the other files is
unchanged (byte-identical ELFs on a 241-file sample). The 46 of them that the
lean_single full suite runs are listed, with the stdlib site that fails each,
in `tests/known_failures/hardened_diagnostics_full_suite.txt`.
