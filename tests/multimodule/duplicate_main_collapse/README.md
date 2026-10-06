# Duplicate `main` on specialized collapse

This fixture forces Madaros's specialized multi-module collapse path by
instantiating an imported generic. The dependency deliberately defines its own
`main`, as many self-testing library modules do. The executable must run only
the importing program's entry point.

The gate first requires the compiler receipt
`specialized_collapse lower_count=1`, rejects the explicit
`specialized lower failed` fallback receipt, and then compares stdout
byte-for-byte with `expected.txt`. This proves the fixed lowering branch
completed, that `USER_MAIN` is present, and that `DEP_MAIN` is absent.

`typecheck_error/` holds a semantic type error only inside the dependency's
`main`. Compilation must be refused without writing an ELF, proving dependency
entry functions remain in the typecheck merge even though lowering filters them.
