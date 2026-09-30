# Duplicate `main` on specialized collapse

This fixture forces Madaros's specialized multi-module collapse path by
instantiating an imported generic. The dependency deliberately defines its own
`main`, as many self-testing library modules do. The executable must run only
the importing program's entry point.

The gate compares stdout byte-for-byte with `expected.txt`. This proves both
that `USER_MAIN` is present and that `DEP_MAIN` is absent.
