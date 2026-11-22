A new gate asserts that a compile-fail fixture exercising an epistemic builtin
is *refused* rather than *crashed*. `scripts/lib/classify_compile.sh` records
that only 2 of the 18 gates demanding a failed compile check for a signal, and
the epistemic builtins sat in that blind spot: 39 of the 42 such fixtures print
the right diagnostic and then segfault, which every gate reading a non-zero exit
as a refusal scored as passing. The existing parity gate does use the classifier
but runs `lean_single`, which refuses these fixtures with E200 because it does
not implement the builtins. The new gate runs the corpus through Madaros, pins
the current crash set in `tests/selfhost/madaros_epistemic_compile_fail_crashers.txt`,
and fails in both directions: an unpinned crash is a regression, and a pinned
fixture that stops crashing must be removed, so the set may only shrink.
