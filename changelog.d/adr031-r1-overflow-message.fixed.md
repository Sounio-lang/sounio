An integer overflow panic now says what overflowed (ADR-031 R1). The program
still exits 1, and stderr carries one line such as
`madaros: integer overflow (i64 +)` or `madaros: integer overflow (u32 *)`.
Before, the trap exited 1 without a word.

The 64-bit guards (`jo`/`jc`) jump to per-function, per-kind stubs (at most
seven per function), so the cost is a few stubs per function rather than one
per site. The i32/u32 range checks call the new `__int_overflow` builtin with
a code naming the type and the operation, in place of a bare `panic`.
The source location (file:line) is not in the message yet: the IR does not
carry spans, and Madaros compile-time diagnostics do not print one either.
Pinned by the message assertions in `scripts/ci/madaros_int_overflow_gate.sh`.
