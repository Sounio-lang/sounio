Integer division and remainder now fail on a zero divisor and on
`MIN / -1` on the Madaros native backend (ADR-031 R2). Release class:
`semantics`.

Before every `idiv`/`div`, the emitter tests the divisor for zero and,
for signed operands, tests for `i64::MIN / -1` (the one quotient that does
not fit). Both jump to the same per-function trap as the SIO-207 overflow
guard, so the program exits through the panic shape (exit 1) instead of
being killed by SIGFPE. `i64::MIN % -1` fails as well, because `idiv`
computes both results. Dynamic `i32` division joins the 32-bit range check,
so `i32::MIN / -1` fails. Constant propagation no longer folds
`MIN / -1`. A divisor that is a constant zero was already refused at
compile time (E056 for `/`, E057 for `%`).

Not covered: the multi-limb wide-integer division helpers, and
`lean_single` (ADR-031 applies to Madaros first).

Pinned by section 5 of `scripts/ci/madaros_int_overflow_gate.sh`.
