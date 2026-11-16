The modular Madaros compiler, which `./bin/souc` runs by default, now
enforces the call-site confidence gate E214 that `lean_single` already had.
Before this change a call into a function declared `with Epistemic(N)` (or
`Confidence(N)`) was accepted whatever the confidence of its arguments:
`safe_dosing(kw.value, kc.value)` with `measure(10.0, uncertainty: 5.0)` and
`measure(5.0, uncertainty: 0.1)` compiled and printed `dose=50.000000`.
It is now refused with `error[E214] ... confidence gate violation (args
confidence 490 < minconf 950)`.

The argument confidence follows the lean_single rule: literals are 1000,
`measure(v, uncertainty: u)` is `1000 - floor(1000 * u / v)`, `.value` and
other field reads keep the confidence of the Knowledge they read, arithmetic
multiplies (floored `a * b / 1000`), and `let` bindings carry the value
forward. A confidence the checker cannot determine is not refused, as in
lean_single. Inside a function that itself declares `Epistemic(N)`, a `let`
whose initialiser is provably below N is refused with E215, which is what
`halo_pgx_gate_refuse.sio` and `tac_compile_gate_refuse.sio` expect.
