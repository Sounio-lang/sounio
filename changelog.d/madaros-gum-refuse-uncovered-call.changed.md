Madaros: an uncertainty query whose answer depends on a call the compiler
cannot differentiate is now refused at compile time, instead of reading
0.0. The queries covered are `variance_of`, `covariance_of`,
`uncertainty_of`, `sensitivity_of`, `hessian_of` and `second_order_mean`.

`error: variance_of in `main` depends on the call to `apply1`, whose
uncertainty the compiler cannot propagate; refused instead of reading 0
(KL-11)`

How the decision is made:

- The variance walk marks a call when it can form no variance for it
  (no tangent clone, no transfer, no re-lowering) while one of its
  arguments or its receiver can carry uncertainty.
- A local bound from such a walk is tainted, so the mark follows it into
  later queries.
- A struct-valued `let` from an uncovered call is tainted when none of its
  fields received a variance.
- Marks raised inside a speculative re-lowering that is then refused do
  not count. Neither do calls refused by the inline depth or budget caps,
  which keep the runtime NaN poison they already had.

A local bound to a named function (`let f = add2; f(x, y)`) is now seen
through, so it takes the same paths as `add2(x, y)`.

Measured on the 401 epistemic programs: two refusals remain.
`tests/run-pass/gum_fo_struct_fields_channels_2776.sio` is a higher-order
call through a function parameter; it is a lean_single-only test and the
Madaros value there was 0.0 against 0.36. The other is the new
`tests/compile-fail/gum_refuse_uncovered_call_zero.sio`, which pins the
refusal. Main's compiler compiles that file and prints 0.000000.
