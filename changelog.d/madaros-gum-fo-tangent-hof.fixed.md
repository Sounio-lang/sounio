Madaros: first- and second-order GUM propagate exactly through higher-order
calls, where a function is passed as a parameter. `variance_of(apply1(dbl, x))`
with `dbl(y) = 2y` at x = 5 +- 0.3 reads 0.36 (it read a silent 0.0, and was
then refused at compile time by the uncovered-call rule).

Specialised clones. A function with function-typed parameters (an `f64`
return, `f64` / `Knowledge` / integer / bool / struct parameters, no receiver)
is cloned once per tuple of function arguments: `apply1__fotan_dbl_s512`, or
`__fotan2_` with second order on. Inside the clone the function parameter is
bound to the known function, so `f(x)` is a direct call to `dbl`, which goes
through `dbl`'s own tangent clone; the chain rule nests. The out-array
protocol, the Jacobian row and the caller's chain rule are those of the
ordinary tangent clones; the function parameter simply carries no tangent.

The function argument must be known at compile time and registered as a
tangent-clone helper: a named function, a `let` alias of one, or a function
parameter of an enclosing specialised clone (a HOF forwarding its parameter to
another HOF, at any depth). A function chosen at run time
(`let g = if c { dbl } else { tri }`) cannot be specialised and the query is
still refused at compile time, as before; its pin is now
`tests/compile-fail/gum_refuse_uncovered_call_zero.sio`. Closures are not
specialised (a closure literal does not typecheck against these function
types, and a capturing closure has no function to bind). A method with a
function-typed parameter keeps the plain call.

Supporting changes found by the witnesses:

- A query whose argument is a call written inside it (`variance_of(apply1(dbl,
  x))`, with no `let` before) has no evaluated value to hang a clone on; when
  the call-site re-lowering refuses the callee, the clone call is now emitted
  inside the query lowering (as `hessian_of` already did), for callees
  without an IO effect. This applies to every registered helper, not only
  higher-order ones.
- The product and quotient rules re-lower their operands' values and bailed
  on any call that is not a pure builtin or an inline helper, so
  `apply1(f, x) * 3.0` read 0 inside a clone. A call to a higher-order helper
  (or to a bound function parameter) whose functions have no IO effect and
  cannot measure is now re-lowerable.
- An earlier `let h = dbl` alias is hidden by a later `let h = ...` of the
  same name instead of staying live.

Pin: `tests/run-pass/gum_fo_tangent_clone_hof.sio` (25 checks: two function
parameters in both orders, a loop calling the parameter, forwarding through
one and two levels, a `let` alias, one HOF at two call sites with different
functions, binary function arguments including a repeated argument,
cancellation `apply1(dbl, x) - 2 x = 0`, `covariance_of`, and the Hessian
through `apply1`, `apply2` and a binary function). Oracle:
`tests/oracles/gum_fo_tangent_clone_hof.cpp`, analytic derivatives of the
composed function, 1e-12 relative first order, 1e-10 second order.
