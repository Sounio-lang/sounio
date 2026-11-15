Madaros: second-order GUM (`hessian_of`, `second_order_mean`) is no longer
limited to 8 channels, and four silent or NaN answers are gone.

Wide second order. The second-order tables now cover all 32 channels (528
Hessian slots, 128 bound locals; the declared correlation matrix is 32 x 32
instead of 8 x 8, so `correlate` between channels above 7 is no longer
dropped) and `hessian_of(r, j, k)` takes `j`, `k` below 32. A call with nine
or twelve tangents returns its exact Hessian (it read NaN), a callee takes up
to 32 tangents (it was 24), and 32 measured quantities in one expression
carry exactly. Pins: `tests/run-pass/gum_so_hyperdual_clones.sio`,
`tests/run-pass/gum_so_32_channels.sio`; oracles
`tests/oracles/gum_so_hyperdual_clones.cpp`, `tests/oracles/gum_so_32_channels.cpp`
(C++23, 1e-10 relative).

A measurement made inside a second-order callee is exact. The clone writes its
internal curvature term `sum H_mm sigma_m^2` in one extra out slot and the
caller adds it as a curvature-only channel, so `second_order_mean` of the call
(and of a function of it, and of a callee that calls a drawing callee) is
exact, including where the callee is stationary in its measurement.

`hessian_of(f(x), j, k)` and `second_order_mean(f(x))` of an IO callee whose
call appears only inside the query: the clone call is emitted inside the
query lowering, and `second_order_mean` lowers its value before the walk, so
the argument is evaluated exactly once (the callee's side effect happens once)
and the value is exact.

A `var` declared before the first `measure` (#3547) now gets its private
sensitivity, Hessian and variance slots once a channel exists. Before, a
loop-carried update through it read variance 19.71 where the value is 59.13,
Hessian 0 where it is 18 and 12.

Refusals instead of wrong numbers. A query that reads a measured quantity
beyond the 32-slot channel table, or a `var` declared before the first
`measure` whose loop-carried update could not be given slots (the first
`measure` sat in a nested block), is refused at compile time ("refused
instead of reading 0", KL-11); before, the variance and mean were silently
short (a sum over 34 quantities read 478.785 where the value is 571.285).
Pins: `tests/compile-fail/gum_refuse_uncovered_hessian.sio`,
`tests/compile-fail/gum_refuse_late_var_nested_measure.sio`; the existing
`gum_refuse_uncovered_call_zero.sio` stays refused. A first-order clone
reached with second order on, which cannot happen for a registered callee,
now refuses instead of reading NaN.
