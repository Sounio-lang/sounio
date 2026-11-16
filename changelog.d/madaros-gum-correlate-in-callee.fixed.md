Madaros: a `correlate` between two quantities measured inside a callee is no
longer ignored in a query on a call to it.

A callee containing `correlate` was never given a tangent clone, so with a
certain argument its own measurement uncertainty was dropped without a
diagnostic (variance 0 where the value is 7.69), and with an uncertain one the
call was refused. It is now cloned: the clone adds the correlated cross terms
`2 J_i J_j rho_ij sigma_i sigma_j` to its internal variance and
`2 H_ij rho_ij sigma_i sigma_j` to its curvature term, so `variance_of` and
`second_order_mean` of the call (and of a function of it, and of a callee that
calls it) match the oracle, first and second order, for either sign of rho
(`corr_*` in `tests/run-pass/gum_so_hyperdual_clones.sio`, oracle
`tests/oracles/gum_so_hyperdual_clones.cpp`, C++23, 1e-10 relative). A
`correlate` whose operand is derived from a parameter of the callee needs the
caller's channel and is refused at compile time ("refused instead of reading
0", KL-11), never answered with the independent value:
`tests/compile-fail/gum_refuse_correlate_param_in_callee.sio`.
