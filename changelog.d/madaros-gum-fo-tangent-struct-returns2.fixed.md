Madaros: tangent clones now cover struct returns that hold a struct field
(`r.pos.x` after `let r = f(..)`), struct results used without a `let`
(`variance_of(f(x).energy)`, `f(x).pos.x`, `g(f(x))`), and struct returns with
second order on (a hyper-dual clone writes one Hessian block per output
field, so `hessian_of(r.energy, j, k)` is exact for callees with at most 8
tangents and no measurement inside).

Pinned by `tests/run-pass/gum_fo_tangent_clone_struct_returns2.sio` against
`tests/oracles/gum_fo_tangent_clone_struct_returns2.cpp` (1e-12 relative first
order, 1e-10 second order).

A measurement made inside a struct-returning call that feeds two output fields
now keeps their covariance: the clone exports each internal measurement once
(sigma^2 plus one sensitivity per field) and the caller makes one shared
channel for it, so `variance_of(p.u - p.w)` is exact. A struct return with more
than four such measurements is refused at compile time (KL-11), pinned by
`tests/compile-fail/gum_refuse_struct_return_internal_cap.sio`.
