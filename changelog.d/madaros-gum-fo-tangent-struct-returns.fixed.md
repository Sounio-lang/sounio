Madaros: first-order GUM through user calls that return a struct, and through
struct parameters with nested struct fields (KL-11). The tangent clone
(`<name>__fotan`) now also covers:

- **Struct returns.** A helper `fn f(...) -> S`, where `S` has scalar `f64`
  fields, is called through its clone. The clone writes a Jacobian matrix: for
  each scalar `f64` field of `S` and each tangent, `∂field/∂tangent`, plus the
  variance of measurements made inside the call for that field. The caller
  applies the chain rule per field when the call is bound by a `let`, so
  `variance_of(r.x)`, `variance_of(r.x - r.y)` and
  `variance_of(r.x - g(input))` are exact and keep the correlation between the
  fields and with every other use of the same measurement. Each return of the
  clone is a struct literal (each field's initialiser gives the row) or a
  local struct (the clone keeps the per-field tangents where the literal was
  built); early returns inside loops work.
- **Nested struct fields.** The tangent layout of a struct parameter (or
  receiver) is enumerated depth-first through struct-typed fields, up to
  depth 3 and the existing 24 tangents. `p.pos.x`, `(*p).pos.x` and
  `self.pos.x` resolve to their own channel in the clone, and the caller walks
  `<local>.pos.x`.

Before this, a loop-bodied helper returning `{energy, force}` read variance
0.0 on every field, and `site_phi(a)` over `Atom { pos: Vec3, q }` kept only
the `q` channel (0.000878 where the first-order value is 0.000973).

Not covered, and left on the previous path: a struct return whose type has a
struct-typed field; a returned struct built with a field assignment or held in
a `var`; a return whose operand is neither a struct literal nor a local (an
`if` tail, a call); a struct result that is not bound by a `let` (a direct
`f(x).energy` chain); a measurement inside the call that feeds two output
fields (each field gets its own independent channel for the internal
variance); second order (a struct-returning call keeps the re-lowering path
when `hessian_of` is in use).

Pinned by `tests/run-pass/gum_fo_tangent_clone_struct_returns.sio` against
`tests/oracles/gum_fo_tangent_clone_struct_returns.cpp` at 1e-12 relative.
