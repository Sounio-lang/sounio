Madaros: first-order GUM through user calls whose tangents live in structs.
A tangent clone (`<name>__fotan`, see the tangent-clone entry) now also
covers:

- struct parameters, by value or by reference (`p: LJ`, `p: &LJ`);
- methods, with `self` by value or `self: &T`;
- a method that calls another method on `self`.

Every scalar `f64` field of each struct parameter and of the receiver is
one tangent. Both the clone and its callers lay the Jacobian row out in the
same order: plain `f64`/`Knowledge` parameters, then the struct fields in
declaration order. The caller reads each field's sensitivity through the
argument's local (`p.eps`), so a struct argument must name a local (`p`,
`&p`, `&!p`); any other argument keeps the ordinary call.

Before this, a struct or receiver whose body loops read variance 0.0 on
Madaros, because the #1706 re-lowering refuses loops outside explicit
queries. Pinned by `tests/run-pass/gum_fo_tangent_clone_structs.sio`, with
expected values from `tests/oracles/gum_fo_tangent_clone_structs.cpp`
(a three-shell Lennard-Jones energy with `eps` as a struct field, at 1e-12).
Main fails 5 of its 6 checks.
