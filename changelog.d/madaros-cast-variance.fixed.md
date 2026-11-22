`variance_of` no longer reads 0 for a value that passed through `as f64`
(or `as f32`, or a cast to a unit such as `h`). The first-order GUM
lowering had no arm for a cast, so the cast dropped the value's variance,
sensitivities and Hessian. Leaving a unit type takes a cast
(`(ka * dt) as f64`), so every unit-typed helper returned an exact
number under `Epistemic`, with no diagnostic, whether the unit was
built in, declared in the same file or imported. A cast to a float or a
unit now passes the value's uncertainty through unchanged. A cast to an
integer gives the same answer as before. Test:
`tests/run-pass/unit_cast_variance_epistemic.sio`.
