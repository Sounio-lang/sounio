`verify::interval::iv_from_epistemic` is gone, replaced by
`probrange_from_epistemic`, which returns the new `ProbRange` struct instead
of an `Interval`: a k-sigma probability range was never a guaranteed
enclosure, and the type system now refuses to feed one where containment is
required (passing a `ProbRange` where an `Interval` is expected is E009).
Likewise `EpistemicGate`, `gate_epistemic`, `gate_auc` and `gate_bpr` were
removed from `epistemic::ode` (they accepted any `f64`, including NaN);
`interval_exp` and `interval_log` were removed from
`epistemic::interval_ieee` (libm does not correctly round them, so no fixed
push can enclose them). None of the removed items had in-repo callers.
Pinned by `tests/compile-fail/lie01d_probrange_is_not_interval.sio`.
