EPI-006, step 1: the checker keeps a stated confidence apart from the error
bound. `Knowledge(v, ε = c)` is a point value with confidence c. It used to
store c in the error-bound slot, so `Knowledge(0.0, ε = 0.05)` met
`Knowledge[f64, epsilon < 1.0]` as if 0.05 were a variance. In the other
direction, a `ε >= x` floor compared a measurement's variance against the floor,
so `measure(10.0, uncertainty: 0.9)` (variance 0.81) met `ε >= 0.5`. Both are now
refused. `ε < x` / `ε <= x` are error bounds, `ε >= x` is a confidence floor,
and `ε = x` is an exact confidence. Seven `ns_*_trace` tests annotated their
`Knowledge(.., ε = ..)` values with an error bound they never had. They now use
`Knowledge<f64>` and pass on both engines; before, neither engine passed them.
