The static confidence of a literal `measure(v, uncertainty: u)` no longer
rises when the measurement gets worse. It was `1000 − 1000·u/|v|` while
`u ≤ |v|`, floored at 1, and otherwise the 990 default. So
`measure(10.0, uncertainty: 20.0)` cleared an `Epistemic(900)` gate that
`measure(10.0, uncertainty: 10.0)` failed with E214. The same held for
`measure(0.0, uncertainty: 0.5)`. A reading with `u > |v|`, `v = 0`
included, now gets 1, the same as `u = |v|`. Found by the EPI-006 inventory
of epsilon and confidence uses.
