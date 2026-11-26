Madaros now finds the field read by `variance_of(s.f)` and by first-order
sensitivities in the receiver's own struct type. It used to look the field up
by name in the first struct that declared one, so an unrelated earlier
`struct X { pad: f64, val: f64 }` moved `s.val` to offset 8: the variance was
read from the wrong channel and printed 0 instead of 0.25, with exit 0.
Nested chains (`o.inner.v`) had the same defect. lean_single was already
right.
