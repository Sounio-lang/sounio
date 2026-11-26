New builtin `hessian_wrt(z, x, y)`: second-order derivatives indexed by
source identity instead of positional channel indices. `x` and `y` are the
measured `Knowledge` bindings themselves, resolved at compile time to the
channel each source was seeded with, so a dead `measure()` removed by DCE —
or reordered code — can no longer silently renumber the question
`hessian_of(z, j, k)` answers. A source that is not a measured source the
value depends on is refused with E312. `hessian_of` keeps working but now
emits the deprecation warning W047 naming `hessian_wrt`.
