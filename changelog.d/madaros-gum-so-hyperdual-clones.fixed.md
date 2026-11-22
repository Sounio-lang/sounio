Madaros: second-order GUM (`hessian_of`, `second_order_mean`) now propagates
exactly through calls to user functions whose bodies loop, reassign a `var`,
return early, recurse or call other such helpers. Before, the Hessian of such a
call read NaN (the first-order tangent clone carries no Hessian), or a silent
0.0 when the call appeared only inside the query. `second_order_mean` of the
call dropped its curvature term without a diagnostic (1.827996 where the
exact value is 1.828498).

Hyper-dual clones. A function that calls `hessian_of` or `second_order_mean`
and calls a registered helper (an `f64` return; `f64`, `Knowledge`, integer,
bool or struct parameters, at most 8 tangents, no measurement inside) goes
through `<name>__fotan2` instead of `<name>__fotan`. The clone is the helper
lowered with second order on. At every return it writes, after the gradient
row `J_k = df/dp_k`, the Hessian `Hf_kl = d2f/dp_k dp_l` over its tangents
(upper triangle) into the caller-owned out-array. The caller applies the exact
second-order chain rule on its own channels,
`H = sum_k J_k H(a_k) + sum_kl Hf_kl g_k g_l^T`, with `g_k` and `H(a_k)` the
gradient and Hessian of argument k. Mixed partials between two measured inputs
passing through one call, arguments that are expressions of the caller's
channels (`shell(x * y, y)`), and correlation with the caller's other uses of
the same input (`hessian_of(f(x) - x * x, 0, 0)`) all come out of the same
rule. A straight-line helper still goes through the call-site re-lowering
(#1706), which is exact for it and unchanged; a helper that assigns a `var`,
loops or returns is read by that re-lowering without its Hessian, so it now
uses the clone directly.

`hessian_of(f(x), j, k)` where the call occurs only inside the query has no
value evaluation to hang a clone on: the clone call is emitted inside the
query lowering itself, which then is its only evaluation. Restricted to
callees with no IO effect (`second_order_mean` evaluates the value again
afterwards, so a pure callee is evaluated twice).

Second order inside one function. The clone needs the Hessian row of its own
body, and the intra-procedural walk dropped it at every assignment: a `var`
that was reassigned, or accumulated in a loop, kept the Hessian of its
initialiser (`var s = 0.0; while ... { s = s + exp(-e * i) }` read
`hessian_of(s, 0, 0) = 0` in plain code, no call involved). A mutable `f64`
local now owns private Hessian slots next to its gradient slots, zero at
declaration; assignment copies the expression's Hessian row into them and loop
entry extends them, as the gradient rows already were. Two Newton steps from
`y = x` at x = 34.2 now print -0.000046 (was 0.000000; a C++ second
difference of the same iteration gives -4.585e-05).

Stale rows. The per-local Hessian rows were not cleared when a bind row was
created, so a row reused after an earlier function read that function's
registers: `epistemic_hessian_of`'s third check printed 40 (the file asks for
10 = 2y) when it ran after the second, and 10 alone. Rows are now cleared on
creation.

Not covered (still NaN or unknown, not a wrong number): a callee with more than
8 tangents or a measurement inside it keeps the first-order clone and its NaN
Hessian; `hessian_of(f(x))` of an IO callee whose value is not used elsewhere
reads 0; a `var` declared before the first `measure` has no private Hessian
slots (its gradient rows have the same limit, #3547); the Hessian rows cover
the first 8 channels and the first 16 bound locals.

First order stays byte-identical when no Hessian is requested: the generated
ELF of `gum_fo_tangent_clone_calls`, `gum_fo_tangent_clone_structs`,
`variance_deep_loop` and `rapamycin_epistemic_adaptive` is identical to
before.

Pins: `tests/run-pass/gum_fo_tangent_clone_second_order_on.sio` (the
`hessian_fallback_nan` check is now `hessian_fallback`, the exact d2/dx2 of six
Newton steps of sqrt at x = 20, -0.0027950849707661; plus loop, mixed partial,
`second_order_mean`, correlation, composed arguments, recursion, early return,
query-only) and `tests/run-pass/gum_so_hyperdual_clones.sio` (struct and
method tangents, branch, early-return loop, nested loops, Knowledge parameter,
IO callee, nine tangents). Oracle: `tests/oracles/gum_so_hyperdual_clones.cpp`,
second-order jets (value, gradient, Hessian) carried through the same
algorithms, 1e-10 relative; the values agree to within 1.5e-16 relative.
