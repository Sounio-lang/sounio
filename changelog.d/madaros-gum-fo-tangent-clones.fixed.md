Madaros: first-order GUM variance now propagates through calls to user
functions whose bodies reassign a `var`, loop, return early, recurse, call
other helpers or measure inside. Before, `variance_of` of such a call's
result read 0.0, with no diagnostic. The call-site re-lowering (#1706) stops
at the first statement that is not a `let` and refuses loop bodies outside
explicit uncertainty queries.

Mechanism. Take a function that returns `f64` and whose parameters are
`f64`, `Knowledge`, integer or bool (at most 15, no receiver, no epistemic
query in the body). When it is called from an `Epistemic` function, the call
goes through a tangent clone, `<name>__fotan`. That is forward-mode AD at
the call boundary:

- The clone is the function lowered once more, with each `f64`/`Knowledge`
  parameter seeded as its own unit channel.
- It takes one hidden last parameter, a caller-owned `[f64; nf + 1]` array.
- At every return it writes the run-time Jacobian row `∂f/∂p_k` into the
  array. Slot `nf` holds the variance from measurements made inside the
  call.
- The caller applies the chain rule on its own channels:
  `s_c = Σ_k (∂f/∂p_k) · s_c(a_k)`. The internal variance is added as one
  independent channel.
- Correlation with every other use of the same measurement is kept, e.g.
  `kn_loop(k) - 3*k.value` has variance 0.
- The rows the intra-procedural lowering already carries through loops,
  branches and reassignment give the derivative. Nothing is re-lowered at
  the call site.

Clones are lowered on demand after the requesting function, transitively
(a clone's own calls go through clones, so recursion works too). A clone
also works when the callee lives in an imported module, including one that
reads its module's globals.

Measured against analytic C++ oracles at 1e-12 relative tolerance, in
`tests/run-pass/gum_fo_tangent_clone_calls.sio` with
`tests/oracles/gum_fo_tangent_clone_calls.cpp`. The witnesses are var
reassignment, a loop, correlation with the input, recursion, a
three-shell Lennard-Jones sum over two measured parameters, a measurement
inside the call, the same call twice, a partition function with an early
return, and a Boltzmann sum over a `Knowledge<WellDepth>`. Main's compiler
fails 7 of the 9.

The differential against main covered 536 programs: every test and example
that uses `measure`, `Knowledge` or `variance_of`, or declares `Epistemic`,
plus 140 other run-pass tests. 533 have byte-identical output, compile result
and exit code included. The new pin is one of the other three. The remaining
two moved toward the right value:

- `variance_deep_loop` now prints `var(acc)=0.002385`, the exact
  first-order value from forward-mode duals in C++. Before, it printed
  0.000100, which is only `Var(k)`.
- `rapamycin_epistemic_adaptive` (#1792 F1) now matches the `lean_single`
  adaptive-step statistics exactly: 56 accepted steps, 39 active, 5 shrink
  and 1 grow signals, final dt 0.006739. Before, it took 45 steps with
  final dt 0.104106.
