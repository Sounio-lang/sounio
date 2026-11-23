Madaros no longer evaluates a call argument twice. Its first-order variance
walk re-lowers the arguments of a call, and it counted a call to a known inline
helper as safe to re-lower whenever the helper's effects were within
`Mut`/`Div`/`Panic`. `with Mut` admits a write to a global, so in
`k3(bump(1.0), 0.5, 0.5)` the side effect of `bump` happened twice, and
`demos/hydrogen/smr_h2_lcoh.sio` drew 7 Gaussians per Monte Carlo iteration
instead of 4 (mean LCOH 3.119014 on Madaros, 3.122075 on lean_single). A helper
is now re-lowered only when it assigns no name it did not bind and calls only
pure math builtins or other such helpers; otherwise the argument is lowered
once and the call carries no inlined variance. Programs affected by the defect
now print what lean_single prints (`tests/run-pass/impure_call_arg_evaluated_once.sio`).
