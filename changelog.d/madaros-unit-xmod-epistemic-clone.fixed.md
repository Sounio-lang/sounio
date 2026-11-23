A function whose parameters are typed by a `unit` declared in its own
module now returns the right value when it is called from a function that
carries `Epistemic` in another module. Such a call goes through the
callee's tangent clone (first-order GUM through the call), and the clone
was lowered under the caller's module id, where the callee's units are
unknown. Each unit parameter was then an integer word, so `as f64`
converted the double's bit pattern: `(ka * dt) as f64` with `ka = 2.0 /h`
and `dt = 0.25 h` returned 2^60 instead of 0.5, and `0.693 / ((cl / v) as f64)`
divided two words. The same calls from a caller without `Epistemic`, or
with the unit declared in the caller's file, were already right. The clone
is now lowered under the module that registered the callee (`TG_MODULE` in
`self-hosted/ir/lower.sio`). Test: `tests/run-pass/unit_xmod_epistemic_clone.sio`.
