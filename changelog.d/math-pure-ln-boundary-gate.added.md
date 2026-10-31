# math::pure ln boundary gate

Adds `scripts/stdlib/math_pure_ln_boundary_gate.sh` (wired into
`deep_science_gate.sh`): compiles `tests/run-pass/math_pure_ln_pos_inf_terminates.sio`
with a source-built Madaros, checks `ln(+inf)` terminates against the C++
reference in `tests/reference/math_pure_ln_boundary_reference.cpp`, and runs a
guard-removed negative control that must time out (discriminating).
