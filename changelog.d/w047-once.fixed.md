W047 (`hessian_of` deprecation) now prints once per call site instead of
three times. A single `madaros build` walks each function body three times
with separate Checker instances (visibility preflight, lower verdict pass,
epistemic export), and the warning sink printed on every walk. The sink
now dedups by (code, current function, span), so distinct call sites still
warn and replays of the same site do not. Witness:
`scripts/ci/w047_once_gate.sh` counts 3 W047 lines for the 3 call sites in
`tests/run-pass/epistemic_hessian_of.sio` (was 9).
