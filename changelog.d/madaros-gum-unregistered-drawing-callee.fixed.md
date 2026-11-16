Madaros: a query on the result of a call to a function that has no tangent
clone and can draw a measurement (directly or through what it calls) is
refused at compile time ("refused instead of reading 0", KL-11) whatever the
arguments. With certain arguments the callee's own measurement uncertainty was
silently dropped (variance 0). Pin:
`tests/compile-fail/gum_refuse_unregistered_drawing_callee.sio`.

The scan also covers methods (resolved through the receiver type at the call
site, by short name inside bodies), `Type::f` paths, and calls through a
function pointer, a parameter or a closure that the HOF specialisation does not
resolve (assumed able to draw). Pins:
`tests/compile-fail/gum_refuse_unregistered_drawing_{method,fnptr,closure}.sio`;
`tests/run-pass/gum_method_no_measure_exact.sio` (oracle
`tests/oracles/gum_method_no_measure_exact.cpp`).
