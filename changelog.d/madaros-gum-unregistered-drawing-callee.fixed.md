Madaros: a query on the result of a call to a function that has no tangent
clone and can draw a measurement (directly or through what it calls) is
refused at compile time ("refused instead of reading 0", KL-11) whatever the
arguments. With certain arguments the callee's own measurement uncertainty was
silently dropped (variance 0). Pin:
`tests/compile-fail/gum_refuse_unregistered_drawing_callee.sio`.
