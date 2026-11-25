Kernel parameter lowering is now driven by a signature-derived plan
(`self-hosted/gpu/gpu_param_plan.sio`) instead of a fixed arity and the literal
parameter names `n`/`a`/`b`/`c`. Roles come from the parameter type and from
store use, so a kernel that names its parameters anything else lowers the same
way.

The extent is the first `i64` parameter wherever it sits, because
`saxpy(a: f64, n: i64)` is a real kernel in the corpus; a scalar-only kernel with
no slice at all stays legal.

Shapes the plan cannot classify now fail closed with a stable diagnostic code
instead of emitting from a guessed signature: `no_extent`, `arity_over_cap`,
`no_output_slice`, `unsupported_param_type`, `multiple_extents`.

`arity_over_cap` also closes a silent truncation: the HLIR-to-GPU mapping writes
at most 8 parameter slots and used to drop the rest without a word, so a
9-parameter kernel lowered to a truncated ABI.

Emission for the previously supported shapes is byte-identical. This does not
add a device launch: `perform GPU.launch` is still lowered to a CPU call, and
`--backend gpu` still refuses a kernel body holding a `var` (KL-22).
