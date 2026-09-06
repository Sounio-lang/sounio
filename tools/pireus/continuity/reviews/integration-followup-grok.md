## Scope (not defects)

- Native remains ptr3-only. Generic virtual `IrXorConvolutionS` (`label_id != 1` or non-AUTO at the v2 gate) fails closed; that is not whole-language / expression-level native support.
- HLIR candidate `3` and IR candidate `1` are separate domains; GPU precheck does not integer-copy them. `TWO_ZMM` is a GPU refusal, not a shuffle rewrite.
- Missing `Hyper<…>` scalar argument is a checker error, not default-`f64`.
- Octonion HLIR `(d,i)` keeps the same per-`d` ascending-`i` subsequence as the old inner loop. Do not treat that as extra FP reassociation.
- GPU effect id `5` matches `effects.sio`. Host `&!T` stores still require `Mut`.
- Metal `float2` recipe is explicitly approximate. Empty callee + checker spans are the GPU identity (name poison cannot select it).
- `GpuXorConvolutionF64` is appended; `dgx-sm121` alone selects `.target sm_121`.

## Defects

1. **PTX/DGX still silently materialize Metal-refused compound shapes, and the XOR emitter cannot do so correctly.**
   Metal now requires exactly one terminal `GpuXorConvolutionF64`, positional `params[0..2]`, and refuses duplicate / mixed / post-op branches. PTX does not. `gpu_emit_xor_convolution_f64` / `_global_f64` bake function-local labels `$XCJ`/`$XCDONE` and `$XCGJ`/`$XCGDONE`. Two deferred sedenion stores in one kernel (`*o1 = *a * *b; *o2 = *c * *d`) pass `hlir_module_xor_contracts_valid`, then emit **duplicate labels** (and `bra $XCDONE` can bind to the first terminator and skip the second product). That is successful bad PTX, not an unsupported-input refusal. Unique labels per emit, or the same compound-shape refusal as Metal, are required before this is closed.

2. **Checker-typed Hyper multiply can still miss HLIR oct/sed kinds and fall through to generic binary lowering.**
   Refusal is only `checked_algebra < 0 && hlir_oct_is(lhs|rhs)`. The operator path needs **both** oct-recorded. `Hyper<Sedenion, f32>` (and any other typed mul that never becomes `hlir_type_sedenion`/`octonion`) can still carry epistemic `op_kind=1` and then take ordinary HLIR `*` — the same class of accidental scalar lowering the metadata check was meant to stop, just on the complementary branch. Non-f64 / non-oct typed Hyper mul should refuse, not fall through.

Not CLEAR.
