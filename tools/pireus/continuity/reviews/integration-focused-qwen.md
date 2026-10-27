CLEAR

The code addresses both defects:

1. **PTX Duplicate Compound Labels**:
   - The `gpu_xor_compound_shape_valid` function explicitly tracks branch labels and rejects kernels with >1 label or branches after the XOR operation. The PTX emitter refuses invalid shapes with diagnostics, ensuring duplicates are caught before PTX emission.

2. **Typed Non-f64 Hyper Multiplication in HLIR**:
   - The HLIR lowering code enforces algebraic validity checks (via `hlir_checked_hyper_mul_algebra`) and dimension constraints. Invalid cases (e.g., non-octonion operands, mismatched dimensions) increment `error_count`, which is propagated to the GPU compiler and CLI, preventing scalar HLIR fallback.

Test gates validate both positive/negative cases for these defects, confirming the fixes. No remaining defects observed in the provided code.
