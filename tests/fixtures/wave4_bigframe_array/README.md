# BigFrame scaled median: array representation diagnostics

This package preserves the original scientific test and production library. It collects type-check, compilation, native-exit and independent C++23 bit-oracle results for19 producers in both loop-region modes. It does not decide the language contract for casting an array value or array reference to a raw pointer.

Run inside Slurm:

    bash tests/fixtures/wave4_bigframe_array/diagnostic.sh RAW_COMPILER STDLIB FRESH_ABSOLUTE_OUTPUT

The output is deliberately a diagnostic collection. Exit0 and ARRAY_DIAGNOSTIC_COLLECTED mean all source/tool guards and the nonempty38-case/152-phase collection completed. Actual behavior failures remain in PHASES.tsv; they are never turned into passes or skips.

GCC defaults to CXX=g++. On a worker without a system compiler, SOUNIO_GCC_SYSROOT may name the existing qualified private GCC13 sysroot, including its deps directory. The driver, frontend, headers, libraries and output executable are hashed; the sysroot is not installed into the system. C++23 computes all numerical expectations. Three actual wrong-output files must be refused by the compiled verifier.

The array controls contain4 and1024 elements. They compare typed element access, casting an array value, casting a mutable array reference, and passing either form through a helper. ABI controls exercise6 versus7 word arguments, mixed integer/f64 arguments, a five-argument wrapper calling a seven-argument helper, and scalar versus BigFrame returns. Every argument contributes to the expected value; the seventh sentinel is707. Float results are compared as exact IEEE754 words.

The scaled pair uses the same five-row source and original median assertion. The index variant copies the supplied stdlib into the fresh output, then changes exactly one expression within bf_group_quantile_scaled: read_f64(res as *mut f64, k) becomes res[k]. The original library remains hashed and unchanged. This is a diagnostic contrast, not an adopted repair or a generic ABI workaround.

The initial ABI observation directly printed f64_to_bits(...); native lowering refused its unresolved print scalar kind. The final fixtures bind the exact same expression to an explicit i64 local before printing. Numerical computations, inputs, sentinels and oracle expectations are unchanged. Initial refusal logs are preserved.

The component evidence is documented in docs/evidence/stdlib-wave4-services/bigframe-array. ABI19426 passes18 controls and the two index variants; its two original scaled variants still SIGSEGV. Array19429 passes all four typed controls, records six value-cast SIGSEGV cases and six reference-cast wrong-bit cases. Collection success is separate from those failures.
