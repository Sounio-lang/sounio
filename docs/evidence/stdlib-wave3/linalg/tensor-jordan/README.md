# Accumulate a Hessian source in one tensor result

compute_orbit previously invoked the functional tensor16_set for each coefficient of the Jordan curvature source, copying a32KiB tensor payload on each update. tensor16_add_jordan_source performs the identical j/k/i traversal and separate symmetric off-diagonal additions in one result tensor. The orbit recursion, iteration count, escape condition and arithmetic order remain unchanged.

Slurm19156 used the same raw compiler and original test_neuro.sio bytes before/after: old library native181; new library native0 with neuro tests passed=16 failed=0. The composite job then failed because the newly written oracle fixture used undeclared print_float. That failed fixture is preserved; its correction uses the actual print_f64 builtin and does not change the library or original neuro test.

Slurm19165 checks/compiles/runs the corrected tensor fixture and validates every coefficient with an independent recursive Cayley-Dickson C++23 reference. Three cases (zero, identity and a dense binary-rational matrix) include an asymmetric nonzero initial tensor. All12288 coefficients match exactly. A deliberately wrong expected coefficient is rejected with exit1. This is executable arithmetic verification, not a Lean theorem.

Slurm19174 reruns the committed portable gate: unchanged original neuro test, tensor fixture, C++23 full coefficient comparison and wrong-value control all pass. All input, compiler and native executable hash checks pass. Raw compiler bd4ba2409a0097fdc9e9b4fe32e5858e396d4d71dd3515648a6f01d451b8ace1 uses opt-in loop regions in both old/new runs. The original neuro test is byte-identical; no iteration count, deadline or arena capacity was reduced.

Reproduce only on a scheduled Linux x86-64 worker:

    MADAROS_RAW_BIN=/absolute/rebuilt/madaros bash scripts/stdlib/tensor_jordan_gate.sh /absolute/fresh/output

This closes the reproduced neuro arena exhaustion and the bounded coefficient concordance gate. It does not establish whole stdlib acceptance or a general tensor compiler/allocator bound.
