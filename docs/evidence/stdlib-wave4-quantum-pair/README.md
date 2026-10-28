# Consume QubitPair once through struct decomposition

The checker correctly treats QubitPair as linear because it contains Qubit owners. Reading pair.first consumes that carrier under the current field-projection contract; reading pair.second later therefore produces E039. This change decomposes each carrier once with a shorthand let struct pattern, then moves the two separate owners to their existing destinations.

qasm_tape changes step2, walk2 and tape_replay_2. born_shot_law changes cnot_joint, cz_joint, ghz4_parity, bell_zz and partial_zz. GHZ4 uses two private helpers, matching the existing quantum::linear::ghz3_tail convention of one shorthand pattern per function. H, qubit allocation, CNOT, probability reads and measurement retain their original order. Public signatures, coefficients, gate angles, sample counts, assertions and tolerances are unchanged. No compiler, checker or runtime file is modified.

## Linear foundation

Both Token and Pair in the fixtures are explicitly linear. Slurm 19521 proves the chosen let_layers and let_take_field forms with raw11332 in loop-region modes 0 and 1: check, compile, native and independent C++23 role oracle all pass. The oracle requires first=17, second=25 and sum=42; swapped roles cannot pass merely because the sum remains 42. Its adversarial controls reject swapped roles, incorrect sum, empty output, the wrong marker and extra output.

The original repeated_projection fixture remains refused by E039 in check and compile, with no ELF. No ordinary struct was substituted for either linear resource layer.

Slurm 19527 repeats the same successful let cases and E039 negative on new raw8abe, in both modes. The job also retains the semantically valid match-based prototypes, which still crash compilation; its aggregate status is FAILED 1 because those prototypes remain broken. Per-case let acceptance is not a claim that the entire match experiment passed.

Raw compiler hashes:

- 11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4
- 8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932

## Independent numerical oracle

Slurm 19530 completes 0 in 13 seconds, MaxRSS 2400924 KiB, using raw8abe. A separate C++23 complex<long double> statevector model computes 39 observables: CNOT/CZ bit joints and covariances at zero angle variance, GHZ4 parity and marginal, Bell/basis CHSH correlations, a partially entangled state, forward/reverse CNOT tapes with Rz and H, and Bell probabilities.

The policy was fixed before execution: finite outputs, exact nonempty label selection, no duplicates/extra rows, and absolute error <=1e-10. The observed maximum absolute error is 3.712308238590367182e-16. Both loop-region modes produce byte-identical Sounio output. Before-library source refuses with E039; after source checks, compiles, runs and passes the C++ oracle in both modes. The oracle's negative controls reject a perturbed value, swapped wire roles, a missing observation and NaN.

This oracle is calculation and native concordance under ADR009, not a Lean theorem or a proof of all quantum APIs. The covariance sample uses sigma=0; no general Gaussian quadrature accuracy claim is made.

## Unchanged original programs

Jobs 19524 (raw11332) and 19533 (raw8abe) use the same source snapshots and unchanged original programs. Both show before E039 and after library checks 0. The three fast originals check, compile and execute 0 with ALL_PASS on both compilers:

- tests/stdlib/quantum/test_born_shot_law.sio
- tests/stdlib/quantum/test_qasm_tape.sio
- tests/stdlib/qpu/test_qpu_chsh.sio

Compiler deadlines remain 1200 seconds; native deadlines remain 30 seconds, matching whole19392. test_qpu_chsh_overtake_slow.sio checks and compiles 0 after the fix and is running under its original explicit 3000-second native deadline. At 2026-10-28 23:01 UTC both long jobs remain RUNNING; the slow native result is PENDING. No full slow acceptance, zero-failure whole-stdlib claim or final long-job guard is inferred from the fast results.

LEDGER.json records exact before/after library hashes and the eight affected consumers. The before/after source and immutable original programs are preserved under services/quantum-pair-library25 and quantum-pair-library28. The exact executed scripts are versioned with this receipt.

## Preserved match-lowering frontier

The initial prototypes use match to destructure Token, including the helper shared by the initial let prototype. They check successfully but crash compilation with SIGSEGV 139 on both raw compilers. Discriminators preserve the distinction: Token consumed by a single projection or let works; Token match and Pair match fail; changing linearity, primitive/nested fields, constructor nesting, names, multiline declarations and annotations does not make the match path valid.

Slurm 19513 reuses root's qualified ptrace tracer without modification. It records RIP 0x5997d3c, fault address 0x8, RAX=RBX=0 during lower_array: seed_begin. This is distinct from the separate chemical report_error_at RIP 0x587464a. The source and 32-frame traces remain diagnostic evidence; no lowerer workaround or change is made here.

No push or merge to main occurred. The library source can be reviewed independently of the still-running slow witness and the separate compiler match defect.
