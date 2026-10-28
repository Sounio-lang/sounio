# Normal pipeline oracle component

Slurm 18841 completed with ExitCode 0:0, elapsed 10 seconds.
Run: `/scratch/sounio-stdlib-wave3-20261027/services/wave4-normal-oracle01`.
The immutable source snapshot and results are in src/ and results/.
The gate explicitly writes PARTIAL and WAVE4_NORMAL_ORACLE_ONLY_OK;
it does not write the full-pipeline COMPLETE marker.

C++23 builds the independent oracle with Arrow C++26.0.0. Its scalar reference
uses erfc, log and log1p in long double, guarded by digits >= 64. The intended
application input is the exact binary64 result of z_milli/1000, for integer
z_milli in [-40000,40000]. The oracle promotes that input to long double;
it does not accidentally use a different unrounded mathematical input.

The 257-point fixture includes z=0, +/-0.001, +/-0.5, +/-1, +/-2, +/-5,
+/-8, +/-10, +/-20, +/-38 and +/-40, plus deterministic interior samples.
One-million-row fixtures repeat the same points with consecutive sample IDs.
The fixture is deterministic test data, not an acquired scientific dataset.

The C++ self-test writes and verifies six nonnull Arrow columns with at most16
rows per batch. Six actual malformed artifacts are rejected with exit1:
wrong probability, wrong order, truncated bytes, trailing bytes, missing
entire EOS8, and duplicate EOS8. Valid input exits0. These controls test the
oracle/protocol, not Sounio numerical concordance.

The full gate remains pending WorkerFn-capable raw compiler and canonical
normal_scalar kernels. It will compare native Sounio file/TCP outputs with
the C++ oracle and compare serial/parallel Arrow bytes exactly. The planned
probability/log relative tolerance is 2e-12 with an eight-denormal absolute
floor; no passing Sounio numerical result is claimed by this component.

Manifests pin compiler/toolchain/source/SDK and dynamic dependencies. Every
final verification returned0 in the oracle-only run.
