# Integrated stdlib replay: diagnostic baseline

Slurm job 19392 replayed source add24582e1e109e84536242b77ad19f70da96cd0 with raw Madaros SHA256 11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4, SOUNIO_LOOP_REGIONS=1, and the snapshot-only Worker opt-in. Production Worker issuance remains closed. This is a diagnostic baseline, not acceptance or a zero-failure result.

The runner recorded 733 files: 670 pass, 53 fail, 10 skip. Strict acceptance status is not_run because skipped files are prohibited. The 53 failures comprise 11 compiler SIGSEGV, 2 other compile failures, 12 check failures, 18 native timeouts, and 10 other native failures (including 3 SIGSEGV). Original deadlines and assertions were retained. No successful rerun replaces this result.

Stdlib matched the compiler build snapshot byte for byte. Final source, compiler/tool, and native runtime/oracle manifests all verified. The worker copies and captured native executables are retained in the durable archive referenced by CUSTODY.txt.

Nine skipped fragments have separately executed compound coverage in job19400: BDF(1), FEM(2), FFT(3), numdiff(3). All four gates returned0. Their input manifests bind the fragments and generated programs. They remain skips in the historical replay JSON; this does not retroactively change runner acceptance. The remaining token_bench fragment is covered independently by job19458, using a C++23 materializer and integer oracle for both actual public parser representations.

Compiler crash tracing job19414 preserves the original chemistry compile139 failures and a positive catalysis compile/native control. The two chemistry failures occur in dependency11 lowering, with fault address -1 and RIP0x587464a. Job19450 repeats the pair with SOUNIO_INTO_ACC_NO_RESET=1 and retains both failures: disabling reset does not remove this defect. This option is a discriminator only, never a production fix. An accidental duplicate submission19454 was refused by the existing fresh-directory guard; it did not overwrite results, though its shared slurm.log was overwritten. The original19450 controller receipt is retained independently.

The token gate job19458 completed0. Each representation checked, compiled, ran, and agreed with C++23:100000 rows,200000 bytes,100000 total,finished1,status0. Independent wrong-total, missing-marker, and duplicate-stream controls were refused with8,9,9. This is correctness evidence, with no throughput claim.
