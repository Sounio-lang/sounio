# Certified public worker pools and scientific pipelines on raw04

The public thread/pool constructors now require compiler-issued WorkerFnI64 or WorkerFnF64 capabilities. The raw extraction bridge lives in the physical ffi/worker module. ADC and the four independent normal kernels use named certified callbacks; no resource owner is encoded as a task integer. The F64 bridge follows Sounio's word ABI, with typed bits adapters. The serial normal branch creates no pool; the parallel branch owns four bounded pools and closes their resources once.

Qualification uses experimental integrated raw04 only: a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333. These receipts do not open Worker capabilities in a released/default compiler or qualify another compiler build. Runtime source SHA-256 is 092cf6bd6cf1bfe54857b308b532a777b1edbffd8f7b592f8e637546583fd29f; the immutable input is SDK04 copied to public-pipeline59/input. CANDIDATE.sha256 verifies that all 31 submitted worktree files are byte-identical to this qualified input.

Slurm 19711's normal gate completed0 in an 8m10 allocation. The job itself is FAILED1 because its independent pool gate hit missing libasan.so.8 (127), a preserved tool-runtime failure. Slurm 19715 retried only that pool gate with the official sanitizer runtime closure and completed0 in 4m55. Its complete pool gate includes ASAN+UBSAN. Slurm 19707 preserves the earlier private-linker failures (-ldl and Arrow dependency closure).

The full normal gate passed:

- public certified I64/F64 pools, native thread and platform consumers;
- all 14 semantic refusals at both check and compile, with no executable and no parse/unknown-intrinsic substitution;
- generated 257-row fixtures across serial, four-worker and eight-worker configurations; 1M-row serial and four-worker files; matching Arrow bytes across serial/parallel/file/TCP;
- 1M rows, 62500 Arrow batches of 16 and six columns, using the canonical four scalar kernels and an independent C++23 Arrow oracle;
- loaded spawn interposer marker plus zero events for real serial, four events for one worker per kernel and eight for two workers per kernel;
- wrong values/order, truncated/trailing/missing/extra EOS refusals; parser input errors, actual output-close EIO5, primary error preservation and complete resource cleanup;
- TCP short reads, EOF, absolute deadline110, byte budget105, truncated22, bounded queues and resource-growth controls.

Observed normal RSS was 4044→4060 KiB for serial small→1M,4040→4064KiB for parallel small→1M and 4260KiB for TCP 1M. Serial 1M wall time was 2m11.86, four-worker file 1M 1m21.27, TCP 1M1m21.55. These are one qualified execution, not a repeatability or general speed claim. The public ADC gate passed workers 1/2/4, file/TCP 1M and its independent integer/Arrow oracle. ADC RSS small/scale/TCP was 4008/3992/4800KiB; the deliberately false growth record was refused.

Only harness/tool selection changed during this qualification: explicit hashed TIME_BIN replaces the fixed /usr/bin/time path, and hashed shell compiler drivers are distinguished from ELF tools before ldd. The private compiler wrapper preserves the dynamic-library search path and adds an isolated library-search prefix. The missing runtime files were copied from official installed packages after owner/version and package MD5 verification; no package was installed globally and no link flags, test deadline, scientific algorithm or expected result was relaxed.

The original C++/Sounio references remain canonical under ADR009. Failed attempts, exact phase logs, generated inputs, ELFs, resources, spawn events and before/after guards are retained. Public API/parallel acceptance here is stronger than the earlier extracted serial-component 19540, which remains historical evidence only.
