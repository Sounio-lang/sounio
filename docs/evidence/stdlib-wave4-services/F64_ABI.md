# Native f64 worker ABI witness

Slurm 18822 completed with ExitCode 0:0 (5 seconds). It executed the unwrapped
Madaros native ELF, not an interpreter. The runtime was compiled with C++23,
-O2 -Wall -Wextra -Werror -shared -fPIC -pthread.

Immutable run:
`/scratch/sounio-stdlib-wave3-20261027/services/wave4-f64-abi01`

Raw compiler:
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/named-ref-v2/results/madaros`
SHA256:
`21b3b011c0d621041d12ba1661f756b55315eb78d0ce071ec1cad2f5e5ab26d2`

The SOURCES.sha256 and EXECUTABLES.sha256 manifests pin source, compiler,
runtime shared library and emitted ELF. final.verify succeeded.

Native controls:

- identity preserves exact bits of +0, -0, 0.5, and quiet NaN payload 0x42;
- half(1.0) produces exact binary64 0.5;
- task IDs and all POSIX statuses are checked independently of scalar results;
- shutdown, join, result discard and resource close all return zero;
- the platform resource count is zero before platform_close.

The markers are NATIVE_F64_WORD_ABI_OK and F64_WORD_ABI_GATE_OK.

Both Sounio scalar ABIs carry their argument in RDI and return in RAX. The
existing adapter saves the native compiler's scratch use of callee-saved
registers. Operation140 adds the F64 creation entry and reuses that word
adapter. It does not call a C double(double) function or use XMM0 as the ABI
boundary. Operations131/132 carry IEEE-754 words, converted by the typed stdlib
with f64_to_bits/bits_to_f64.

This witness intentionally exercises raw FFI. It does not establish safety of
arbitrary Sounio callbacks. The safe WorkerFnI64/WorkerFnF64 public API and its
transitive compiler proof are separate, pending qualifications. dladdr is a
runtime guard, not a certification mechanism. C++ catch(...) does not claim
to catch Sounio panic, faults, or syscall exit.
