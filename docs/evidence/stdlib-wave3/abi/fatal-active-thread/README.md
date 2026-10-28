# Fatal process exit with a live native thread

Slurm19042 completed0 using the portable gate and the exact committed fixture.
The same source and same native runtime DSO are compiled by before raw21b3b011c0d621041d12ba1661f756b55315eb78d0ce071ec1cad2f5e5ab26d2
and after rawf3ceba1af545a006ab61c66dafde4bbc5eb63c0b1463f63d6ed2f3dd853cb332.
Both check and compile successfully.

A capacity-one channel is filled before starting a general C++ range producer.
The producer's absolute deadline is INT64_MAX. Main waits for its actual wait-path
counter to become positive, then prints FATAL_WITH_ACTIVE_THREAD_ARMED and calls
assert(false). No Sounio callback is run on the child thread. Cleanup operations
after the assertion keep the linear source path complete; a correct fatal exit
does not reach them.

Both executables print the handshake marker. The old executable leaves its native
thread alive and reaches the five-second timeout (124). The new executable exits
with code1 before that timeout. Thus the exit_group change is exercised with a
real active native thread, not inferred solely from syscall bytes. The timeout
supervisor bounds and cleans up only the test process group. Source, compiler,
runtime DSO and executable guards all pass.

The initial identical experiment19029 also passed. Full portable artifacts are at
/scratch/sounio-stdlib-wave3-20261027/fatal-thread-portable/result; small receipts
are in this directory. This closes the active-thread fatal-exit component gap in
SCALAR_CAST_PROCESS_EXIT.md. It does not establish general IO callbacks, TLS,
WorkerFn clone certification or whole-stdlib acceptance.
