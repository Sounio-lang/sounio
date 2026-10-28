# Normal pipeline component qualification

These receipts qualify a source-profile component and the real serial
CSV/normal-kernel/Arrow execution core. The public imported pipeline,
WorkerFn IR cloning and concurrent execution remain pending.

| Slurm job | Component | Result |
| --- | --- | --- |
| 19026 | Exact normal-kernel source closure, before/after checker | COMPLETED 0:0, 2 s |
| 19032 | First serial launch | FAILED 1:0 before compilation: component staging incomplete |
| 19035 | Serial file input, 257 and 1,000,000 rows | COMPLETED 0:0, 2 min 6 s |
| 19040 | Serial TCP input and adversarial inlet controls | COMPLETED 0:0, 2 min 30 s |
| 19045 | Initial custody collector | FAILED 1:0: expected a local TOOLS manifest for a run using shared sealed tools |
| 19047 | Correct shared-tool verification, archive and readback | COMPLETED 0:0, 9 s |

Failed jobs and their logs are retained. The serial launch was repeated in a
fresh directory after staging completed. The custody correction checks the
actual shared tool manifest; numerical sources, inputs, or acceptance limits
were not changed for either repeat.

## Source-profile component

The input contains 27 complete function definitions extracted literally from
canonical normal/math sources at `ef1961e527`; each function body has a SHA256
in `FUNCTION_BODIES.tsv`. Four `worker_fn_f64` constructors target
`normal_cdf_scalar`, `normal_sf_scalar`, `normal_logcdf_scalar` and
`normal_logsf_scalar`.

The checker from job 18989 accepts that closure, including confined scalar
stack outrefs and their forwarding. Three controls append transitive mutable
global access, an indirect call, or an aggregate-returning helper. All three
are accepted by the previous checker and rejected by the new checker with
`WorkerFn source closure refused`. The actual verdicts are 0 before and 1
after. Parser and unknown-intrinsic failures are excluded.

This driver loads one file. It does not test module BFS resolution, physical
FFI origin, final IR, region-barrier elision, emitted callback ABI, or execution
on worker threads.

## Serial execution and independent Arrow oracle

The serial component keeps the literal parser, header check, scalar emitter,
serial executor and configuration/summary definitions from the pipeline WIP.
Its driver calls the serial executor directly, without importing or replacing
the typed pool implementation. The canonical math/statistics modules are
imported from the staged stdlib. No pool is constructed.

Input is `sample_id,z_milli`, with consecutive IDs and z in [-40,40] represented
as integer thousandths. The generated test fixture contains 257 distinct z
values, repeated for the scale case; it is numerical test data, not an acquired
scientific dataset.

The independent C++23/Arrow26 verifier checks every row, six non-null columns,
order, values, at most 16 rows per batch, end-of-stream and absence of trailing
bytes. Its probability/log tolerance stays 2e-12 with an eight-denormal absolute
floor. The same verifier source previously qualified the six corrupt-Arrow
controls; these component runs do not replace those receipts.

Both file and TCP cases produce 1,000,000 rows in 62,500 batches. Their complete
Arrow files compare byte for byte, SHA256:

`0bd9e8d4c60d95dfe232324b95ea49944ac637e8aaa0fc24094fb08a084d0cee`.

TCP also passes the 257-row normal-send and one-byte-at-a-time send cases.
The following actual native error exits are required and observed:

| Inlet control | Exit status |
| --- | --- |
| Fixed absolute 10 ms deadline with peer holding the stream open | 110 |
| Byte budget 16 | 105 |
| Unterminated quoted field at EOF | 22 |

The socket reader owns its duplicated descriptor, the source socket is closed
before parsing, and the read lease rejects a concurrent socket receive with
EBUSY. All success and refusal drivers assert zero platform resources before
closing the platform. Work/cleanup status is zero in the successful cases.

An immediate pthread interposer records L when loaded in the target executable
and S after each successful pthread_create. A C++ witness creates and joins one
real thread and yields LS. Every serial run yields exactly L, including both
one-million-row cases. Exit destructors are not used to count thread creation.

Observed maximum resident memory:

| Case | KiB |
| --- | --- |
| File, 257 rows | 3220 |
| File, 1,000,000 rows | 3168 |
| TCP, 1,000,000 rows | 3248 |

This is an observed bounded-memory component result. The elapsed job times
include compilation and validation and are not performance comparisons.
Serial/parallel bit identity and the 0/4/8 worker-creation matrix remain gates
for the complete certified compiler and public pipeline.

Raw compiler:
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/named-ref-v2/results/madaros`,
SHA256 `21b3b011c0d621041d12ba1661f756b55315eb78d0ce071ec1cad2f5e5ab26d2`.
C++ compiler is Ubuntu GCC 13.3. These component runs seal staged source,
raw/compiler tools, Arrow shared library, inputs and generated executables;
the complete gate additionally seals SDK headers and dynamic dependency
manifests.

## Durable custody

Scratch prefix:
`/scratch/sounio-stdlib-wave3-20261027/services/`.

Runs:
`normal-profile-component01`, `normal-serial-component01`,
`normal-serial-component02`, `normal-serial-tcp-component01`,
`normal-components-custody01` and `normal-components-custody02`.

Archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/normal-components-export02/normal-components.tar.gz`.

SHA256:
`f8a81225324160d38f083573626862707fa471ff4af9ca1bc495c018315c6ad8`.

Job 19047 rechecks source/tool/input/executable/artifact manifests, archives the
component runs and preserved failures, lists the archive, syncs it and verifies
its hash by readback. Its marker is `NORMAL_COMPONENTS_DURABLE_OK`.
