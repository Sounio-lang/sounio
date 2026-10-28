# Arrow abort and cleanup status

The additive API `arrow_stream_writer_abort(p, writer)` consumes the writer,
releases its byte buffer and closes its native file without writing EOS.
It reports a real close error separately from the stream's prior work error.
`arrow_stream_writer_close` keeps its strict requirement that the writer
was successfully finished.

Adoption order:

1. `c5aeddcf13`: abort API, focused public fixture and standalone gate.
2. `cc9bee1859`: failed-open owners do not repeat their creation error as cleanup.
3. This receipt and the clarification in `NORMAL_COMPONENTS.md`.

A NativeFile whose construction failed owns no descriptor. Abort consumes that
failed owner and releases its buffer with cleanup status 0; the opening error
remains available on the writer. A file that opened successfully has its actual
close status returned without normalization, including EINVAL or EIO.
This distinguishes the owner's opening state before closing it; it does not
convert errno 22 to zero.

The normal pipeline WIP now finishes and strictly closes successful output.
On a work error it aborts incomplete output, preserves that work error as
the main summary status, and records the separate abort/cleanup result.
Pool, stream and row cleanup still proceeds after an earlier cleanup failure.

| Slurm job | Purpose | Result |
| --- | --- | --- |
| 19052 | Empty staged script caused by stdin not being attached to kubectl | No qualification: exit 0 without gate marker, results or executable |
| 19055 | Public abort candidate and original Arrow native path | COMPLETED 0:0, 40 s |
| 19057 | Normal file/TCP component after explicit abort | COMPLETED 0:0, 53 s |
| 19060 | Exact git archive of initial c5a commit | COMPLETED 0:0, 33 s |
| 19062 | Failed-open control against c5a | Expected-failure collector 0:0, actual child 1, printed cleanup 2 |
| 19064 | Exact git archive of final cc9 commit | COMPLETED 0:0, 33 s |
| 19066 | Normal component using final cc9 stdlib source | COMPLETED 0:0, 53 s |
| 19068 | Source/receipt verification and durable custody | COMPLETED 0:0, 8 s |

The empty-script launch is retained as a harness failure. Subsequent staging
uses tar through Slurm, a fresh directory, a nonempty-script check and a required
gate marker. Its Slurm exit code is not counted as a successful test.

## Public API controls

The exact committed-tree gate emits `WAVE4_ARROW_ABORT_GATE_OK`.
Sounio checks, native compilation and execution pass for:

- Strict close of an unfinished writer returning 22.
- Abort before begin and after begin returning 0 without EOS.
- Finish followed by strict close returning 0 with EOS.
- Abort of a failed-open owner returning 0 with zero platform resources.
- A real file-close call followed by an injected delayed-I/O EIO result
  remaining visible as errno 5.

The C++23 interposer calls the real close first, verifies F_GETFD returns
EBADF for that descriptor, emits
`INJECTED_CLOSE_EIO_DESCRIPTOR_CLOSED`, then returns -1/EIO to the runtime.
It is explicit fault injection, not evidence of a physical storage failure.
The Sounio assertion requires that abort returns 5 while the platform resource
count is zero. The unfinished-file outputs have no EOS; successful strict
finish has the exact Arrow EOS bytes.

The existing Arrow native-stream and dictionary paths pass with path-only
relocation, verified by reversing the substitution and comparing the original
source bytes. Its optional externally supplied C++ fixture branch is absent
in this isolated gate; broader Arrow interoperability receipts remain separate.

The focused gate can be run under Slurm or the existing GitHub Actions
authority with MADAROS_RAW_BIN and ARROW_CPP_ROOT explicitly set:

`bash scripts/stdlib/wave4_arrow_abort_gate.sh /absolute/fresh/output`.

## Normal component before and after

The historical component jobs 19035/19040 closed all resources, but their
expected work refusals reported cleanup 22 because strict close also reported
that output lacked EOS. Those files and logs remain unchanged.

Jobs 19057 and 19066 require cleanup 0 for twelve malformed/budgeted CSV
controls and for the TCP deadline, byte-budget and truncated-field controls.
Observed main/cleanup pairs include 110/0, 105/0 and 22/0. They also inject
close EIO on the actual incomplete output file and require 22/5: the EOF/CSV
error remains primary, close failure remains visible, and resources are zero.
All refused outputs lack EOS.

The positive 257-row file/TCP outputs still pass the independent C++23/Arrow26
oracle and compare byte for byte. The underlying kernels and tolerances are
unchanged. These are direct serial-core component tests; the imported public
normal pipeline, WorkerFn IR clones, concurrent execution and serial/parallel
identity remain pending.

## Custody

Raw compiler SHA256:
`21b3b011c0d621041d12ba1661f756b55315eb78d0ce071ec1cad2f5e5ab26d2`.

Scratch prefix:
`/scratch/sounio-stdlib-wave3-20261027/services/`.

The archive retains the empty-script harness run, candidate/final committed
sources, expected failure, direct normal components, source/tool/input/
executable/artifact/SDK/dependency manifests and validation logs. Job 19068
rechecks the manifests, includes its submitted custody script and receipt
metadata, lists and syncs the archive, then verifies its hash by readback.

Archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/arrow-abort-export01/arrow-abort.tar.gz`.

SHA256:
`817ccf5421470141ec5a2a0a34eec81a9e99ff4b12358a8ef3d1d422d4033d21`.

Durable marker: `ARROW_ABORT_DURABLE_OK`.
