# Native scalar workers and the ADC CSV/TCP → Arrow pipeline

This tranche implements a real bounded pthread worker pool and an exact
discrete-signal pipeline. It is qualified on Linux x86-64 with the source-built
Madaros raw named-ref-v2 artifact. It does not close the whole-stdlib, final
compiler, generic owner transport or fixed-point gates.

## Contracts

NativeThreadPool has fixed workers and fixed admitted-task capacity. Queued,
running and completed/uncollected tasks all occupy one slot. Results are taken
by monotonic task ID, exactly once; value zero is a successful result when
status is zero. Zero-time submit/result is nonblocking (EAGAIN). Positive
timeouts do not consume a result or admit a failed submission. Shutdown drains
or cancels queued tasks; running callbacks finish. Close joins workers before
destroying storage.

The reservation charges fixed control/slots, one 1 MiB stack and 4 KiB guard
per worker, and pthread handles. It excludes OS allocator bookkeeping and
shared runtime pages. The pool accepts 1..64 workers and 1..1048576 slots.

The qualified scientific callback is the private ADC leaf x*x. Inputs are
strict incremental CSV, header sample_id,amplitude_adc, consecutive IDs from
zero, amplitude in [-1000000,1000000], at most 1000000 samples. Values are
ADC counts and ADC², not a calibrated SI energy. C++23 checks that the worst
allowed sum fits i64, and independently verifies every emitted value using
__int128 products. No Python, Rust or Julia numerical helper is used.

The producer retains every CSV/Arrow owner. Workers receive scalar amplitudes
only. A min(capacity,16) window supplies backpressure; collection by ID preserves
source order. Arrow retains the existing 16-row batch limit. Output schema is
three non-null int64 columns: sample_id, amplitude_adc, squared_amplitude.

Both regular files and an owned TCP duplicate use the same BufferedReader and
CsvStream parser. native_file_from_tcp borrows a socket and duplicates its fd
with CLOEXEC; it does not change O_NONBLOCK. The reader owns its duplicate
independently: closing the socket owner cannot invalidate it. An exclusive
runtime read lease makes original reads/read-shutdown and a second reader
return EBUSY. Original writes and close remain valid. Reads use poll plus
MSG_DONTWAIT, retry EAGAIN against the original absolute monotonic deadline,
and respect an explicit total byte budget. At the exact budget, MSG_PEEK
distinguishes EOF from excess input without consuming the excess byte.

The additive *_into, *_count and *_code interfaces reuse caller-owned storage.
Nonnegative count/code APIs return negative POSIX errno for I/O, or negative
ByteBuffer storage status for the ByteBuffer-only primitives. Legacy result
APIs remain available. UTF8 validation has one shared scalar implementation.

CsvStream now rejects a delimiter beyond an explicit/learned column count
with status22 before allocating an extra field. The old wrong-width fixture
returned105 when its field budget was reached first. Unspecified-width
max_fields exhaustion still returns105, as covered by the unchanged original
CSV test. Checked integer conversion covers the exact i64 endpoints and rejects
overflow, fractions, exponents, sign-only values and whitespace.

ADC result status is the primary parser/config/POSIX or Arrow status.
cleanup_status also reports Arrow close protocol errors: unfinished output can
return22 while all runtime resource owners have been closed. Error output is
not finished with a success EOS. The probes explicitly require zero runtime
resources before platform_close, then return the primary errno exactly.
Public positive assertions are retained.

## Current callback boundary

The fn(i64)->i64 type alone does not establish parallel safety for every pure
Sounio function. The current compiler uses a shared RIP-relative runtime context,
including non-atomic managed allocation and region state. Calls can introduce
region operations even without explicit Alloc effects. The ADC x*x callback is
an IR leaf: its region-enter token is lowered to zero and the scalar region-exit
tests that token before accessing the context. That exact profile is qualified.

General callbacks that construct aggregates or enter regions, managed allocation
on workers, captured environments and owner/loan transport across threads are
not qualified by this receipt. A compiler-derived callback admission capability
or thread-local runtime support remains an integration requirement for a general
callback-safety claim. The runtime's dladdr check is not such a proof. Sounio
panic/process faults are not caught by the C++ exception handler.

## Execution evidence

All builds and tests below ran in Slurm.

| Job | Verdict and scope |
|---|---|
|18723|COMPLETED 0:0: initial C++ pool oracle and Sounio pool/ADC check+compile.|
|18727|Outer job COMPLETED 0:0, but native phases failed126 (missing chmod) and Arrow C++ build failed from SDK headers passed with -I under -Werror. This is not a passing native receipt.|
|18731|COMPLETED 0:0: corrected harness, Sounio pool native0 and 257 exact Arrow rows/17 batches; sum 99534467929025.|
|18741|FAILED 124:0: original million-row path exhausted4,194,304 managed handles. The old fatal path left worker threads until the 600s timeout. Source, inputs and failure output are retained.|
|18768|FAILED 1:0: my initial pool/PollSet union missed one closing dispatch brace; corrected before the next source snapshot.|
|18774|FAILED 1:0: E040/E039 on owned-file early return; restructured into an if/else wrapper that consumes the owner once in both branches. C++ normal/sanitized TCP reader controls had passed.|
|18775|FAILED 1:0: million-row file native+oracle0, then the negative harness expected132 although this raw returns1 for assert failure. Gate-only probes now close resources and return the exact primary errno; positive assertions remain.|
|18777|FAILED 1:0: extra-column fixture returned105 instead of22. The same fixture passed after the delimiter-width fix.|
|18778|COMPLETED 0:0: full file/TCP million-row gate, malformed inputs, budgets, cleanup, private fields, linear close, closure refusal, original CSV/Unicode and original socket deadlines.|
|18780|FAILED 1:0: independent oracle audit proved that Arrow C++ accepted a stream without its final 8-byte EOS (actual oracle exit0). This was a verifier defect, not a producer failure.|
|18781|COMPLETED 0:0,3m42: final full component gate. The verifier now requires explicit 8-byte EOS and exact consumption, rejects removed EOS and duplicated EOS, alongside the original truncation/trailing/wrong-value/wrong-order controls.|
|18785|FAILED 1:0: the first same-raw audit accidentally selected the self-contained3-row public caller, not the original scale caller. It is not million-row causality evidence.|
|18788|COMPLETED 0:0 expected-failure control: exact original scale caller/library and identical million-row input, recompiled with the same 21b3 raw used after the fix, emitted managed-handle exhaustion. The supervisor then terminated only its own isolated process group because the original fatal path leaves workers alive; actual supervised exit 143. No smaller workload or larger deadline was substituted.|

Final18781 results:

- File and TCP each produced 1000000 rows / 62500 batches, all checked by Arrow C++ 26.
- Exact sum of squares: 378114110446473770.
- Native producer max RSS from GNU time: small 3192 KiB, file 3152 KiB, TCP 3948 KiB.
- File wall time 34.35s; TCP 35.87s. These are per-run observations, not a dedicated
  comparative benchmark. Slurm batch RSS includes compilation and must not be
  confused with the per-native GNU time measurements.
- C++ normal and ASan/UBSan controls cover actual two-worker rendezvous, 10000
  tasks, value zero, full slots, timeouts, cancellation, once-only collection,
  failed partial pthread_create joined completely, invalid arguments and fd count.
- TCP controls cover exclusive lease, original close, short reads, absolute
  deadline 110, exact budget+EOF, excess input 105 without consumption, truncated
  quoted CSV 22 and a slow one-byte peer. The peer's small send buffer exercises
  backpressure without a producer-side whole-input spool.
- Source, raw compiler, C++ executables/tools, dynamic dependencies, SDK headers,
  derived inputs and artifacts all passed their declared before/after guards.
- The final tracked gate tightens the capturing diagnostic matcher to an anchored
  error line; the archive supplement validates it on the same real error and
  rejects a filename-only spoof. Native code and scientific inputs are unchanged.

Raw compiler:
  /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/named-ref-v2/results/madaros
  SHA256 21b3b011c0d621041d12ba1661f756b55315eb78d0ce071ec1cad2f5e5ab26d2

This is a reused source-built artifact, not a rebuild of the current integration
HEAD and not a claim that its broader historical worker passed. The named-ref-v2
worker retained an outer failure; the focused raw/source guard supplement and
this component receipt are distinct.

Final gate:
  /scratch/sounio-stdlib-wave3-20261027/services/native-pool-45
Same-raw original failure:
  /scratch/sounio-stdlib-wave3-20261027/services/adc-handles-before-same-raw-02
EOS failing-before:
  /scratch/sounio-stdlib-wave3-20261027/services/adc-eos-before-01

## Integration

Own branch: devpass/stdlib-wave3-services-20261027.

83778fc7b9 is the initial implementation. The subsequent 54be2fc9ea and ffe693e88c
are own-branch adoptions of root's existing e6a78eafba and c1b92bf7d6; do not adopt
those duplicate runtime commits onto root again. The final feature delta includes
the corrected dispatch brace, TCP lease reader and scalar/reuse APIs. During
integration retain both Kind::pollset and Kind::pool, every preexisting opcode,
and exactly one closing brace for the pool dispatch block.

The full stdlib corpus, final coherent compiler/runtime replay, general callback
admission, official CI, push, PR and main merge are separate gates and are not
claimed here. The stiff ODE model/assertions were not changed.

The durable archive path, checksum and export job are recorded by the follow-up
custody receipt once read back successfully.
