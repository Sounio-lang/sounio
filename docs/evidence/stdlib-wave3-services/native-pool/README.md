# Bounded native workers and exact ADC stream

The pool is an actual pthread worker pool. Its capacity bounds admitted tasks,
including queued, running and completed/uncollected tasks. Workers never block
waiting to publish results into a second unbounded queue. The callback profile
is a terminating plain fn(i64)->i64; integers carry scalar data, not owners,
loans, pointers or closure environments. Sounio panic/process faults are not
caught by the C++ exception handler.

The reservation includes one 1 MiB stack and 4 KiB guard per worker, pthread
handles, fixed control allowance and task slots. OS allocator bookkeeping and
shared runtime pages are excluded. Worker count is 1..64; capacity 1..1048576.
Zero-time submit/result are nonblocking (EAGAIN); positive timeout is bounded.
An unsuccessful submit admits nothing; a result timeout consumes nothing.
IDs increase monotonically and collection is once-only. Shutdown drains or
cancels queued tasks; it cannot interrupt a running callback. Close cancels,
joins, discards remaining scalar results and destroys the resource.

The ADC pipeline accepts strict incremental CSV with header
sample_id,amplitude_adc and consecutive IDs from zero. Amplitudes are integers
in [-1000000,1000000], at most 1000000 rows. The Sounio callback computes an
exact i64 square. The total sum fits i64. These are ADC-count squared values,
not a calibrated SI energy estimate. CSV/Arrow owners remain in the producer;
the admitted window is min(capacity,16). Results are collected by ID and written
in source order. Arrow batches retain the existing 16-row limit.

The public self-contained fixtures create their own files. Gate-only .sio.in
templates receive independently generated CSV fixtures in private run
directories. The checked CSV integer accessor accepts the complete i64 domain;
it rejects overflow, fractions, exponents, sign-only and whitespace values.

## Evidence and remaining gates

- Slurm 18723: C++23 worker oracle and both initial Sounio check/compile phases
  passed. C++ controls include an actual two-worker rendezvous, zero-valued
  result, bounded outstanding work, timeout, cancellation, once-only collection,
  10000 tasks, partial pthread_create failure and complete join/destruction.
- Slurm 18727: diagnostic harness failed to chmod the emitted ELF and treated
  third-party Arrow SDK headers as -I under -Werror. Preserved as a failed
  attempt; no native success is claimed from it.
- Slurm 18731: replay with explicit chmod and SDK -isystem passed. Sounio pool
  execution passed. Native CSV-to-Arrow produced 257 rows in 17 batches;
  independent Arrow C++ verified every integer and order, with sum of squares
  99534467929025. Runtime resource count was zero before platform_close.
- Slurm 18741: full guarded gate submitted; its result must be read before
  claiming the million-row, sanitizer, mutated-Arrow and negative-input gates.
  No final result was available when this initial implementation was committed.

Source-built raw compiler used by these focused jobs:
  /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/enum-integrated/results/madaros
  SHA256 c23c64941858c5bb631f61f3e0a925306458caca3612abe4fce10b8bb6bf8c08

Receipts:
  /scratch/sounio-stdlib-wave3-20261027/services/native-pool-37
  /scratch/sounio-stdlib-wave3-20261027/services/native-pool-38
  /scratch/sounio-stdlib-wave3-20261027/services/native-pool-39

The runtime in this first tranche predates root's PollSet/op110..112 additions.
Integration must retain those additions and rerun with one coherent compiler
and runtime. TCP inlet, whole-stdlib acceptance, official CI, PR and merge are
not claimed by the focused file-pipeline receipt.

