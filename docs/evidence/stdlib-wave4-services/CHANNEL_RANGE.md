# NativeChannelRangeProducer

This is a general bounded-channel protocol, not a Sounio callback executor.
The native thread executes only C++ queue logic. A private C++ shared_ptr lease
retains the already-created Channel; no Sounio owner, loan, pointer, closure or
runtime context crosses into the worker.

Public module: `sync::native_channel_range`.

`NativeChannelRangeConfig` contains start, count, step, deadline_ns and
max_bytes. For each index in [0,count), the producer admits start+index*step.
Count must be nonnegative; zero is valid. Step may be negative, zero or positive.
The complete interval is checked with signed __int128 before pthread creation.
If an endpoint would exceed i64, construction returns EOVERFLOW75. All
intermediate values are within the checked endpoints.

The queue remains bounded by its existing NativeChannel capacity. The producer
stores scalar control state and creates one joinable pthread. Its reservation
charges 1024 control bytes, a dedicated1MiB stack,4KiB guard and pthread handle.
The C++ receipt on this target reports1053704 bytes. Existing channel storage,
OS allocator/pthread bookkeeping and shared runtime pages are separate.
An insufficient budget is rejected before allocating producer control.

The constructor keeps a builder private in three checked ABI calls:

1. Reserve the full budget and retain the checked C++ Channel lease.
2. Copy and validate start/count/step as scalar words.
3. Copy the absolute monotonic deadline and create the pthread.

No POD address, managed record address or config pointer is reinterpreted or
retained. Mutex-protected phases reject duplicate configure/start. A retired
flag prevents a lookup obtained before pending-builder close from starting a
thread afterwards. All construction failures close the unpublished builder.

The deadline is a positive signed64 CLOCK_MONOTONIC nanosecond timestamp.
The worker checks it under the queue mutex and uses pthread_cond_clockwait with
that same absolute clock/timespec. Spurious wakeups and successful prefixes do
not restart it. INT64_MAX is supported. No relative chrono addition is used.

Cancel sets the producer's stop flag under the same queue mutex as admission
and wakes the native waiter. After cancel returns, this producer admits no new
values. Values already admitted remain in the channel. Cancellation affects
only this producer, not other leases.

Join succeeds once. Its PlatformValue.value is the admitted prefix length;
zero is a valid successful result. PlatformValue.status reports the work error
after a successful pthread join. Relevant errors include EPIPE32 on channel
close, EBADF9 on channel destroy, ETIMEDOUT110 and ECANCELED125. C++ exceptions
become EIO5; there is no claim to catch Sounio panic or faults. The worker never
calls Sounio.

Close cancels unfinished work, joins and releases the lease. It reports cleanup
errors independently of a work error already returned by join. Failed
constructors preserve their creation error and expose cleanup_status. A second
join returns EINVAL22. Raw close of an unjoined started producer returns EBUSY16.

The wait_count_code getter returns a nonnegative number of queue waits attempted,
or negative POSIX errno. A positive value proves that the actual native worker
entered the bounded-channel backpressure path. Polling this getter creates no
per-observation result record.

Closing/destroying the Sounio channel owner does not free storage under an
active producer. Channel close wakes the producer with EPIPE; destroy wakes it
with EBADF. The retained C++ lease is released after join/close.

Validation includes zero count, zero data, negative start, zero/negative step,
i64 endpoint overflow, maximum step, huge zero-step count without per-item
allocation, expired/maximum deadlines, absolute deadline after progress,
channel close/destroy, cancellation, once-only/concurrent join, failed
pthread_create with another producer alive, two independent leases and cleanup.
The implementation is exercised both directly by C++23 with ASan/UBSan and by
the public Sounio owner/config API.

The old massive/test_channel.sio source is preserved byte-for-byte as
`tests/stdlib/wave4-science-pipeline/legacy_io_channel.original.sio.in`, SHA256
`5c6c3499ecd641d7fd9cf348d0d94b7cbcd3089be03b86214e57164e119ed112`.
The migrated test retains all128 values, ordering, zero-result, backpressure,
timeouts, close/EOF, stale-token and budget assertions. It additionally checks
the producer's joined work status.

This does not implement generic Sounio IO/Alloc callbacks or TLS runtime
isolation. The compiler-protected thread/pool API migration is a separate gate.
The generic platform module provides explicitly unchecked raw thread entrypoints
at the FFI boundary; these are not replacements for safety acceptance. No old
IO/Alloc security assertion was reclassified as an unchecked ABI success.

Candidate run18907 passed the original finite-deadline controls. The subsequent
INT64_MAX audit reproduced signed chrono overflow on that exact runtime in
diagnostic18927 (actual child1; expected-failure collector0). Run18984 passed the
native absolute-clock correction, the same miniprogram, all prior C++ controls,
ASan/UBSan, and both Sounio check/compile/native programs. Exact committed-source
qualification and durable custody are recorded separately.
