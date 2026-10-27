# stdlib/async

The async root exports the cooperative typed polling API from
`async::poll_executor`: PollExecutor, PollStep, PollTask, spawn/tick,
cancellation, result queries and explicit terminal-task reaping.

A task consists of a plain typed callback and owned i64 state. Callbacks
return promptly; a pending step may specify an absolute monotonic wake
time. Tick polls at most its caller-supplied budget and does not sleep or
perform blocking network I/O on its own. Use socket tokens and readiness
operations in callbacks that need I/O.

Executor payload is six i64 words/task. Construction and growth follow the
explicit byte budget. Cancel marks a pending task terminal. Reap refuses a
pending task, returns a terminal result once, compacts task storage and
makes room for later tasks without reusing their monotonically increasing
IDs. Reaped IDs remain unavailable. Callback state that represents an
external resource ID remains the caller's cleanup responsibility.

PollTask status 0 is pending,1 ready,2 failed,3 canceled and4 absent.
Reaping pending work returns22. Close consumes the executor explicitly.

Older `async::future`, `async::task` and `async::executor` modules retain
their direct paths. They are not aliases for the polling executor.
Thread/channel primitives live in their own modules.

See [the root-import smoke fixture](../../tests/stdlib/massive/test_service_root_exports.sio)
and [the focused receipts](../../docs/evidence/stdlib-wave3-services/README.md).
