# LOOM Subprocess Membrane V1

> Status: frozen Sounio semantics with an OCaml/C diagnostic realization.
> General Exec, commit, and CI attachment remain refused.

## Authority Split

Sounio action `9023` is the semantic authority. It owns the event algebra,
language roles, state transitions, decision codes, expected results, and four
causal sabotage controls. Its frozen identity is:

```text
manifest_sha256  0024178b8928f0c82d794d390244e83e5ce431054587fc7dd609c0f25c2e5b4f
semantics_sha256 43384e39d027c93bb46cb4a3636f1432123f5f4bef60d297d6b4485ccdbd4dd1
runtime_sha256   0470cc776e841cd57d98b988a0e1f7f402126b16289356ba49d93f71cd2b912a
```

`loom_membrane.ml` is the operational adapter. It validates the complete
freeze chain, constructs exact `9023` frames, invokes the hash-pinned Sounio
runtime for every decision, and writes hash-bound decision records.

`loom_membrane_stubs.c` and a hash-measured Bubblewrap are Linux x86_64
mechanisms, not policy oracles. Bubblewrap constructs the filesystem, network,
IPC, UTS, user, and PID namespaces. The C supervisor closes inherited
descriptors, stops tracees, extracts syscall evidence, calls the OCaml adapter
before the effect, resumes only an explicit Sounio `ALLOW`, and terminates the
process tree on refusal, policy error, or deadline expiry.

## Current Mechanism

The diagnostic probe creates a new process group and asks the root policy
before `fork`. It then executes the exact `/usr/bin/bwrap` image whose SHA-256
is recorded in every decision. The sandbox has a read-only `/`, an ephemeral
`/tmp`, a single read-write bind at the exact claim scope, a private network,
IPC, UTS, user, and PID namespace, no capabilities, and parent-death handling.

The C child closes every descriptor above standard input/output/error before
the Bubblewrap image can execute. It calls `PTRACE_TRACEME`, stops, and cannot
reach `execve` until the supervisor installs:

- `PTRACE_O_EXITKILL`;
- `PTRACE_O_TRACEFORK`, `PTRACE_O_TRACEVFORK`, and `PTRACE_O_TRACECLONE`;
- `PTRACE_O_TRACEEXEC` and `PTRACE_O_TRACEEXIT`;
- syscall-entry stops for the complete observed process tree.

Bootstrap filesystem effects made by the hash-measured Bubblewrap occur before
the user-executable transition. The C state machine recognizes the first
Bubblewrap `exec`, then requires an exact `exec` of the already admitted user
executable. Filesystem observation becomes mandatory at that boundary. A
sandbox process that exits without reaching it is a native error.

At user-command syscall entry the probe emits Sounio effects for:

- `execve` and `execveat`;
- `fork`, `vfork`, `clone`, and `clone3`;
- write-capable `open`, `openat`, `openat2`, and `creat`;
- the declared rename, link, unlink, directory, mode, and ownership mutation
  families represented by the native decoder.

Targets are read from tracee memory and resolved against the tracee's current
working directory or directory file descriptor before the Sounio decision.
Unresolved evidence becomes an explicit policy input and fails closed.
`io_uring_setup` is treated as unsupported rather than silently escaping the
membrane.

After the trace tree becomes empty, OCaml asks Sounio for `EXIT_RECORD`. On
deadline expiry it first kills the process group and every known tracee, reaps
them, then asks Sounio for `TIMEOUT_TERMINATE`. Root exit alone is never a
successful completion signal.

## Proved Diagnostic Behaviors

The native gate deliberately attempts each route and checks the external
effect, not only the printed decision:

| Route | Sounio result | External observation |
| --- | --- | --- |
| `/usr/bin/true` | `ALLOW` | exits 0 |
| in-scope mechanical write | `ALLOW` | exact bytes materialize |
| direct Python interpreter | `DENY 410` | sentinel absent |
| Python hidden behind `/bin/sh` | `DENY 410` | sentinel absent |
| executable named `rustc` | `DENY 410` | sentinel absent |
| out-of-scope write | `DENY 422` | target absent |
| filesystem observer disabled | kernel namespace refusal | target absent |
| inherited writable descriptor | pre-exec `close_range` | target remains empty |
| network namespace comparison | distinct namespace | host namespace is not reused |
| non-Sounio `.sio` write | `DENY 413` | target absent |
| out-of-scope path mutation | `DENY 422` | original target preserved |
| unsupported descriptor mutation | `DENY 415` | original target preserved |
| descendant surviving deadline | timeout termination | no late write after observation window |
| missing or altered policy | pre-spawn refusal | child effect absent |
| altered runtime | pre-spawn refusal | child effect absent |
| altered sandbox image | pre-spawn refusal | child effect absent |
| final outcome marked incomplete | `DENY 426` | child exit 0 is overridden |

Every operational decision record carries the frozen manifest, source,
semantics, runtime, and Bubblewrap hashes. The probe command requires
`SOUNIO_LOOM_HOOK_TEST_MODE=1` and always reports `attachment=refused`.

## Why Attachment Is Still Refused

Ptrace establishes a useful causal slice, but this implementation has not yet
proved the closed-world conditions required for arbitrary agent Bash/Exec:

- another thread can race pathname state inside the writable scope between
  measurement and kernel use;
- inherited writable file descriptors are not yet represented as effects;
- descriptor duplication, `mmap`, `memfd`, device, IPC, and network-mediated
  effects are not yet a closed Sounio algebra, even though the current probe
  closes inherited descriptors and isolates network and IPC namespaces;
- filesystem Unix sockets exposed by the read-only host root can still carry
  external effects; a network namespace does not isolate `AF_UNIX`, so a
  production root must mask host control sockets such as Docker and DBus;
- the inherited host `/proc` is read-only but not yet replaced by a procfs
  mounted for the child PID namespace in this pod;
- the syscall decoder is Linux x86_64-specific and has no architecture refusal
  wrapper at the production hook boundary;
- `io_uring` is detected, but the broader asynchronous-I/O surface has not been
  exhaustively closed;
- commit and CI admission are frozen semantic events, not attached controls;
- hostile same-UID isolation is not claimed.

Therefore all of these remain false:

```text
native_coverage_attested=false
exec_attached=false
write_attached=false
commit_attached=false
ci_attached=false
claim_ready=false
```

## Attachment Path

The production membrane should combine complementary kernel controls rather
than stretch ptrace into a universal reference monitor:

1. retain Sounio `9023` as the only decision authority;
2. retain the hash-measured Bubblewrap launcher that establishes the private
   namespace boundary and deny-by-default filesystem view;
3. use seccomp user notification for the syscall classes that require a
   per-effect Sounio decision;
4. use Landlock or an equivalent kernel-enforced path envelope as a monotonic
   backstop against pathname races;
5. close or explicitly inventory inherited descriptors before root admission;
6. make supervisor loss revoke the whole execution generation;
7. run a sabotage gate for every newly claimed coverage family;
8. attach Bash/Exec only after the coverage receipt is true, then serialize
   commit and CI attachment behind quiescent-outcome receipts.

The kernel mechanism may change. The frozen Sounio semantics and causal tests
remain the acceptance authority.
