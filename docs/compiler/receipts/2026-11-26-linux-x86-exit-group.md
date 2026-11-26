# Linux x86-64 thread-group termination regression (2026-11-26)

## Defect and scope

**OBSERVED (Granite coordinator):** a pinned native audit returned from main
while a CUDA driver thread remained in poll; the thread-group leader was Z
and Slurm never collected program exit. Numerical output was complete.

**OBSERVED (source):** Linux x86-64 syscall 60 is thread-local exit, not
process termination. `emit_entry_trampoline_call_exit_into`, the alternate
entry trampoline (including capability failure), the heap-reserve failure
and `nc_emit_exit_code` emitted 60. The imported `native::encode::emit_exit`
helper did likewise. Replace these with 231 (`exit_group`), preserving the
status argument and existing profile-dump ordering. The two post-fork
`execve`-failure `_exit(127)` paths deliberately retain syscall 60: they
execute in the child only. The shared encoder byte test now expects `e7`
instead of `3c`. Other target architectures are unchanged.

## Discriminating regression

`tests/native-v2/exit_group_thread_return.sio` loads a C++23/POSIX DSO via
`dlopen` and `dlsym`. The DSO creates a pthread, waits for its readiness
byte on a pipe, and leaves the worker blocked on a distinct pipe whose
write end remains open. Sounio prints `THREAD_READY_RETURN_37` and returns
37 from main. `scripts/ci/madaros_exit_group_thread_gate.sh` requires that
marker and process status 37 under a 5-second timeout (with a 2-second
forced-termination grace). A timeout or any other code fails the gate.

Base origin/main: `ad5beda241dc61f2abfc5ba09355c8d74401fa13`.
Before source (test-only): `648d25bfdb959fe7407e81fb99b7d0f75bdd3789`.
Compiler fix: `bacbd75b256b16515f3e2d53787f984ed6be942c`.
After source and corrected test harness:
`46f3c22e84c42a210c237c344446213e641605d9`. Before run used the corrected
harness from this commit, overlaid without altering compiler sources.

Both runs built Madaros from their respective source checkouts with
`scripts/ci/build_modular_madaros.sh`; both checked and compiled the same
Sounio regression, using the same C++ helper source and C++23 compiler.
All jobs used Slurm, D (`dl380-dl380-proxmox`), CPU-only, 12G,
`--comment=lane:llm-granite`. No node state changes.

| Job | Source | Batch state | Elapsed | Batch MaxRSS | Program status |
|---|---|---|---|---|---|
| 29563 | Before, initial harness | FAILED 1:0 | 00:02:00 | 2209536K | 126 (missing execute bit, not a valid defect test) |
| 29564 | After, initial harness | CANCELLED | 00:00:00 | unavailable | not run |
| 29566 | Before, corrected harness | COMPLETED 0:0 | 00:02:17 | 2475884K | 124 (timeout, expected failure reproduced) |
| 29567 | After, corrected harness | COMPLETED 0:0 | 00:01:48 | 2194760K | 37 (PASS) |

The before batch succeeds only because it asserts the expected regression
failure, not because the program exited correctly. The initial harness
failure was fixed by setting the generated ELF executable before running
it. Both meaningful runs recorded `THREAD_READY_RETURN_37`.

| SHA256 artifact | Before 29566 | After 29567 |
|---|---|---|
| source-built Madaros | `82e085c09464e7446d6345a14324802ea68bf8c66a3988d4d3f23bda9da1e1d1` | `7b4a7e53da0e13fc2c0459eb086faaad33343d4268e9d4f335fbdd06f1ff0628` |
| regression executable | `6919e16b848cad4e4a397db95f30dad4e5fb64fab01021eed5fe27f050f56f3d` | `46992b58b4f926f19b527d711b845ac27eb0c3d028d73f054530dacde2ce3804` |
| helper DSO | `88028ef48dd4b065c996b1ecd2621cc64e67340db1e2851dbc901d925373428d` | `6bc6f17188bc25a90f9bb891af22a40776300bdbb22c3599cb5b68b308d16ace` |

The DSO digests differ between builds; the same helper source and compiler
were used, and the readiness marker in both runs establishes the blocked
thread precondition. Evidence logs and manifests are retained under
`/orangefs/training/llm-granite-n10/exit-group-29566/` and `exit-group-29567/`.

**OBSERVED:** unpatched program status 124 after timeout, patched program
status 37 before timeout, `EXIT_GROUP_THREAD_RETURN_PASS`, independently
recorded in Slurm accounting. The patched Madaros build completed from
source. This establishes the thread-group return regression on Linux x86-64.
**NOT ESTABLISHED:** an actual CUDA run with the patched compiler, the exact
cause of the earlier Granite tail timeout, or cross-GPU numerical identity.
The Granite numerical audit pin is intentionally unchanged; a separate
watchdog handles its historical compiler's process-exit hang.
