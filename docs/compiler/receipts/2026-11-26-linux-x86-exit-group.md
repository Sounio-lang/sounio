<!-- docs:meta
topic_id: repo.docs.compiler.receipts.2026-11-26-linux-x86-exit-group
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.receipts.2026-11-26-linux-x86-exit-group
-->

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

## Historical recurrence and CI enforcement

This re-lands the process-termination correction of commit
`168d9f9585aa3d9fa4e1b13f41ed9fe5716b7bf6` ([#292](https://github.com/Sounio-lang/sounio/pull/292),
2026-06-14, “use exit_group(231) for program exit, not exit(60)”), adapted
from the earlier lean_single epilogue to the current native x86-64 Madaros
backend. Its commit message records the same NVIDIA L4 sm_89 PASS-then-hang
symptom (timeout 124 before, clean exit 0 after rebuilding). Neither that
commit nor its patch-identical predecessor `87df161522af1e9f59aef5b95634b6a870b7d985`
is an ancestor of `origin/main` as checked on 2026-11-26. The local historical
remote-tracking refs `origin/regen/souc-bin-exit-group` and
`origin/fix/gpu-exit-group` retain the original and patch-identical version,
respectively. The historical GPU result is a claim in that commit message,
not a newly repeated GPU test.

The first PR CI Decision failed in Fast Checks because the new regression
script was not directly named by a workflow (unnamed gate count 456 → 457).
The regression is now an executable step in `.github/workflows/ci.yml`'s
Madaros Witness Gate, immediately following placement of the verified
current-source compiler at `/tmp/madaros-ci.elf`. The step passes
`MADAROS_RAW_BIN` to the gate, builds the C++23 pthread DSO with `g++`,
checks and compiles the Sounio probe with that exact compiler, and enforces
the same five-second exit-status test. No frozen ratchet baseline is
relaxed. CI execution on the updated head is tracked separately from the
observed Slurm results above.

CI run [45898638135](https://github.com/Sounio-lang/sounio/actions/runs/45898638135)
on signed-off merge `5aa6c3b0ca5073e64b13f789cd6d4b65bf4a7899` **passed**
the Fast Checks workflow-reference ratchet. Impact then refused the missing
release-note fragment, so the current-source compiler and witness jobs were
skipped in that run. Added the user-visible `linux-x86-exit-group.fixed.md`
fragment; no CI policy or gate is bypassed.

## CI byte-witness and docs-registry follow-up

CI run [45898928768](https://github.com/Sounio-lang/sounio/actions/runs/45898928768)
**OBSERVED** the newly wired pthread gate returning 37 and passing, and the
workflow-reference and changelog gates passing. The subsequent Madaros
self-test failed T168 (1171/1172), because its pinned slow-allocation trap
still expected syscall 60 rather than the changed process exit. Contracts
Core (core-b) also refused this receipt's missing registry metadata.

The tracked source/test tree (`self-hosted/compiler`, `self-hosted/native`,
`tests`) was searched for hexadecimal `0x3c`, decimal syscall 60 in byte
assertions, and literal x86 `mov rax/eax, 60` byte sequences. The affected
witnesses and their disposition are:

| File / witness | Correction |
|---|---|
| `self-hosted/compiler/main.sio`, `compiler_main_test_native_v2_alloc_slow_path_trap` (T168) | `seq[3] = 0xE7`, exit_group(231); trap status 181 and remaining bytes unchanged. |
| `self-hosted/native/suite.sio`, `native_test_emit_exit_bytes` (T01) | Expected sequence and asserted immediate changed from 60 to 231. |
| `self-hosted/native/test_encode.sio`, `te_test_emit_exit_bytes` (T01) | Already changed in the original fix; verified the 231 expectation. |
| `tests/run-pass/sprint235_print_f64_e2e.sio`, `emit_main` | Independently hand-emitted Linux process-exit fixture now uses 231, keeping its other instructions unchanged. |

Other matches are unrelated byte values (e.g. `cmp` ModR/M encodings,
floating-point constants, bit masks, font and hash data), or independent
compiler emitters rather than pinned witnesses of this Madaros x86 exit
path. The post-fork child-only exit calls remain unchanged. No numerical
or trap status change was introduced by updating the witnesses.

This receipt is registered as
`repo.docs.compiler.receipts.2026-11-26-linux-x86-exit-group` with
`repo_only` authority and `contributors` audience; its matching effective
registry entry is derived by the checker. CI verification of the
follow-up is pending at this entry; the earlier pthread Slurm observations
and the successful CI pthread step are unchanged.

CI run [45910557660](https://github.com/Sounio-lang/sounio/actions/runs/45910557660)
on `16dbb598ba44c3c7dca03548403dc9a3391529a2` **OBSERVED** Docs registry
PASS and pthread regression PASS. A later Generated snapshot guard refused
editing the checked-in `docs/governance/topic-registry.v1.json` in this
ordinary compiler PR. Per that enforced rule, the snapshot is restored to
`origin/main`; the metadata above remains, and the effective registry entry
is rebuilt from it and the path classifier by the actual Docs registry
checker. The checked-in snapshot is updated only by a separate
`snapshot/refresh-*` PR after merge. No guard exemption or baseline change.

The Madaros full self-test step on the same run also completed successfully; its gate requires the source-derived denominator and zero failed self-tests (including T168).
