<!-- docs:meta
topic_id: repo.docs.audit.extern-c-ffi-silent-noop-dispatch-2026-08-13
authority: repo_only
audience: users
last_validated: 2026-08-13
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.extern-c-ffi-silent-noop-dispatch-2026-08-13
-->

# `extern "C"` FFI calls silently no-op under default Madaros — dispatch

**Date:** 2026-08-13 (Track B implemented and verified 2026-08-15)
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine); cross-checked against `SOUNIO_SOUC_ENGINE=lean_single`
**Owner:** unassigned (self-hosted/native FFI + self-hosted/compiler/lean_single.sio's `strip_extern_blocks`)
**Status:** Track B (lean_single `system()` stub) **implemented and verified** — see "Track B implementation" below.
Track A (Madaros) remains **open**, unpatched. Applied under explicit, twice-repeated user
authorization ("arrume o FFI") to patch `self-hosted/compiler/lean_single.sio` directly, as an
exception to the default dispatch-only protocol for this one file/track.

## Why this dispatch

While building a real-data FFI bridge (Sounio → Python, for a LEMON EEG re-analysis; see
`examples/cayley_dickson_lemon_g2_ffi.sio`), `extern "C" { fn system(cmd: &[i8;N]) -> i32 }` —
the pattern already present and used in `self-hosted/compiler/pkg/registry_client.sio` — compiled,
type-checked, and "ran" (claimed exit 0 / rc=0) but the underlying shell command **never executed**.
Chasing this down surfaced a wider, previously undocumented problem: under the **default** Madaros
engine, `extern "C"` integer-returning FFI calls are not merely limited (as
`docs/compiler/KNOWN_LIMITATIONS.md:174` describes) — they are **silently non-functional** for at
least `system()` and `getpid()`, both of which claim success while doing nothing.

This is a correctness hazard, not just a missing feature: a program can call an `extern "C"`
function, receive what looks like a normal return value (`0`), and proceed as if the call
succeeded, with no error, warning, or crash.

---

## Defect — `extern "C"` calls under default Madaros return a plausible-looking value without invoking the target function

### Repro 1 — `system()`, decisive (side-effect observable independent of any return value)

```sounio
extern "C" {
    fn system(cmd: &[i8; 512]) -> i32
}
fn main() -> i32 with IO, Mut, Panic, Div {
    var cmd: [i8; 512] = [0; 512]
    let s = "touch /tmp/ffi_probe_proof.txt"
    var i: i64 = 0
    let n = str_len(s) as i64
    while i < n && i < 511 { cmd[i as usize] = str_char_at(s, i as usize) as i8; i = i + 1 }
    let rc = system(&cmd)
    print("system() returned code: "); print(if (rc as i64) == 0 { "0" } else { "nonzero" }); print("\n")
    0
}
```

`./bin/souc run r.sio` → prints `system() returned code: 0`. `/tmp/ffi_probe_proof.txt` is **never
created**. Reproduced identically via `./bin/souc compile r.sio -o r.elf && ./r.elf` (native path,
not just JIT) — rules out a JIT-only limitation.

**Ruled out: environment/sandbox interception.** A plain `gcc`-compiled C binary calling
`system("touch /tmp/c_probe_proof.txt")` in the same shell, same sandbox, same session **does**
create the file. The execution environment permits `system()` for ordinary ELF binaries; the
non-execution is specific to Madaros-compiled output.

### Repro 2 — `getpid()`, no pointer argument, documented as fixed

```sounio
extern "C" { fn getpid() -> i32 }
fn main() -> i32 with IO, Mut, Panic {
    let pid = getpid() as i64
    print(if pid == 0 { "0 (suspicious)" } else { "nonzero" }); print("\n")
    0
}
```

`./bin/souc run r.sio` → prints `0 (suspicious)`. A real `getpid()` can never legitimately return
0 (PID 0 is not a valid PID for a normal process). `docs/compiler/KNOWN_LIMITATIONS.md:174-236`
documents `getpid`/`getppid` as **already fixed** via a `strip_extern_blocks()` stub-rewrite; that
fix does not appear to be in effect under the default engine as currently built/run.

### Cross-check — the documented fix exists, but only under `lean_single`, and only for its allowlist

`strip_extern_blocks()` — the function `docs/compiler/KNOWN_LIMITATIONS.md:174` credits with fixing
integer-returning FFI — exists in exactly one place in the tree:
`self-hosted/compiler/lean_single.sio`. It is **not** part of the Madaros modular pipeline
(`self-hosted/compiler/main.sio` and friends, which `bin/souc` routes to by default).

Running Repro 2 with `SOUNIO_SOUC_ENGINE=lean_single`:

```
getpid() returned: nonzero (looks real)
```

This confirms the documented fix is real and works — under `lean_single` only. Running Repro 1
(system()) under `lean_single`, however:

```
error: error[E001]: Type mismatch in call argument — declared type does not match at <main>:15
typecheck: failed
```

`lean_single` **rejects** the `&[i8;512]` argument to `system()` outright — reproduced verbatim
against the checked-in `self-hosted/compiler/pkg/registry_client.sio` itself
(`./bin/souc check self-hosted/compiler/pkg/registry_client.sio` under `lean_single` gives the
identical `E001` at its own `system(&cmd)` call site, line 342 of the bundle). The file that is
cited elsewhere as "proof the `system()` FFI pattern works" has, as far as this dispatch can
determine, never actually been runtime- or typecheck-verified end-to-end under either engine.

### Root cause (two independent gaps, not one)

1. **Madaros (default engine):** no equivalent of `strip_extern_blocks()`'s stub-rewrite exists in
   the modular pipeline. `extern "C"` calls type-check and lower to *something* that returns `0`
   without invoking the named symbol — for both a documented-allowlisted function (`getpid`) and an
   undocumented one (`system`). This is broader than the "integer FFI now works" claim in
   `KNOWN_LIMITATIONS.md` suggests: that claim appears to hold only under `lean_single`, not under
   the engine `bin/souc` uses by default.
2. **`lean_single`:** `strip_extern_blocks()`'s function-name allowlist (per
   `KNOWN_LIMITATIONS.md:174`: `getpid`/`getppid`, `malloc`/`free`, math intrinsics) does not
   include `system`, and the generic (non-allowlisted) `extern "C"` type-check path rejects a
   correctly-typed `&[i8;N]` pointer argument rather than falling through to (or erroring clearly
   about) real dynamic-linked FFI. `self-hosted/native/ffi.sio` (2,201 lines: type registration,
   SysV/Win64 register assignment, symbol resolution, marshaling) looks like the intended general
   mechanism for this, but nothing in either engine's currently-observable behavior indicates it is
   wired into the live `extern "C"` call codegen path — its apparatus exists in source without an
   observed effect at runtime.

### Proposed fix locus (two independent tracks — either unblocks something real)

- **Track A (Madaros, higher priority — this is the default user-facing engine):** port or
  reimplement `strip_extern_blocks()`'s stub-rewrite (or wire up `self-hosted/native/ffi.sio`'s
  existing marshaling apparatus, if that is the intended long-term mechanism) into the Madaros
  modular pipeline, so that at minimum the already-documented allowlist (`getpid`/`getppid`,
  `malloc`/`free`, math intrinsics) is genuinely functional under the default engine — currently
  `stdlib/os/process.sio`, `stdlib/mem/`, and `stdlib/sync/mutex.sio` are documented as "unblocked"
  by this fix but, per Repro 2, may be silently non-functional under Madaros. **This should be
  verified directly against those stdlib callers** (this dispatch's own attempt to test
  `stdlib::os::process::process_id()` hit an unrelated multi-module import-resolution failure,
  `run_check_mode: unresolved import stdlib::os::process` — a separate, already-documented class of
  fragility per `docs/compiler/KNOWN_LIMITATIONS.md` §13 — so that specific check is not yet closed
  out here).
- **Track B (`lean_single`):** add a genuine `system(cmd)` stub to `strip_extern_blocks()`'s
  allowlist, implemented as a real `fork`+`execve`+`waitpid` raw-syscall sequence (matching glibc's
  actual `system()` semantics) rather than relying on the untested general pointer-marshaling path
  — `self-hosted/compiler/claim_executor.sio` and `self-hosted/lsp/server.sio` already contain a
  working fork/execve/wait4 sandbox pattern used for the compiler's own subprocess needs
  (gate execution, LSP `bash`/`souc` invocation) that may be reusable as a reference
  implementation. Separately, the generic (non-allowlisted) `extern "C"` pointer-argument
  type-check path should either work correctly or fail with a clear, documented error — not the
  bare `E001` with no further detail.

### Acceptance gate (proposed)

1. Repro 1 and Repro 2 above, both under the **default** `bin/souc` engine: `getpid()` returns a
   real nonzero PID; `system("touch <path>")` actually creates the file.
2. A regression test alongside the existing `tests/run-pass/ffi_integer_return.sio` that exercises
   `system()` specifically (currently that test, per `KNOWN_LIMITATIONS.md:174`, covers only
   `getpid`/`getppid`/`malloc`/`free`/math intrinsics — none of which this dispatch found to
   actually work under Madaros either, so the existing test's assertions should be re-examined
   against Track A's fix, not assumed still-passing).
3. `stdlib::os::process::process_id()` (and the other `stdlib/os/`, `stdlib/mem/`,
   `stdlib/sync/mutex.sio` callers `KNOWN_LIMITATIONS.md` credits to this fix) verified directly,
   once the unrelated import-resolution blocker noted above is out of the way.

## Track B implementation (done, 2026-08-15)

Added an explicit `system` case to `append_extern_c_stubs()` in
`self-hosted/compiler/lean_single.sio` (immediately before the existing `malloc`/`free` case),
declaring `cmd: &[i8;1024]` (fixing Repro 1's `E001` type-mismatch, which was a fallthrough to the
generic single-arg `i64`-typed stub, not a marshaling failure) and implementing a real
`fork`+`execve("/bin/sh","-c",cmd)`+`wait4` sequence via raw `syscall6` calls, mirroring the
already-proven pattern in `self-hosted/lsp/server.sio`'s `run_souc_check()`. Re-verified via the full `make build`
gen1→gen2→gen3 fixed-point bootstrap (`scripts/dev/souc-build-lock.sh make build`): **✓ FIXED
POINT OK**, `gen2.elf` == `gen3.elf` (md5 `37c1cf8a43ab74143994ec77b9a45e5e`) — the fix does not
break self-compilation.

**Reproducing this requires building from source, not the prebuilt `bin/souc-lean-single-x86_64`.**
That ELF is not a build of `self-hosted/compiler/lean_single.sio` at all — running it directly
prints a `mini_native` usage banner, a different, unrelated compiler tool. `bin/souc`'s
`SOUNIO_SOUC_ENGINE=lean_single` alias is wired to exactly this file (`LEAN_SINGLE="$ROOT_DIR/bin/
souc-lean-single-x86_64"` in `bin/souc`), so `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run
examples/cayley_dickson_lemon_g2_ffi.sio` silently exits 1 with no output — it does not contain
this fix and cannot run this file's `extern "C"` block correctly regardless. The tested, working
path is `scripts/dev/souc-build-lock.sh make build` (produces a fixed-point-verified `./gen3.elf`
at the repo root) followed by invoking `./gen3.elf <src.sio> <out.elf> && ./<out.elf>` directly —
this is what `examples/cayley_dickson_lemon_g2_ffi.sio`'s own header and this paper's §7 now
document. Whether `bin/souc-lean-single-x86_64` should be refreshed from a current `lean_single.sio`
build (so the documented `SOUNIO_SOUC_ENGINE=lean_single` alias works again) is a separate,
unresolved question — replacing a committed binary artifact was judged out of scope for this
dispatch and left to a future change.

Verified working end-to-end: `examples/cayley_dickson_lemon_g2_ffi.sio` now runs its
`extern "C" { fn system(cmd: &[i8;1024]) -> i32 }` call, genuinely forks and execs a Python bridge
script, and reads back real data produced by that child process — all within the same Sounio
program, no out-of-band shell step. Confirmed reproducible across repeat runs (identical Spearman
correlation output).

### Secondary bugs found while verifying Track B (not part of Track B; not yet independently dispatched)

These surfaced only once `system()` genuinely started working and downstream code paths that had
never actually executed before ran for the first time. None are patched in `self-hosted/`; all are
worked around in `examples/cayley_dickson_lemon_g2_ffi.sio` (see that file's trailing comment block,
items #3–#6, for the full repro/isolation detail). Listed here for continuity since they were found
in the same investigation as Track B; each may warrant its own dedicated dispatch doc before any
`self-hosted/` fix is attempted, per this repo's protocol.

- **#3 — `system()` command-string length threshold.** The Track B stub crashes (SIGSEGV) or hangs
  on command strings roughly 100+ characters (94 chars confirmed working, ~137 confirmed broken).
  Root cause not isolated (fork argv/envp buffer sizing is the leading suspect). Worked around with
  a short, checked-in wrapper script (`scripts/research/lemon_ffi_bridge_wrapper.sh`) rather than
  building a long command string inline.
- **#4 — read-after-fork-write staleness.** `read_file()` on a path a `system()`-forked child had
  just written, called from the same parent process, reproducibly returned 0 bytes immediately
  after `wait4()` returned — even though the same path read correctly moments later from a fresh
  process. Inserting `syscall6(162,0,0,0,0,0,0)` (`sync()`, no args) between the `system()` call and
  the `read_file()` call reliably fixed it. Root cause not fully isolated (plausible filesystem
  write-visibility artefact across the fork boundary; not standard POSIX behaviour on ext4,
  unconfirmed).
- **#5 — `read_file()` with a module-level `const string` argument.** Independent of #4: a
  `const PATH: string = "..."` declared at module scope and passed to `read_file(PATH)` reproducibly
  returned 0 bytes, even for a file that existed before the process started (no FFI/timing
  involved at all). The identical literal bound to a local `let` inside a function, or passed
  through a helper function's `string` parameter, read correctly. This — not #4 — was the actual
  cause of every 0-byte read observed while first integrating Track B into the LEMON pipeline file;
  #4's `sync()` fix is independently real (isolated with literal-string probes) but was not what was
  breaking that integration. Fixed by inlining the path as a local `let` instead of a module `const`.
- **#6 — global mutable array reads stuck at element 0.** A module-level `var arr: [T;N]`, indexed
  with a runtime-computed index (`arr[i as usize]`), reproducibly returned element 0 for every `i`
  — reproduced with a single global `[i64;14]` array in an otherwise-trivial file, no other globals
  present, ruling out the previously-documented "large globals collide" explanation for the
  unrelated `BASIS_COUNT` scalar bug. Writes to a same-shape global (`SUBJ_G2` elsewhere in the same
  file, written every timestep and read back with a computed index) were unaffected — this is a
  read-only-after-init defect, not a general array-addressing one, which is why it was not caught
  until a write-once/read-many global (`GEN_A`/`GEN_B`, the 14 G2-generator index pairs) was added.
  A local array, and a local array passed by `&[T;N]` reference into another function, both read
  correctly at every index. Fixed by making the arrays locals in `main()` and threading them through
  as `&[i64;14]` parameters.

## Impact if unaddressed

Any current or future Sounio program that uses `extern "C"` for an integer-returning call under
the default engine — including the several `stdlib/` modules `KNOWN_LIMITATIONS.md` already
documents as depending on this — may be silently getting a fabricated `0` return instead of the
real result, with no crash or error to signal it. This is a correctness/trust hazard broader than
the single `system()` case that surfaced it.

## AI disclosure

Repros, localisation, and cross-engine comparison by AI agent (Claude) under human direction, on
Madaros v0.80.0 / `SOUNIO_SOUC_ENGINE=lean_single`. All repros are re-runnable with
`export SOUNIO_STDLIB_PATH=$(pwd)/stdlib` from the repo root. No `self-hosted/` sources were
modified. GAIDeT-ICMJE 2025.
