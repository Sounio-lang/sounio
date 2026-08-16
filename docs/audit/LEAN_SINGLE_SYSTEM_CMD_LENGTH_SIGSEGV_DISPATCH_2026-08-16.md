<!-- docs:meta
topic_id: repo.docs.audit.lean-single-system-cmd-length-sigsegv-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-system-cmd-length-sigsegv-dispatch-2026-08-16
-->

# String literals ≥127 bytes mis-jump (`jmp rel8` saturation) — crash masquerading as a `system()` command-length defect — dispatch

**Date:** 2026-08-16
**Engine:** lean_single, source-built fixed point (`make build` gen3, md5 `37c1cf8a43ab74143994ec77b9a45e5e`; identical to the refreshed `bin/souc-lean-single-x86_64`)
**Parent:** `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md` §"Secondary bugs", item #3; originally reported as "`system()` command-string length threshold (~94 works, ~137 broken)". Repro detail first recorded in the trailing comment block of `examples/cayley_dickson_lemon_g2_ffi.sio` at `e1109c4773`.
**Owner:** unassigned
**Status:** OPEN — dispatched; **root cause isolated** (see §Root-cause locus). No `self-hosted/` change made here.

## Why this dispatch

The Track B `system()` stub was reported to crash or hang on command strings of roughly 137+ characters, and the LEMON pipeline worked around it by routing long commands through a checked-in wrapper script. This dispatch shows the defect is not in the stub, not in FFI, and not in `fork`/`execve`/`wait4` at all: **any string literal of ≥127 bytes, anywhere in a lean_single-compiled program, emits code that jumps into the middle of the instruction stream.** Every long literal is a latent crash, with no diagnostic. The `system()` report was simply the first place a long literal was ever exercised at runtime.

## Defect and reproduction

### Bisect through `system()` (the original framing)

Command built into a `[i8;1024]` buffer from a string literal of total length L, passed to the Track B stub, run with `timeout 8`:

| L | result |
|---|---|
| 88–131 | rc=0, side-effect file created (at L>126 this was already luck — see below) |
| 132–135 | SIGSEGV (rc=139) or hang (rc=124), varying between runs of the same ELF |
| 136, 144 | hang (rc=124) |

(The original "94 works / ~137 broken" bracket is consistent; the exact edge is at 126/127 content bytes.)

### Minimal repro — no FFI, no `system()`, one literal

```sounio
fn main() -> i32 with IO, Mut, Panic, Div {
    let s = "aaa…(exactly 127 'a')…"
    print("len="); print_i64(str_len(s) as i64); print("\n")
    0
}
```

With the literal at exactly 126 bytes: prints `len=126`, rc=0. At 127 bytes: no output, rc=139 (SIGSEGV). At 128–129: rc=42 (garbage exit). At 130: SIGSEGV. At 131: rc=42. Failure mode varies with length and is not deterministic in form, but **every length ≥127 fails**. Probes: `/tmp/ffi_probe/b3/one_127.sio` … `one_131.sio` (regenerable from this table).

### Root cause, read from the emitted code path

`self-hosted/compiler/lean_single.sio:11830-11837` (x86-64 string-literal emission):

```
// Emit: jmp over string data; string bytes; null terminator; lea rax, [rip - N]
em(0xeb)                 // jmp rel8  — SHORT jump, signed 8-bit displacement
let jmp_off = CL
em(slen + 1)             // displacement = literal length + NUL, truncated to one byte
let str_start = CL
… bytes … ; em(0)        // the literal itself, inline in the instruction stream
```

The literal's bytes are emitted **inline in the code stream**, and execution skips over them with a `jmp rel8` whose displacement is `slen + 1`. A signed rel8 saturates at **+127**:

- `slen = 126` → displacement 127 (0x7F) — the exact maximum that works.
- `slen ≥ 127` → displacement ≥ 128 → `em()` writes only the low byte (0x80 = −128 …) → the jump lands **behind** the literal, mid-instruction → SIGSEGV / SIGILL (rc=132/139) / hang / garbage exit code, depending on what now executes.

The measured 126/127 boundary is the rel8 saturation boundary, exactly. The `system()` "length threshold" was this defect observed through the stub: a long command means a long **literal**, and the program crashes before the syscall is ever reached. This also explains why the failure looked nondeterministic — what executes after the mis-jump depends on the surrounding code bytes.

The arm64 twin (`compile_or_a64`, ~`:32410`) uses an ADR-family immediate with a 19-bit range and is not affected by this boundary.

## Ruled out

- **The Track B stub itself** — the minimal repro contains no FFI; a plain `str_len(<long literal>)` crashes identically. The stub's `argv`/`envp` (`[i64;4]`/`[i64;2]`) and buffer sizing, the original leading suspect, are not implicated.
- **`fork`/`execve`/`wait4` marshaling** — never reached in the failing cases.
- **Runtime heap/BSS layout** — the crash occurs before any data structure is touched; the corruption is in emitted code, not data.
- **My own earlier "wrong length at 120" reading** — retracted: that probe's expected values were miscounted; exactly-constructed literals ≤126 measure correctly. Recorded here so the retraction is auditable.

## Root-cause locus

**Isolated:** `self-hosted/compiler/lean_single.sio:11830-11833`, `jmp rel8` + `em(slen + 1)` displacement truncation. Confidence: the 126/127 measured edge matches the 127 rel8 maximum with no slack, and the failure-mode zoo (SEGV/ILL/hang/odd rc) is what execution-after-mis-jump produces. Not yet double-checked by disassembling a failing ELF (the inference is from source + boundary measurement; a `objdump` of `one_127.elf` around the literal would close that last inch).

## Proposed fix locus

At the same site: when `slen + 1 > 127`, emit a `jmp rel32` (`0xE9` + `em32(slen + 1)`) instead of `0xEB` + byte (or unconditionally use rel32; the cost is 4 bytes per literal). Per house protocol this is recorded for a future dispatch-gated change, not applied here — `lean_single.sio` is under an active coordination claim and the fixed-point bootstrap must be re-verified after any edit.

## Acceptance gate (proposed)

A checked-in test, engine-forced à la `tests/run-pass/ffi_system_exec.sio` (`//@ ignore` + a gate that runs it under a source-built lean_single): a `main` containing string literals at 126, 127, and 300 bytes, asserting `str_len` of each and rc=0. Pre-fix, the 127/300 arms crash; post-fix, all pass. Then `scripts/dev/souc-build-lock.sh make build` must still report FIXED POINT OK.

## Impact if unaddressed

Any lean_single-compiled program with a string literal of ≥127 bytes — long paths, long shell commands (the `system()` case), formatted messages, embedded JSON/CSV — is a latent crash with no compile-time diagnostic. Workarounds (wrapper scripts, split concatenation) silently mask it. The LEMON bridge's `lemon_ffi_bridge_wrapper.sh` exists because of this defect.

## AI disclosure

Repros, bisect, and localisation by AI agent (Claude) under human direction, 2026-08-16, on lean_single gen3 (md5 `37c1cf8a…`). All probes regenerable from the tables above with `unset SOUC_BIN SOUNIO_STDLIB_PATH`. No `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.
