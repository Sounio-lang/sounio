<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.read-line
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.read-line
-->

# A64 PARITY — `read_line()` stdin builtin

**Opened / closed.** 2026-05-21 (same session).
**Status.** RESOLVED — CODE CHANGE LANDED.
**Class.** Codegen edit on `self-hosted/compiler/lean_single.sio` (new
`emit_read_line_a64` + dispatch in `compile_primary_a64`).
**Branch.** `feat/windows-assert-exit`.
**Cadence siblings.** `AUDIT.md` (emit_assert_fail_a64), `TRIG_BUILTINS.md`.

---

## §1 — The gap

`read_line()` (read one line from stdin into a fresh buffer, return a string)
was implemented in the x86 expression compiler (`emit_read_line_x86`, dispatched
in `compile_primary` ~`:10923`) but absent from `compile_primary_a64`. On
`aarch64-macos` / `aarch64-linux` it was not recognised as a builtin and fell
through the identifier path — unsupported.

## §2 — The fix

Added `emit_read_line_a64` (modelled on the proven `emit_read_file_a64` syscall
scaffolding) and a dispatch clause in `compile_primary_a64` immediately before
`read_file`, mirroring the x86 ordering and the `IO` effect check
(`FN_EFFECTS & 1`), `EXPR_TY = 3` (string). The emitted sequence:

```
mmap(NULL, 4096, PROT_RW, MAP_PRIVATE|MAP_ANON, -1, 0)   ; x8=222 / x16=197
push buf
read(0, buf, 4095)                                        ; x8=63  / x16=3
clamp bytes_read < 0 → 0
if bytes != 0: strip trailing '\n'  (ldurb / cmp #10 / sub)
null-terminate buf[bytes]
return buf
```

OS dispatch (Linux syscall vs macOS `x16`/`svc #0x80`) is inherited from the
`read_file_a64` pattern and `emit_syscall_a64`.

## §3 — Two deliberate a64↔x86 divergences (preserved on purpose)

1. **Negative-length clamp.** x86 does `mov byte[rcx+rax],0` with `rax` possibly
   a negative errno → latent OOB write. `emit_read_line_a64` clamps `bytes<0`
   to `0` first (the `emit_read_file_a64` house pattern). User-visible behaviour
   for normal input (bytes ≥ 0) is identical.
2. **Function signature.** `emit_read_line_a64` is `with Mut, Panic` (it calls
   `patch_branch_a64`, which is `Panic`), whereas `emit_read_line_x86` is `Mut`
   only (it uses fixed-offset `jne`, no backpatching). Required for the a64
   self-host typecheck to pass.

## §4 — Verification

- **Self-host fixed point.** `lean_single_fixed_point_gate.sh` PASS:
  stage1 == stage2 == stage3, `md5=7afd6bc2ca3288d15061364a056e5fab`
  (size 2 177 378 B); `bin/souc-linux-x86_64` rebuilt to match.
- **A64 codegen, empirical disasm** (`--target aarch64-linux`): the emitted
  `read_line` body is byte-correct — `mmap #222`, `read #63`,
  `ldurb w3,[x1,#-1]` / `cmp w3,#0xa` / `b.ne` / `sub x2,x2,#1`, `strb wzr,[x1]`,
  post-index pop. (A first cut had two transposed encodings — `0x381FF023`
  decoded as STURB not LDURB, and `0x71002A7F` as `cmp w19` not `cmp w3` — caught
  by disasm and fixed to `0x385FF023` / `0x7100287F` before commit. The
  fixed-point gate alone did *not* catch them; disasm did.)
- **x86 non-regression.** `read_line()` compiled to the default x86 target and
  run with piped stdin echoes the line with the trailing newline stripped,
  exit 0.
- **A64 runtime:** no aarch64 emulator on this host (no qemu, no binfmt), so the
  disasm + the structural identity with the proven `read_file_a64` syscall path
  are the evidence; no end-to-end aarch64 execution.

## §5 — Conclusion

`read_line()` is now supported on the ARM64 backend at parity with x86 (modulo
the two documented deliberate divergences). Remaining tracked a64 parity item:
transcendental AD shadows.
