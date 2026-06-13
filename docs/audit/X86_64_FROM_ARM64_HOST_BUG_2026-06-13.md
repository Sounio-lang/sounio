# x86_64 emission from an arm64 host — typecheck false-positive (E200 flood)

Status: **root-cause class identified, exact a64-codegen fault not yet pinned.**
This is a *separate* bug from the four macOS arm64 fixes (FN_RET_TY_TOK, epilogue,
merge-nops, fn-param closure). It does not affect arm64-target output.

## Symptom

The self-hosted compiler running on an **arm64 host** (Apple Silicon, or arm64-linux
under qemu) emits a flood of `E200 unknown identifier ` `` (empty identifier name)
and `typecheck: failed` (exit 1, no binary) whenever it targets **x86_64** —
`--target x86_64-macos` *and* `--target x86_64-linux` both fail identically. The
**same binary** targeting `aarch64-*` compiles the same source cleanly. The x86
host emits x86_64 fine. Reproduces on a trivial program (`fn main(){ let x=5; x }`).

## Linux reproduction (no Mac needed)

```bash
sudo apt-get install -y qemu-user-static          # qemu-aarch64-static
# Build the compiler as an arm64-linux ELF with a known-good x86 host:
scripts/dev/souc-build-lock.sh <x86-souc> self-hosted/compiler/lean_single.sio \
  /tmp/souc-arm64linux --target aarch64-linux
chmod +x /tmp/souc-arm64linux
printf 'fn main() -> i64 {\n    let x = 5\n    x\n}\n' > /tmp/tiny.sio
qemu-aarch64-static /tmp/souc-arm64linux /tmp/tiny.sio /tmp/o --target x86_64-linux
#   → 403× "E200 `` at line N" + "typecheck: failed"; aarch64-linux target → clean
```
The arm64-linux host under qemu is a faithful stand-in for the arm64-macos host
(the divergent code is target-driven, not host-OS-driven).

## What is established

- **It is an a64-codegen miscompilation of `compile_primary` (the x86 codegen
  function), not a source logic bug.** The x86 host runs the identical source
  correctly. `compile_primary_a64` (the a64 variant) is fine — only the x86
  variant, when compiled to a64, misbehaves.
- **Regression point: merge `5f1e397a2`** ("merge garden/above-stars …"), found by
  binary-searching distinct `lean_single.sio` blobs with the qemu repro. The merge
  botched the `compile_primary` merge — it took the garden branch's divergent
  version, changing ~390 lines (lost the N-ary tuple-staging + unit-dim handling).
  Good parent `7d3166367`, first bad `5f1e397a2`.
- **Ruled out:** a64 branch-displacement overflow (instrumented `patch_branch_a64`:
  `BRANCH_OVF_N=0`); large-frame local offset encoding (`emit_load/store_var_a64`
  handle >4095 offsets via full-immediate address compute); macOS-specificity
  (x86_64-linux fails the same).
- **Fault shape:** control-flow corruption in `compile_primary` on the a64 host.
  Instrumenting the E200 site showed `ns` reads correctly (e.g. 119) but execution
  **skips the rest of the block right after a `print_int(ns)` call** — a print
  immediately following the call never runs; control jumps to the next identifier.
  Looks like a premature return / mis-executed path after a call in this very large
  a64-compiled function. The empty name follows because `ne` is never reached.

## Open

The exact miscompiled instruction in `compile_primary`'s a64 lowering is not yet
pinned. Next tools: qemu `-g` + gdb-multiarch on the tiny repro, or instruction
trace around the E200 region. Candidate fix directions: (1) the a64-codegen bug
itself; (2) reconciling the `5f1e397a2` `compile_primary` merge against `7d3166367`.

x86_64-from-arm64 is a cross-compile-to-Intel path; arm64-native targets are
unaffected, so this does not block arm64 Mac usage.
