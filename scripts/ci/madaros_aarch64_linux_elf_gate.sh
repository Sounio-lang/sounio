#!/usr/bin/env bash
# SIO-32: Madaros emits an ARM aarch64-linux ELF (was: bare rc=11, no diagnostic).
#
# `souc compile --target aarch64-linux` typechecked and lowered fine, then
# died at the native binary write: the file-writing entry only knew
# x86_64-linux and arm64-macos. The aarch64-linux preview backend (machine
# module + tabulated encoder + Linux entry trampoline + ELF finalizer)
# existed but was unreachable from `souc compile`. This gate pins the
# wire-up: a hello-world program must compile to an ARM aarch64 ELF, and a
# genuinely unsupported target must fail with a sentence, not a number.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_aarch64_linux_elf_gate =="
printf 'fn main() with IO { print("hi\\n") }\n' > "$OUT/min_a64.sio"
if ! "$SOUC" compile "$OUT/min_a64.sio" -o "$OUT/min_a64.elf" --target aarch64-linux >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile --target aarch64-linux"
  tail -20 "$OUT/compile.log" || true
  exit 1
fi
KIND="$(file "$OUT/min_a64.elf" 2>/dev/null || true)"
case "$KIND" in
  *"ELF 64-bit LSB executable, ARM aarch64"*) ;;
  *)
    echo "FAIL: unexpected artifact: $KIND"
    exit 1
    ;;
esac
# Unsupported targets fail with a sentence, never a bare rc.
if "$SOUC" compile "$OUT/min_a64.sio" -o "$OUT/nope.elf" --target riscv64-linux >"$OUT/refuse.log" 2>&1; then
  echo "FAIL: riscv64-linux unexpectedly compiled"
  exit 1
fi
grep -q "is not supported by this backend yet" "$OUT/refuse.log" || {
  echo "FAIL: refusal without a named diagnostic"
  cat "$OUT/refuse.log" || true
  exit 1
}
echo "MADAROS_AARCH64_LINUX_ELF_GATE_OK"
