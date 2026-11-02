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
# The verified surface beyond hello-world (executed on real arm64 hardware
# via scripts/dev/a64_run.sh, which no GHA runner can do): integer
# arithmetic and comparisons, float arithmetic/comparison/conversion, and
# the print builtins with real write-syscall bodies. CI pins the compile;
# hardware execution stays manual.
cat > "$OUT/wide_a64.sio" <<'EOF'
fn main() with IO {
  print("hi\n")
  let n: i64 = 40 + 2
  print_int(n)
  print("\n")
  if n == 42 {
    print("eq\n")
  }
  if n < 100 {
    print("lt\n")
  }
  let x: f64 = 1.5 + 2.25
  let w: f64 = (n as f64) + 0.5
  if x > 3.0 {
    print("fgt\n")
  }
  if -x < -3.0 {
    print("fneg\n")
  }
  if w > 42.0 {
    print("cv\n")
  }
  let y: f64 = x
  let m: i64 = y as i64
  print_int(m)
  print("\n")
}
EOF
if ! "$SOUC" compile "$OUT/wide_a64.sio" -o "$OUT/wide_a64.elf" --target aarch64-linux >"$OUT/wide.log" 2>&1; then
  echo "FAIL: compile --target aarch64-linux (wide surface)"
  tail -20 "$OUT/wide.log" || true
  exit 1
fi
WIDE="$(file "$OUT/wide_a64.elf" 2>/dev/null || true)"
case "$WIDE" in
  *"ELF 64-bit LSB executable, ARM aarch64"*) ;;
  *)
    echo "FAIL: unexpected wide artifact: $WIDE"
    exit 1
    ;;
esac
# Unsupported targets fail with a sentence, never a bare rc.
# Routed through the compile classifier (#2425): rc alone cannot tell a
# refusal from a segfault, timeout or OOM kill, and this backend's "Error:"
# messages carry no E-code, so the outcome lands in REFUSED_UNDIAGNOSED
# unless a crash marker says otherwise. The named fragment below pins which
# refusal fired.
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/classify_compile.sh"
refuse_rc=0
sounio_capture_rc refuse_rc "$OUT/refuse.log" -- \
  "$SOUC" compile "$OUT/min_a64.sio" -o "$OUT/nope.elf" --target riscv64-linux
sounio_classify_compile "$refuse_rc" "$OUT/refuse.log" "$OUT/nope.elf"
case "$SOUNIO_CC_CLASS" in
  REFUSED|REFUSED_UNDIAGNOSED) ;;
  *)
    echo "FAIL: riscv64-linux outcome is not a clean refusal: $(sounio_refusal_reason) rc=$refuse_rc"
    cat "$OUT/refuse.log" || true
    exit 1
    ;;
esac
grep -qF "is not supported by this backend yet" "$OUT/refuse.log" || {
  echo "FAIL: refusal without a named diagnostic"
  cat "$OUT/refuse.log" || true
  exit 1
}
echo "MADAROS_AARCH64_LINUX_ELF_GATE_OK"
