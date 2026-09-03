#!/usr/bin/env bash
# Focused regression gate for for/while loops + break/continue in Madaros.
#
# break/continue were silently broken in BOTH for and while: push_loop_labels
# recorded loop labels with a two-level nested store
# (`(*lo.loop_labels).break_labels[depth] = ...`) that the codegen discards, so
# current_break/continue_label found no target and every break/continue reported
# ir_bodies_failed. for-in range loops were also unlowered before the ExprForIn
# lowering landed. The generic full gate exercised neither, so both shipped green.
# This gate builds AND runs each fixture and checks the exit code.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
WORK="${SOUNIO_MADAROS_LOOP_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-loop.XXXXXX)}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
[[ "${SOUNIO_MADAROS_LOOP_GATE_KEEP:-0}" == "1" ]] || trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
fail() { echo "[madaros-loop] FAIL: $*" >&2; exit 1; }
pass() { echo "[madaros-loop] PASS: $*"; }
build_run() {
  local name="$1" expected="$2" src="$3"
  local sio="$WORK/$name.sio" elf="$WORK/$name.elf" log="$WORK/$name.log"
  printf '%s\n' "$src" > "$sio"; rm -f "$elf"
  # Check ELF-produced (production launcher prints "Output:", raw build subcommand
  # prints "emitted path="; both produce the ELF on success, a crash produces none).
  set +e; "$MADAROS" build "$sio" -o "$elf" >"$log" 2>&1; local crc=$?; set -e
  if [[ ! -s "$elf" ]]; then
    echo "[madaros-loop] build log tail for $name (rc=$crc):" >&2; tail -n 20 "$log" >&2 || true
    fail "$name: compiler produced no ELF (loop/break-continue lowering regression?)"
  fi
  chmod +x "$elf"
  set +e; "$elf"; local rrc=$?; set -e
  [[ "$rrc" == "$expected" ]] || fail "$name: expected exit $expected, got $rrc"
  pass "$name (exit $rrc)"
}
printf '[madaros-loop] madaros=%s\n' "$MADAROS"
build_run for_sum        45 'fn main() -> i64 { var s = 0
    for i in 0..10 { s = s + i }
    s }'
build_run for_inclusive  55 'fn main() -> i64 { var s = 0
    for i in 0..=10 { s = s + i }
    s }'
build_run for_var_end    45 'fn main() -> i64 { let n = 10
    var s = 0
    for i in 0..n { s = s + i }
    s }'
build_run for_array      60 'fn main() -> i64 { var a: [i64; 3] = [10, 20, 30]
    var s = 0
    for i in 0..3 { s = s + a[i] }
    s }'
build_run nested_for      9 'fn main() -> i64 { var s = 0
    for i in 0..3 { for j in 0..3 { s = s + 1 } }
    s }'
build_run for_break      10 'fn main() -> i64 { var s = 0
    for i in 0..10 { if i == 5 { break }
        s = s + i }
    s }'
build_run for_continue    8 'fn main() -> i64 { var s = 0
    for i in 0..5 { if i == 2 { continue }
        s = s + i }
    s }'
build_run while_break     10 'fn main() -> i64 { var s = 0
    var i = 0
    while i < 10 { if i == 5 { break }
        s = s + i
        i = i + 1 }
    s }'
echo "[madaros-loop] all loop/break/continue fixtures passed"
