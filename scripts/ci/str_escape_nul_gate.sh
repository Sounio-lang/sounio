#!/usr/bin/env bash
# ADR-015: literal and value strings preserve embedded NUL on both engines.
# Compare binary output files; shell variables cannot hold embedded NUL.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *) echo "[str-escape-nul] SKIP: x86-64 Linux only"; exit 0 ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
source "$ROOT_DIR/scripts/lib/souc_invoke.sh"
sounio_require_souc
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

TMP="$(mktemp -d /tmp/str-escape-nul.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
python3 - "$TMP" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
(root / "literal.expected").write_bytes(b"a\0b\nEND\n")
(root / "values.expected").write_bytes(
    b"a\0b\n\0xy\0\na\0b\0c\n\0b\0\n\0b\na\0b\n\n"
    + "é\0😀\nEND\n".encode("utf-8")
)
(root / "input with spaces.bin").write_bytes(b"\0a\0b\0")
(root / "io.expected").write_bytes(b"\0a\0b\0\n" + "é😀\n".encode("utf-8"))
accepted = {
    "simple": (r"""println("\n\t\r\0\\\"\'")""", bytes([10, 9, 13, 0, 92, 34, 39, 10])),
    "hex": (r'println("\x41\x00\xff")', b"A\0\xff\n"),
    "unicode": (
        r'println("\u{7f}\u{80}\u{7ff}\u{800}\u{d7ff}\u{e000}\u{ffff}\u{10000}\u{10ffff}")',
        bytes.fromhex("7f c2 80 df bf e0 a0 80 ed 9f bf ee 80 80 ef bf bf f0 90 80 80 f4 8f bf bf 0a"),
    ),
}
rejected = {
    "unknown": r"\q", "hex_short": r"\x4", "hex_digit": r"\xGG",
    "unicode_empty": r"\u{}", "unicode_surrogate": r"\u{d800}",
    "unicode_overflow": r"\u{110000}", "unicode_width": r"\u{0000041}",
    "unicode_unclosed": r"\u{41", "unicode_digit": r"\u{gg}",
}
for name, (body, expected) in accepted.items():
    (root / (name + ".sio")).write_text("fn main() -> i64 with IO {\n" + body + "\n0\n}\n")
    (root / (name + ".expected")).write_bytes(expected)
for name, escape in rejected.items():
    (root / (name + ".sio")).write_text(
        'fn main() -> i64 with IO {\nprintln("' + escape + '")\n0\n}\n'
    )
PY

run_one() {
  local name="$1"
  local bin="$2"
  local kind="$3"
  if [ "$(souc_banner "$bin")" != "$kind" ]; then
    echo "[str-escape-nul] FAIL: $name has the wrong engine identity ($bin)"
    return 1
  fi
  local fixture src elf
  for fixture in literal values io simple hex unicode; do
    if [ "$fixture" = literal ]; then
      src="$ROOT_DIR/tests/run-pass/str_escape_nul_print.sio"
    elif [ "$fixture" = values ]; then
      src="$ROOT_DIR/tests/run-pass/str_escape_nul_values.sio"
    elif [ "$fixture" = io ]; then
      src="$ROOT_DIR/tests/run-pass/str_escape_nul_io.sio"
    else
      src="$TMP/$fixture.sio"
    fi
    elf="$TMP/$name-$fixture.elf"
    if ! souc_compile "$bin" "$src" "$elf" >"$TMP/$name-$fixture.build" 2>&1; then
      echo "[str-escape-nul] FAIL: $name/$fixture did not compile ($bin)"
      cat "$TMP/$name-$fixture.build"
      return 1
    fi
    chmod +x "$elf" || return 1
    local args=()
    if [ "$fixture" = io ]; then
      args=("$TMP/input with spaces.bin" "é😀")
    fi
    if ! "$elf" "${args[@]}" >"$TMP/$name-$fixture.out"; then
      echo "[str-escape-nul] FAIL: $name/$fixture exited nonzero"
      return 1
    fi
    if ! cmp -s "$TMP/$name-$fixture.out" "$TMP/$fixture.expected"; then
      echo "[str-escape-nul] FAIL: $name/$fixture bytes differ"
      od -An -tx1 "$TMP/$name-$fixture.out"
      return 1
    fi
    echo "[str-escape-nul] PASS: $name/$fixture"
  done
  for fixture in unknown hex_short hex_digit unicode_empty unicode_surrogate \
      unicode_overflow unicode_width unicode_unclosed unicode_digit; do
    elf="$TMP/$name-$fixture.elf"
    souc_compile "$bin" "$TMP/$fixture.sio" "$elf" >"$TMP/$name-$fixture.build" 2>&1
    local rc=$?
    if [ "$rc" -ne 1 ] || [ -s "$elf" ]; then
      echo "[str-escape-nul] FAIL: $name/$fixture must cleanly refuse without an ELF (rc=$rc)"
      cat "$TMP/$name-$fixture.build"
      return 1
    fi
    echo "[str-escape-nul] PASS: $name/$fixture refused"
  done
}

fail=0
run_one madaros "$SOUC_BIN" madaros || fail=1

SEED="${SOUC_LEAN_SINGLE:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
if [ -x "$SEED" ]; then
  run_one lean_single "$SEED" lean_single || fail=1
else
  echo "[str-escape-nul] FAIL: no executable seed at $SEED"
  fail=1
fi
exit "$fail"
