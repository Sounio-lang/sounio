#!/usr/bin/env bash
# Parse every Sounio source file in tests/run-pass with the current source
# parser bundle. Non-Sounio sidecars such as .econf JSON files are intentionally
# excluded from this parser-only sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc-linux-x86_64}"
TMPDIR="${TMPDIR:-/tmp}"
BUNDLE="${BUNDLE:-$TMPDIR/sounio_parse_runpass_current_bundle.sio}"
ELF="${ELF:-$TMPDIR/sounio_parse_runpass_current_bundle.elf}"
OUT="${OUT:-$TMPDIR/sounio_parse_runpass_current.out}"

: > "$BUNDLE"

for f in \
  self-hosted/intern.sio \
  self-hosted/lexer/token.sio \
  self-hosted/lexer/span.sio \
  self-hosted/lexer/cursor.sio \
  self-hosted/lexer/errors.sio \
  self-hosted/lexer/tables.sio \
  self-hosted/lexer/numparse.sio \
  self-hosted/lexer/mod.sio \
  self-hosted/parser/ast.sio \
  self-hosted/parser/parser.sio \
  self-hosted/parser/exprs.sio \
  self-hosted/parser/items.sio \
  self-hosted/parser/stmts.sio \
  self-hosted/parser/types.sio \
  self-hosted/parser/patterns.sio \
  self-hosted/parser/mod.sio \
  self-hosted/parser/recovery.sio \
  self-hosted/test_parse_all.sio
do
  /bin/sed '/^module /d; /^use /d; /^pub use /d' "$f" >> "$BUNDLE"
done

/bin/cat >> "$BUNDLE" <<'SIO'

fn parse_runpass_one(path: string) -> bool with IO, Mut, Panic, Div, Alloc {
    test_parse_file_str(path, -2)
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    var passed: i64 = 0
    var failed: i64 = 0
SIO

while IFS= read -r path; do
  esc=${path//\\/\\\\}
  esc=${esc//\"/\\\"}
  /usr/bin/printf '    if parse_runpass_one("%s") { passed = passed + 1 } else { failed = failed + 1; print("RUN_PASS_PARSE_FAIL "); print("%s"); print("\\n") }\n' "$esc" "$esc" >> "$BUNDLE"
done < <(/usr/bin/find tests/run-pass -type f -name '*.sio' | /usr/bin/sort)

/bin/cat >> "$BUNDLE" <<'SIO'
    print("RUN_PASS_PARSE_SUMMARY selected=")
    print_int(passed + failed)
    print("\npassed=")
    print_int(passed)
    print("\nfailed=")
    print_int(failed)
    print("\n")
    return if failed == 0 { 0 } else { 1 }
}
SIO

"$SOUC_BIN" "$BUNDLE" "$ELF"
/bin/chmod +x "$ELF"
"$ELF" > "$OUT" || true
/bin/grep -E "RUN_PASS_PARSE_(FAIL|SUMMARY)|passed=|failed=" "$OUT"
! /bin/grep -q "RUN_PASS_PARSE_FAIL" "$OUT"
