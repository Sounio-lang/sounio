#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export LC_ALL=C

MODE="strict"
if [[ "${1:-}" == "--parser-only" ]]; then
  MODE="parser-only"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--parser-only]" >&2
  exit 64
fi

fail() {
  printf 'DECIMAL_LITERAL_ROUNDING_V1_FAIL reason=%s\n' "$1" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || fail python3_missing
command -v timeout >/dev/null 2>&1 || fail timeout_missing

SOURCE_HEAD="$(git rev-parse HEAD)"
SEED_COMPILER="$(realpath "${SOUNIO_DECIMAL_LITERAL_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
PARSER_PROBE="tests/compiler/decimal_literal_rounding_v1_parser_probe.sio"
LITERAL_PROBE="tests/compiler/decimal_literal_rounding_v1_literals.sio"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-decimal-literal-v1.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

[[ -x "$SEED_COMPILER" ]] || fail seed_compiler_missing
[[ "$(od -An -tx1 -N4 "$SEED_COMPILER" | tr -d ' \n')" == "7f454c46" ]] ||
  fail seed_compiler_must_be_elf

LITERALS=(
  "0.1"
  "3.14"
  "0.333333333333333"
  "2.718281828459045"
  "6.022e23"
  "1.7976931348623157e308"
  "5e-300"
  "1.23456789012345678"
  "5e-324"
  "2e-324"
  "3e-324"
  "2.2250738585072014e-308"
  "1.7976931348623158e308"
)

python3 - "${LITERALS[@]}" >"$TMP_DIR/oracle.tsv" <<'PY'
import math
import struct
import sys

for raw in sys.argv[1:]:
    value = float(raw)
    if not math.isfinite(value):
        raise SystemExit(f"oracle unexpectedly produced non-finite value for {raw}")
    bits = struct.unpack("<q", struct.pack("<d", value))[0]
    print(f"{raw}\t{bits}")
PY

printf 'source_head=%s\n' "$SOURCE_HEAD"
printf 'seed_elf=%s\n' "$SEED_COMPILER"
printf 'seed_sha256=%s\n' "$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
printf 'oracle=cpython_float_plus_struct_pack count=%s\n' "${#LITERALS[@]}"

PARSER_ELF="$TMP_DIR/parser-probe.elf"
if ! timeout 120 "$SEED_COMPILER" "$PARSER_PROBE" "$PARSER_ELF" \
    >"$TMP_DIR/parser-probe.build.log" 2>&1; then
  cat "$TMP_DIR/parser-probe.build.log" >&2
  fail source_fresh_parser_probe_compile
fi
chmod +x "$PARSER_ELF"

while IFS=$'\t' read -r literal expected_bits; do
  run_log="$TMP_DIR/parser-$(printf '%s' "$literal" | sha256sum | cut -c1-16).log"
  if ! timeout 20 "$PARSER_ELF" "$literal" >"$run_log" 2>&1; then
    cat "$run_log" >&2
    fail source_fresh_parser_probe_runtime
  fi
  actual_line="$(tr -d '\r' <"$run_log" | awk 'NF == 2 { print $1 "\t" $2 }' | tail -n 1)"
  if [[ "$actual_line" != "$literal"$'\t'"$expected_bits" ]]; then
    printf 'literal=%s expected_bits=%s actual_line=%s\n' \
      "$literal" "$expected_bits" "$actual_line" >&2
    fail source_fresh_parser_oracle_mismatch
  fi
  printf 'DECIMAL_LITERAL_ROUNDING_V1_CASE_PASS path=parser literal=%s bits=%s\n' \
    "$literal" "$expected_bits"
done <"$TMP_DIR/oracle.tsv"

assert_parser_rejects() {
  local label="$1"
  local raw="$2"
  local expected="$3"
  local log="$TMP_DIR/parser-reject-$label.log"
  local rc=0
  timeout 20 "$PARSER_ELF" "$raw" >"$log" 2>&1 || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    cat "$log" >&2
    fail "parser_${label}_unexpectedly_accepted"
  fi
  if [[ "$rc" -ne 1 ]]; then
    cat "$log" >&2
    fail "parser_${label}_unexpected_exit_${rc}"
  fi
  grep -Fq "$expected" self-hosted/parser/parser.sio ||
    fail "parser_${label}_source_diagnostic_missing"
  local transport="seed_panic_text_unavailable"
  if grep -Fq "$expected" "$log"; then
    transport="emitted"
  fi
  printf 'DECIMAL_LITERAL_ROUNDING_V1_REJECT_PASS path=parser label=%s diagnostic=%s transport=%s\n' \
    "$label" "$expected" "$transport"
}

assert_parser_rejects malformed_exponent "1e+" "invalid decimal float literal exponent"
assert_parser_rejects malformed_separator "1_.0" "invalid decimal float literal separator"
assert_parser_rejects f64_rounding_overflow \
  "1.7976931348623159e308" \
  "decimal float literal overflows f64"
assert_parser_rejects exponent_boundary "1e309" "decimal float literal overflows f64"
assert_parser_rejects exponent_overflow \
  "1e99999999999999999999999999999999999999999999999999" \
  "decimal float literal overflows f64"
PRECISION_LIMIT_LITERAL="$(python3 -c 'print("1" + "0" * 768 + "e-768")')"
assert_parser_rejects precision_limit \
  "$PRECISION_LIMIT_LITERAL" \
  "decimal float literal exceeds exact parser limit"

printf 'DECIMAL_LITERAL_ROUNDING_V1_PARSER_PASS source_fresh=true retained=8 total=%s rejects=6\n' \
  "${#LITERALS[@]}"

if [[ "$MODE" == "parser-only" ]]; then
  echo "DECIMAL_LITERAL_ROUNDING_V1_LITERAL_PATH_NOT_RUN reason=explicit_parser_only"
  exit 0
fi

COMPILER="${SOUNIO_DECIMAL_LITERAL_COMPILER:-}"
COMPILER_SOURCE_SHA="${SOUNIO_DECIMAL_LITERAL_COMPILER_SOURCE_SHA:-}"
COMPILER_CLI="${SOUNIO_DECIMAL_LITERAL_COMPILER_CLI:-$ROOT_DIR/bin/madaros}"
[[ -n "$COMPILER" ]] || fail explicit_source_fresh_compiler_required
COMPILER="$(realpath "$COMPILER")"
[[ -x "$COMPILER" ]] || fail source_fresh_compiler_missing
[[ "$(od -An -tx1 -N4 "$COMPILER" | tr -d ' \n')" == "7f454c46" ]] ||
  fail source_fresh_compiler_must_be_elf
[[ -x "$COMPILER_CLI" ]] || fail compiler_cli_missing
[[ -n "$COMPILER_SOURCE_SHA" ]] || fail compiler_source_sha_required
[[ "$COMPILER_SOURCE_SHA" == "$SOURCE_HEAD" ]] || fail compiler_source_sha_mismatch

printf 'compiler_elf=%s\n' "$COMPILER"
printf 'compiler_sha256=%s\n' "$(sha256sum "$COMPILER" | awk '{print $1}')"
printf 'compiler_source_sha=%s\n' "$COMPILER_SOURCE_SHA"

LITERAL_ELF="$TMP_DIR/literal-probe.elf"
if ! MADAROS_RAW_BIN="$COMPILER" timeout 180 "$COMPILER_CLI" \
    compile "$LITERAL_PROBE" -o "$LITERAL_ELF" \
    >"$TMP_DIR/literal-probe.build.log" 2>&1; then
  cat "$TMP_DIR/literal-probe.build.log" >&2
  fail source_fresh_literal_probe_compile
fi
[[ -f "$LITERAL_ELF" ]] || fail source_fresh_literal_probe_artifact_missing
chmod +x "$LITERAL_ELF"
if ! timeout 30 "$LITERAL_ELF" >"$TMP_DIR/literal-probe.run.log" 2>&1; then
  cat "$TMP_DIR/literal-probe.run.log" >&2
  fail source_fresh_literal_probe_runtime
fi

python3 - "$TMP_DIR/oracle.tsv" "$TMP_DIR/literal-probe.run.log" <<'PY'
import sys

expected = {}
with open(sys.argv[1], encoding="ascii") as handle:
    for line in handle:
        literal, bits = line.rstrip("\n").split("\t")
        expected[literal] = bits

actual = {}
with open(sys.argv[2], encoding="ascii", errors="replace") as handle:
    for line in handle:
        fields = line.split()
        if len(fields) == 2 and fields[0] in expected:
            if fields[0] in actual:
                raise SystemExit(f"duplicate runtime line for {fields[0]}")
            actual[fields[0]] = fields[1]

missing = sorted(set(expected) - set(actual))
mismatched = sorted(k for k in expected if actual.get(k) != expected[k])
if missing or mismatched:
    raise SystemExit(f"runtime oracle mismatch missing={missing} mismatched={mismatched}")
PY

assert_literal_rejected() {
  local label="$1"
  local source="$2"
  local expected="$3"
  local output="$TMP_DIR/$label.elf"
  local log="$TMP_DIR/$label.compile.log"
  local rc=0
  MADAROS_RAW_BIN="$COMPILER" timeout 180 "$COMPILER_CLI" \
    compile "$source" -o "$output" >"$log" 2>&1 || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    cat "$log" >&2
    fail "literal_${label}_unexpectedly_accepted"
  fi
  if [[ "$rc" -ne 1 ]]; then
    cat "$log" >&2
    fail "literal_${label}_unexpected_exit_${rc}"
  fi
  [[ ! -e "$output" ]] || fail "literal_${label}_artifact_produced"
  grep -Fq "$expected" self-hosted/parser/parser.sio ||
    fail "literal_${label}_source_diagnostic_missing"
  local transport="compiler_panic_text_unavailable"
  if grep -Fq "$expected" "$log"; then
    transport="emitted"
  fi
  printf 'DECIMAL_LITERAL_ROUNDING_V1_REJECT_PASS path=literal label=%s diagnostic=%s transport=%s\n' \
    "$label" "$expected" "$transport"
}

assert_literal_rejected malformed_exponent \
  tests/compiler/decimal_literal_rounding_v1_malformed_exponent.sio \
  "invalid decimal float literal exponent"
assert_literal_rejected malformed_separator \
  tests/compiler/decimal_literal_rounding_v1_malformed_separator.sio \
  "invalid decimal float literal separator"
assert_literal_rejected f64_overflow \
  tests/compiler/decimal_literal_rounding_v1_f64_overflow.sio \
  "decimal float literal overflows f64"
assert_literal_rejected exponent_overflow \
  tests/compiler/decimal_literal_rounding_v1_exponent_overflow.sio \
  "decimal float literal overflows f64"

printf 'DECIMAL_LITERAL_ROUNDING_V1_PASS source_fresh=true retained=8 total=%s rejects=4 fallback=0\n' \
  "${#LITERALS[@]}"
