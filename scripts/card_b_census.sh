#!/usr/bin/env bash
# Card B baseline census — per docs/SELF_HOSTING_PUNCHLIST.md
# Distinguishes compile-time CRASH (compiler binary SIGSEGV) from runtime CRASH (ELF crashes).
set -u
SOUC="${1:-bin/souc.elf}"
OUT="${2:-artifacts/omega/card_b_census.v1.json}"
LISTS_DIR="${3:-artifacts/omega/card_b_lists}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
export SOUNIO_STDLIB_PATH="$WORKDIR/stdlib"
cd "$WORKDIR"

if [ ! -x "$SOUC" ]; then
  echo "error: souc binary not executable: $SOUC" >&2
  exit 1
fi

mkdir -p "$LISTS_DIR"
L_CRASH_C="$LISTS_DIR/crash_compile.txt"
L_CRASH_R="$LISTS_DIR/crash_runtime.txt"
L_BF="$LISTS_DIR/backend_fail.txt"
L_ELF="$LISTS_DIR/elf_ok.txt"
L_PARSE="$LISTS_DIR/parse_fail.txt"
L_IR="$LISTS_DIR/ir_bodies.txt"
L_OTHER="$LISTS_DIR/other.txt"
: > "$L_CRASH_C"; : > "$L_CRASH_R"; : > "$L_BF"; : > "$L_ELF"
: > "$L_PARSE"; : > "$L_IR"; : > "$L_OTHER"

git ls-files '*.sio' | grep -E '^(examples|tests/run-pass)/' | while read -r f; do
  rm -f /tmp/x.elf
  out=$(timeout 25 "$SOUC" --native-v2-compile "$f" -o /tmp/x.elf 2>&1)
  rc=$?
  if [ "$rc" -ge 128 ] && [ "$rc" -ne 124 ]; then
    echo "SIGSEGV rc=$rc $f" >> "$L_CRASH_C"
  elif [ "$rc" = 124 ]; then
    echo "TIMEOUT $f" >> "$L_CRASH_C"
  elif [ -f /tmp/x.elf ] && [ -s /tmp/x.elf ]; then
    chmod +x /tmp/x.elf 2>/dev/null || true
    if /tmp/x.elf >/dev/null 2>&1; then
      echo "$f" >> "$L_ELF"
    else
      rc2=$?
      echo "RUNTIME_CRASH rc=$rc2 $f" >> "$L_CRASH_R"
    fi
  elif echo "$out" | grep -qi "parse_failed\|parse error"; then
    echo "$f" >> "$L_PARSE"
  elif echo "$out" | grep -qi "ir_bodies_failed"; then
    echo "$f" >> "$L_IR"
  elif echo "$out" | grep -qi "backend\|FAIL to_file\|unsupported"; then
    echo "$f" >> "$L_BF"
  else
    echo "$f" >> "$L_OTHER"
  fi
done

CRASH_C=$(wc -l < "$L_CRASH_C")
CRASH_SIGSEGV=$(grep -c '^SIGSEGV' "$L_CRASH_C" || true)
CRASH_TIMEOUT=$(grep -c '^TIMEOUT' "$L_CRASH_C" || true)
CRASH_R=$(wc -l < "$L_CRASH_R")
BF=$(wc -l < "$L_BF")
ELF=$(wc -l < "$L_ELF")
PARSE=$(wc -l < "$L_PARSE")
IR=$(wc -l < "$L_IR")
OTHER=$(wc -l < "$L_OTHER")
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$(dirname "$OUT")"
cat > "$OUT" <<JSON
{
  "schema": "sounio.card_b_census.v1",
  "generated_at_utc": "$TS",
  "souc_binary": "$SOUC",
  "total": $((CRASH_C + CRASH_R + BF + ELF + PARSE + IR + OTHER)),
  "crash_compile": $CRASH_C,
  "crash_compile_sigsegv": $CRASH_SIGSEGV,
  "crash_compile_timeout": $CRASH_TIMEOUT,
  "crash_runtime": $CRASH_R,
  "backend_fail": $BF,
  "elf_ok": $ELF,
  "parse_failed": $PARSE,
  "ir_bodies_failed": $IR,
  "other": $OTHER
}
JSON
echo "--- CENSUS SUMMARY ($TS) ---"
cat "$OUT"
echo "--- list dir ---"
ls -la "$LISTS_DIR"
