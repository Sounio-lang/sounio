#!/usr/bin/env bash
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; TOTAL=10

run() { local n=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS: $n"; PASS=$((PASS+1)); else echo "  FAIL: $n"; FAIL=$((FAIL+1)); fi; }
grepcheck() { local n=$1 f=$2 p=$3; if grep -q "$p" "$f"; then echo "  PASS: $n"; PASS=$((PASS+1)); else echo "  FAIL: $n"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 125 — Wide Buffers for Native Pipeline ==="
run "T1_encode_check" $SOUC check self-hosted/native/encode.sio
run "T2_elf_check" $SOUC check self-hosted/native/elf.sio
run "T3_reloc_check" $SOUC check self-hosted/native/reloc.sio
run "T4_file_write_check" $SOUC check self-hosted/io/file_write.sio
grepcheck "T5_WideCodeBuffer" self-hosted/native/encode.sio "WideCodeBuffer"
grepcheck "T6_WideElf64Binary" self-hosted/native/elf.sio "WideElf64Binary"
grepcheck "T7_WideRelocationTable" self-hosted/native/reloc.sio "WideRelocationTable"
grepcheck "T8_262144_code" self-hosted/native/encode.sio "262144"
grepcheck "T9_262144_elf" self-hosted/native/elf.sio "262144"
grepcheck "T10_wide_write" self-hosted/io/file_write.sio "io_write_wide_binary"

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$TOTAL"
