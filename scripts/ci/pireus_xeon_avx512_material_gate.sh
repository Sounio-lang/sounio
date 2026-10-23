#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
expected_materializer="e62c0c639ceece49823240565c4c2bd90fe2a766d98f059a9c59f9965f8c21d2"
expected_semantics="100404ef5ea29c6d7fb945bfca3fb2433eb2f88aece42d6f5ef8e6b9067c326e"
[[ "$(sha256sum self-hosted/native/pireus_xor_materializer.sio | cut -d' ' -f1)" == "$expected_materializer" ]]
[[ "$(sha256sum tools/pireus/xor_basis4_semantics.values.v1 | cut -d' ' -f1)" == "$expected_semantics" ]]
work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-xeon-material.XXXXXX")"
trap 'rm -rf "$work"' EXIT

SOUNIO_SOUC_ENGINE=lean_single bin/souc run \
  self-hosted/native/pireus_xor_materializer.sio >"$work/materializer.log"
grep -q '^PIREUS_XEON_MATERIALIZER_PASS$' "$work/materializer.log"
grep -q '^bridge_negative_tests=4/4$' "$work/materializer.log"
grep -q '^negative_pairs=120 signed_lane_checksum=21336$' "$work/materializer.log"
sed -n '/^---BEGIN-ASSEMBLY---$/,/^---END-ASSEMBLY---$/p' "$work/materializer.log" | \
  sed '1d;$d' >"$work/pireus_xor_avx512.S"
cc -c "$work/pireus_xor_avx512.S" -o "$work/pireus_xor_avx512.o"
c++ -std=c++20 -O2 tools/pireus/xeon_avx512_xor_material_harness.cpp \
  "$work/pireus_xor_avx512.o" -o "$work/pireus_xor_material"
objdump -d "$work/pireus_xor_material" >"$work/disassembly.txt"
grep -q 'vpermpd' "$work/disassembly.txt"
printf 'PIREUS_XEON_AVX512_MATERIAL_GATE_PASS binary=%s assembly=%s\n' \
  "$work/pireus_xor_material" "$work/pireus_xor_avx512.S"
