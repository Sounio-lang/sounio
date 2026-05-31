#!/usr/bin/env bash
# scripts/ci/native_sass_encode_gate.sh
#
# Track-3 native SASS encoder gate.
#
# Proves the Kretikos native SASS encoder (self-hosted/gpu/kretikos_sass_encode.sio)
# regenerates the known-good gpu_bare_sm80_vec_add_f64 .text section BYTE-FOR-BYTE
# from K-AXI-driven instruction fields. Byte-identity = encoder correctness
# (the GPU analog of souc's self-reproducing fixed point).
#
# PASS  -> selftest compiles and exits 0 (512/512 .text bytes match)
# FAIL  -> compile error or byte mismatch

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${ROOT}/bin/souc"
GPU="${ROOT}/self-hosted/gpu"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

echo "[native-sass-gate] compiling encoder library ..."
"${SOUC}" "${GPU}/kretikos_sass_encode.sio" "${OUT}/enc.elf" >/dev/null 2>"${OUT}/enc.err" || {
    echo "[native-sass-gate] FAIL: encoder library did not compile" >&2
    grep -iE '^error|typecheck' "${OUT}/enc.err" >&2 || true
    exit 1
}

echo "[native-sass-gate] compiling + running byte-identity selftest ..."
"${SOUC}" "${GPU}/kretikos_sass_selftest.sio" "${OUT}/selftest.elf" >/dev/null 2>"${OUT}/st.err" || {
    echo "[native-sass-gate] FAIL: selftest did not compile" >&2
    grep -iE '^error|typecheck' "${OUT}/st.err" >&2 || true
    exit 1
}
chmod +x "${OUT}/selftest.elf"

if "${OUT}/selftest.elf"; then
    echo "[native-sass-gate] PASS: sm_80 vec_add_f64 .text reproduced byte-identical"
    exit 0
else
    echo "[native-sass-gate] FAIL: byte mismatch (encoder != known-good blob)" >&2
    exit 1
fi
