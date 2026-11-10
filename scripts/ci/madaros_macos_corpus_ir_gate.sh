#!/usr/bin/env bash
# Isolated real-IR/runtime regression, independent of the full CLI rebuild.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${1:?pass a native current-source bootstrap seed}"
case "$SEED" in /*) ;; *) SEED="$PWD/$SEED" ;; esac
WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-corpus-ir.XXXXXX")"
echo "artifacts=$WORK"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
"$SEED" tests/native/arm64_macos_corpus_ir.sio "$WORK/driver" --target aarch64-macos >"$WORK/build.log" 2>&1
chmod +x "$WORK/driver"
codesign -s - -f "$WORK/driver"
"$WORK/driver" "$WORK/program" >"$WORK/emit.log"
test -s "$WORK/program"
file "$WORK/program" | grep -Eq 'Mach-O 64-bit.*arm64'
chmod +x "$WORK/program"
codesign -s - -f "$WORK/program"
codesign --verify "$WORK/program"
"$WORK/program" >"$WORK/actual.log"
printf '77\n77\n2.250000\n-2.250000\n1\n2.250000\n77\n123\n123456\n77\n7\n154\n154\n0.100000\n1\n' >"$WORK/expected.log"
"${CXX:-/usr/bin/clang++}" -std=c++23 "$ROOT/scripts/ci/arm64_macos_math_oracle.cpp" -o "$WORK/math-oracle"
"$WORK/math-oracle" >>"$WORK/expected.log"
printf '77\n' >>"$WORK/expected.log"
diff -u "$WORK/expected.log" "$WORK/actual.log"
echo 'PASS: ARM64 arrays/references, float/mixed arithmetic, indirect/stack/sret calls, globals, large file read, dyld/libSystem math imports'
