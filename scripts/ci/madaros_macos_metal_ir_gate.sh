#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${1:?pass the current-source native bootstrap seed}"
case "$SEED" in /*) ;; *) SEED="$PWD/$SEED" ;; esac
OUT="${2:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-metal-ir.XXXXXX")}"
mkdir -p "$OUT"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_METAL_RUNTIME="$ROOT/artifacts/self-hosted/libsounio_metal.dylib"
export SOUNIO_METAL_TRACE=1
bash scripts/dev/build_native_metal_runtime.sh "$SOUNIO_METAL_RUNTIME" > "$OUT/runtime-build.log" 2>&1
"$SEED" tests/native/arm64_macos_metal_ir.sio "$OUT/driver" --target aarch64-macos > "$OUT/driver-build.log" 2>&1
chmod +x "$OUT/driver"
codesign -s - -f "$OUT/driver"
timeout 30 "$OUT/driver" "$OUT/program" > "$OUT/emit.log" 2>&1
file "$OUT/program" | grep -Eq 'Mach-O 64-bit.*arm64'
chmod +x "$OUT/program"
codesign -s - -f "$OUT/program"
codesign --verify "$OUT/program"
timeout 30 "$OUT/program" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
test "$(<"$OUT/stdout.log")" = '3.750000'
grep -Fq 'SOUNIO_METAL_EXECUTED device=' "$OUT/stderr.log"
grep -Fq 'kernel=sounio_kernel_vec_add_in_place lanes=4 policy=software_binary64' "$OUT/stderr.log"
echo "PASS: emitted Mach-O executed a parsed Sounio kernel on Metal; artifacts=$OUT"
