#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/artifacts/self-hosted/libsounio_metal.dylib}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-metal-runtime.XXXXXX")"
echo "artifacts=$WORK"
"${CXX:-/usr/bin/clang++}" -std=c++23 -Wall -Wextra -Werror "$ROOT/scripts/dev/embed_metal_softfloat.cpp" -o "$WORK/embed"
"$WORK/embed" "$ROOT/vendor/metal-softfloat/softfloat64.metal" > "$WORK/softfloat64_source.inc"
mkdir -p "$(dirname "$OUT")"
"${CXX:-/usr/bin/clang++}" -std=c++23 -O2 -fobjc-arc -Wall -Wextra -Werror \
    -Wno-deprecated-declarations -I"$WORK" -dynamiclib \
    "$ROOT/self-hosted/gpu/runtime/native_metal_bridge.mm" \
    -framework Foundation -framework Metal -o "$OUT"
codesign -s - -f "$OUT"
codesign --verify "$OUT"
echo "runtime=$OUT"
