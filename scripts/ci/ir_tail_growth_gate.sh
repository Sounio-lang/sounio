#!/usr/bin/env bash
# Production allocator and ARM64 address-window discriminators, without shrinking pools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:?pass native raw Madaros}"
OUT="${2:?pass new output directory}"
mkdir -p "$OUT"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
source "$ROOT/scripts/lib/souc_invoke.sh"
_souc_limits
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/ir_tail_growth_probe.cpp -o "$OUT/generate"
"$OUT/generate" "$ROOT" "$OUT/tail.sio" "$OUT/address.sio"
case "$(uname -s)/$(uname -m)" in
 Darwin/arm64) target=aarch64-macos ;;
 Linux/x86_64) target=x86_64-linux ;;
 *) echo 'unsupported native host' >&2; exit 2 ;;
esac
for probe in tail address; do
 "$HOST" check "$OUT/$probe.sio" >"$OUT/$probe.check.log" 2>&1
 "$HOST" build "$OUT/$probe.sio" --target "$target" -o "$OUT/$probe" >"$OUT/$probe.build.log" 2>&1
 chmod +x "$OUT/$probe"
 if [[ "$target" = aarch64-macos ]]; then codesign -s - -f "$OUT/$probe"; codesign --verify "$OUT/$probe"; fi
 timeout 90 "$OUT/$probe" >"$OUT/$probe.stdout.log" 2>"$OUT/$probe.stderr.log"
done
grep -Fxq IR_TAIL_GROWTH_OK "$OUT/tail.stdout.log"
grep -Fxq ARM64_DATA_ADDRESS_OK "$OUT/address.stdout.log"
echo 'PASS: production tail growth, non-tail relocation, generation/sealing/refusal and ARM64 address encoding/window'
