#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:?pass current native Madaros}"
SEED="${2:?pass source-tracking seed}"
OUT="${3:?pass evidence directory}"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]]
mkdir -p "$OUT"
OUT="$(mktemp -d "$OUT/run.XXXXXX")"
cd "$ROOT"
source scripts/lib/souc_invoke.sh
_souc_limits
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/bss_madvise_deny.cpp -o "$OUT/deny"
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/bss_madvise_witness.cpp -o "$OUT/witness"
"$SEED" tests/run-pass/madaros_global_zero_initializer_order.sio "$OUT/order" >"$OUT/order.build.log" 2>&1
"$HOST" build tests/run-pass/madaros_bss_zero_pages.sio --target x86_64-linux -o "$OUT/pages" >"$OUT/pages.build.log" 2>&1
for program in order pages; do
    chmod +x "$OUT/$program"
    "$OUT/$program" >"$OUT/$program.stdout.log" 2>&1
    "$OUT/witness" "$OUT/deny" "$OUT/$program" >"$OUT/$program.fallback.log" 2>&1
    cat "$OUT/$program.stdout.log" "$OUT/$program.fallback.log"
done
grep -Fxq GLOBAL_ZERO_ORDER_OK "$OUT/order.stdout.log"
grep -Fxq GLOBAL_ZERO_ORDER_OK "$OUT/order.fallback.log"
grep -Fxq BSS_ZERO_PAGES_OK "$OUT/pages.stdout.log"
grep -Fxq BSS_ZERO_PAGES_OK "$OUT/pages.fallback.log"
echo 'PASS: seed initializer order and modular BSS pages preserve values under observed madvise denial'
