#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:?pass native Madaros}"
OUT="${2:?pass output directory}"
mkdir -p "$OUT"
cd "$ROOT"
source scripts/lib/souc_invoke.sh
_souc_limits
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/native_env_refresh_probe.cpp -o "$OUT/generate"
"$OUT/generate" "$ROOT" self-hosted/native/codegen_x86_linux.sio "$OUT/probe.sio"
target=x86_64-linux
if [[ "$(uname -s)" = Darwin ]]; then target=aarch64-macos; fi
"$HOST" build "$OUT/probe.sio" --target "$target" -o "$OUT/probe" >"$OUT/build.log" 2>&1
chmod +x "$OUT/probe"
if [[ "$(uname -s)" = Darwin ]]; then codesign -s - -f "$OUT/probe"; fi
off_status=0
env -u SOUNIO_RGN_BARRIER_SABOTAGE -u SOUNIO_WIDE_MUL_SABOTAGE -u SOUNIO_SABOTAGE_IEEE_F64_SPECIAL -u SOUNIO_SABOTAGE_DYNLINK_THUNK_CALLRET PROBE_ON=0 "$OUT/probe" >"$OUT/off.log" 2>&1 || off_status=$?
cat "$OUT/off.log"
if [[ "$off_status" != 0 ]]; then exit "$off_status"; fi
on_status=0
env PROBE_ON=1 SOUNIO_RGN_BARRIER_SABOTAGE=global SOUNIO_WIDE_MUL_SABOTAGE=1 SOUNIO_SABOTAGE_IEEE_F64_SPECIAL=1 SOUNIO_SABOTAGE_DYNLINK_THUNK_CALLRET=1 "$OUT/probe" >"$OUT/on.log" 2>&1 || on_status=$?
cat "$OUT/on.log"
if [[ "$on_status" != 0 ]]; then exit "$on_status"; fi
grep -Fxq NATIVE_ENV_REFRESH_OK "$OUT/off.log"
grep -Fxq NATIVE_ENV_REFRESH_OK "$OUT/on.log"
echo 'PASS: production diagnostic flags retain values with four reads across 256 refresh cycles'
