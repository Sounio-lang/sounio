#!/usr/bin/env bash
# Native production-container boundary plus actual compiler pressure/refusal.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:?pass native raw Madaros}"
OUT="${2:?pass new output directory}"
mkdir -p "$OUT"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
# Match the documented raw-compiler invocation contract, not Linux's 8 MiB
# shell default. The public wrapper and self-compile ladder reserve stack too.
source "$ROOT/scripts/lib/souc_invoke.sh"
_souc_limits
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/type_env_capacity_probe.cpp -o "$OUT/generate"
"$OUT/generate" "$ROOT" "$OUT"
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) target=aarch64-macos ;;
  Linux/x86_64) target=x86_64-linux ;;
  *) echo 'unsupported native host' >&2; exit 2 ;;
esac
"$HOST" check "$OUT/probe.sio" >"$OUT/check.log" 2>&1
"$HOST" build "$OUT/probe.sio" --target "$target" -o "$OUT/probe" >"$OUT/build.log" 2>&1
chmod +x "$OUT/probe"
if [[ "$target" = aarch64-macos ]]; then codesign -s - -f "$OUT/probe"; codesign --verify "$OUT/probe"; fi
timeout 90 "$OUT/probe" >"$OUT/stdout.log" 2>"$OUT/stderr.log"
grep -Fxq TYPE_ENV_CAPACITY_OK "$OUT/stdout.log"
"$HOST" check "$OUT/pressure.sio" >"$OUT/pressure.log" 2>&1
"$HOST" check "$OUT/global-boundary.sio" >"$OUT/global-boundary.log" 2>&1
set +e
"$HOST" check "$OUT/overflow.sio" >"$OUT/overflow-check.log" 2>&1
check_status=$?
"$HOST" build "$OUT/overflow.sio" --target "$target" -o "$OUT/must-not-exist" >"$OUT/overflow-build.log" 2>&1
build_status=$?
set -e
test "$check_status" != 0
test "$build_status" != 0
test ! -e "$OUT/must-not-exist"
grep -q 'error\[E-TYPE-ENV-CAP\]' "$OUT/overflow-check.log"
grep -q 'error\[E-TYPE-ENV-CAP\]' "$OUT/overflow-build.log"
echo 'PASS: TypeEnv scope isolation, exact capacity, sticky refusal, fresh reset, global/local pressure and no overflow artifact'
