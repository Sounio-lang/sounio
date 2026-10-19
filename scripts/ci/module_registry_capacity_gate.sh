#!/usr/bin/env bash
# Executes extracted production registry functions, including the refusal edge.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPILER="${1:?pass a native raw Madaros compiler}"
case "$COMPILER" in /*) ;; *) COMPILER="$PWD/$COMPILER" ;; esac
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/module-registry-capacity.XXXXXX")}"
mkdir -p "$WORK"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
source "$ROOT/scripts/lib/souc_invoke.sh"
_souc_limits
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror \
  scripts/ci/module_registry_capacity_probe.cpp -o "$WORK/generate"
"$WORK/generate" "$ROOT" "$WORK/probe.sio"
"$COMPILER" check "$WORK/probe.sio" > "$WORK/check.log" 2>&1
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) target=aarch64-macos ;;
  Linux/x86_64) target=x86_64-linux ;;
  *) echo 'unsupported native registry probe host' >&2; exit 2 ;;
esac
"$COMPILER" build "$WORK/probe.sio" --target "$target" -o "$WORK/probe" > "$WORK/build.log" 2>&1
chmod +x "$WORK/probe"
if [[ "$target" = aarch64-macos ]]; then
  codesign -s - -f "$WORK/probe"
  codesign --verify "$WORK/probe"
fi
timeout 60 "$WORK/probe" > "$WORK/stdout.log" 2> "$WORK/stderr.log"
grep -Fxq 'registry_capacity_ok: collision, isolation, duplicate, 4096/4097 refusal' "$WORK/stdout.log"
echo 'PASS: production module registries preserve scoped identity and refuse beyond 4096'
