#!/usr/bin/env bash
# Prove runtime values and trace semantics before asserting the allocation read bound.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
host=${1:?source-built Madaros}
out_root=${2:?evidence directory}
mkdir -p "$out_root"
out=$(mktemp -d "$out_root/run.XXXXXX")
source scripts/lib/souc_invoke.sh
_souc_limits
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE=""
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/box_env_once_report.cpp -o "$out/report"
"$out/report" --self-test
for mode in 0 1 unset; do
  if test "$mode" = unset; then env_mode=(-u SOUNIO_BOX_TRACE); else env_mode=("SOUNIO_BOX_TRACE=$mode"); fi
  env "${env_mode[@]}" SOUNIO_INTO_ACC_ALLOC_TRACE=1 "$host" build \
    tests/run-pass/madaros_box_env_values.sio --target x86_64-linux -o "$out/values-$mode" >"$out/build-$mode.log" 2>&1
  chmod +x "$out/values-$mode"
  "$out/values-$mode" >"$out/values-$mode.stdout"
  grep -Fxq BOX_ENV_VALUES_OK "$out/values-$mode.stdout"
  if test "$mode" = 1; then
    grep -q '^boxtrace: READ field=' "$out/build-$mode.log"
  elif grep -q '^boxtrace:' "$out/build-$mode.log"; then
    echo 'FAIL: disabled BOX trace emitted diagnostics' >&2; exit 1
  fi
done
echo BOX_ENV_SEMANTICS_OK
for mode in 0 1 unset; do "$out/report" "$out/build-$mode.log"; done
echo BOX_ENV_ONCE_OK
