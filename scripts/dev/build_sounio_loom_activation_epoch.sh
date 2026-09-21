#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="${SOUNIO_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_ACTIVATION_EPOCH_SOUC:-$ROOT/bin/souc}"
ENGINE="${SOUNIO_LOOM_ACTIVATION_EPOCH_ENGINE:-lean_single}"
OUTPUT="${SOUNIO_LOOM_ACTIVATION_EPOCH_OUTPUT:-$ROOT/tools/loom/_build/default/src/sounio-loom-activation-epoch}"
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-activation-epoch.XXXXXX")"
trap 'rm -rf "$work"' EXIT
sed -n '1,$p' "$ROOT/stdlib/coordination/loom_activation_epoch_authority.sio" \
  "$ROOT/tools/loom/activation_epoch_authority_main.sio" >"$work/action-9049.sio"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$work/action-9049.sio" -o "$work/runtime"
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$work/runtime" "$OUTPUT"
[[ "$(printf '0\n' | "$OUTPUT")" == 'SOUNIO_ACTIVATION_EPOCH_SELFTEST PASS cases=13' ]]
printf 'BUILT_ACTIVATION_EPOCH path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9049 engine=%s cases=13\n' "$OUTPUT" "$ENGINE"
