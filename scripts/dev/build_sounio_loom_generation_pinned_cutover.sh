#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="${SOUNIO_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_GENERATION_PINNED_CUTOVER_SOUC:-$ROOT/bin/souc}"
ENGINE="${SOUNIO_LOOM_GENERATION_PINNED_CUTOVER_ENGINE:-lean_single}"
MODULE="$ROOT/stdlib/coordination/loom_generation_pinned_cutover_authority.sio"
MAIN="$ROOT/tools/loom/generation_pinned_cutover_authority_main.sio"
OUTPUT="${SOUNIO_LOOM_GENERATION_PINNED_CUTOVER_OUTPUT:-$ROOT/tools/loom/_build/default/src/sounio-loom-generation-pinned-cutover}"
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-generation-pin.XXXXXX")"
trap 'rm -rf "$work"' EXIT
sed -n '1,$p' "$MODULE" "$MAIN" > "$work/action-9048.sio"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$work/action-9048.sio" -o "$work/runtime"
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$work/runtime" "$OUTPUT"
[[ "$(printf '0\n' | "$OUTPUT")" == 'SOUNIO_GENERATION_PINNED_CUTOVER_SELFTEST PASS cases=16' ]]
printf 'BUILT_GENERATION_PINNED_CUTOVER path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9048 engine=%s cases=16\n' "$OUTPUT" "$ENGINE"
