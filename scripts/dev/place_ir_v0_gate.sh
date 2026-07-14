#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/lib/resolve_souc.sh"

MODEL="$ROOT/self-hosted/ir/place_v0.sio"
PROBE="$ROOT/self-hosted/ir/place_v0_probe.sio"
SPEC="$ROOT/docs/internal/compiler/PLACE_IR_V0.md"

sounio_require_souc

for path in "$MODEL" "$PROBE" "$SPEC"; do
  if [[ ! -f "$path" ]]; then
    echo "PLACE_IR_V0_BLOCKED missing=$path"
    exit 2
  fi
done

# Static authority boundary: the shadow module must not enter the default IR,
# lowerer, compiler driver, or backend import graph in this phase.
if rg -n 'use ir::place_v0' \
    "$ROOT/self-hosted/ir/mod.sio" \
    "$ROOT/self-hosted/ir/lower.sio" \
    "$ROOT/self-hosted/compiler" \
    "$ROOT/self-hosted/native" \
    "$ROOT/self-hosted/wasm" \
    "$ROOT/self-hosted/vm" >/tmp/place-ir-v0-imports.$$ 2>/dev/null; then
  echo "PLACE_IR_V0_FAIL reason=default_pipeline_import"
  cat /tmp/place-ir-v0-imports.$$
  rm -f /tmp/place-ir-v0-imports.$$
  exit 1
fi
rm -f /tmp/place-ir-v0-imports.$$

rg -q '^Status: experimental differential shadow' "$SPEC"
rg -q '^Integration: disabled' "$SPEC"
rg -q 'Legacy path: retained as the differential oracle' "$SPEC"
rg -q 'Default pipeline integration: \*\*NOT DONE\*\*' "$SPEC"
rg -q 'Differential pipeline integration and program-level parity runner: \*\*NOT DONE\*\*' "$SPEC"
rg -q 'pub fn place_v0_default_pipeline_enabled\(\) -> bool' "$MODEL"
rg -q 'pub fn place_v0_legacy_field_label7_status\(\) -> i64' "$MODEL"
rg -q 'pub fn place_v0_legacy_field_valid_codegen_status\(\) -> i64' "$MODEL"
rg -q 'PLACE_V0_ERR_DIFFERENTIAL_ONLY' "$MODEL"

"$SOUC_BIN" check "$MODEL"
"$SOUC_BIN" check "$PROBE"

probe_output="$("$SOUC_BIN" run "$PROBE")"
printf '%s\n' "$probe_output"
if ! grep -q '^PLACE_IR_V0_PASS tests=12 mode=differential-shadow default_pipeline=off$' <<<"$probe_output"; then
  echo "PLACE_IR_V0_FAIL reason=unexpected_probe_receipt"
  exit 1
fi

echo "PLACE_IR_V0_GREEN checker=model,probe truth_table=12 default_pipeline=off legacy_oracle=kept"
