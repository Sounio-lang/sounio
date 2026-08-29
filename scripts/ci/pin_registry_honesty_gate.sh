#!/usr/bin/env bash
# Guards against pin_registry_ready / ffi_pinning_model regressing back to
# `true` under self-hosted/native without a live pin writer.
#
# Measured 2026-08-18: runtime_context.pin_count is never incremented on the
# active Madaros emitter (codegen_x86_linux.sio). Advertising
# pin_registry_ready:true made external probes treat pin=0 as "zero live pins"
# rather than "unwired" — the same honesty class as silent GUM var=0 (#1792)
# and precise_stack_maps:true before the stack-map honesty gate.
#
# `true` is legitimate again once a reclaim/pin lane actually increments
# runtime_context.pin_count (and drops the UNWIRED sentinel). Flip this gate
# in the same commit as that wiring.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PATTERN='pin_registry_ready",[[:space:]]*true|ffi_pinning_model",[[:space:]]*true'
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if grep -RIn --include='*.sio' -E "$PATTERN" self-hosted/native >"$TMP"; then
  echo "pin_registry_ready/ffi_pinning_model claimed true under self-hosted/native:"
  cat "$TMP"
  echo
  echo "These fields are currently only honest as false: the live x86_linux" >&2
  echo "emitter never increments runtime_context.pin_count (UNWIRED sentinel)." >&2
  echo "If this change wires real pinning, update this gate in the same commit." >&2
  exit 1
fi

echo "pin_registry honesty check passed: no self-hosted/native site claims pin ready."
