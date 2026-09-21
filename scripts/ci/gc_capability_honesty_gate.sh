#!/usr/bin/env bash
# Guards against GC capability contract fields regressing to `true` under
# self-hosted/native without a live collector / retry / root-scan path.
#
# Measured 2026-08-18 (RUNTIME_CONTEXT_UNWRITTEN_FIELDS_CENSUS): the live
# Madaros emitter (codegen_x86_linux.sio) fail-closes at handle exhaustion
# and never calls empty_frame_reset / tracing GC. Advertising
# tracing_gc / gc_mark_compact_model / … as true is the same honesty class
# as pin_registry_ready:true before #1830 and precise_stack_maps:true before
# its honesty gate.
#
# Flip these back to true only in the same commit that wires the mechanism.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PATTERN='tracing_gc:[[:space:]]*true|gc_mark_compact_model",[[:space:]]*true|gc_precise_descriptor_scanning",[[:space:]]*true|gc_handle_relocation_model",[[:space:]]*true|gc_runtime_retry_active",[[:space:]]*true|gc_runtime_retry_active",[[:space:]]*runtime_metadata_active|gc_current_frame_root_scan",[[:space:]]*true|gc_current_frame_root_scan",[[:space:]]*runtime_metadata_active'
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if grep -RIn --include='*.sio' -E "$PATTERN" self-hosted/native >"$TMP"; then
  echo "GC capability claimed true / metadata-tied under self-hosted/native:"
  cat "$TMP"
  echo
  echo "Live x86_linux path is bump-alloc + fail-closed 182; empty_frame_reset" >&2
  echo "is defined but never called. These flags are only honest as false until" >&2
  echo "a reclaim lane wires a real collector. Update this gate in that commit." >&2
  exit 1
fi

echo "gc_capability honesty check passed: no false collector advertisements."
