#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP:-0}"

fail() {
  echo "[madaros-f64-lowering] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-f64-lowering.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-$WORK/madaros}"
BUILD_RECEIPT="${SOUNIO_MADAROS_F64_LOWERING_GATE_RECEIPT:-}"
AUTHORITY="external-unbound"
MERGE_READY=0

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-}" ]]; then
  BUILD_RECEIPT="${BUILD_RECEIPT:-$WORK/madaros-build-receipt.tsv}"
  if ! SOUNIO_MADAROS_BUILD_RECEIPT="$BUILD_RECEIPT" \
      bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
  [[ -s "$BUILD_RECEIPT" ]] || fail "current-source Madaros build receipt is missing: $BUILD_RECEIPT"
  AUTHORITY="locally-built-content-bound"
else
  if [[ -n "$BUILD_RECEIPT" ]]; then
    [[ -s "$BUILD_RECEIPT" ]] || fail "external build receipt is missing: $BUILD_RECEIPT"
  else
    BUILD_RECEIPT="none"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

SOUNIO_MADAROS_DEREF_F64_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_DEREF_F64_GATE_DIR="$WORK/deref" \
SOUNIO_MADAROS_DEREF_F64_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_deref_f64_array_gate.sh"

SOUNIO_MADAROS_GLOBAL_F64_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_GLOBAL_F64_GATE_DIR="$WORK/global" \
SOUNIO_MADAROS_GLOBAL_F64_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_global_f64_scratch_gate.sh"

SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_DIR="$WORK/global-capacity" \
SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_global_capacity_gate.sh"

SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_DIR="$WORK/imported-capacity" \
SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_capacity_gate.sh"

echo "[madaros-f64-lowering] PASS: one shared Madaros ELF passed dereference, global f64, direct capacity, and imported capacity gates authority=$AUTHORITY merge_ready=$MERGE_READY receipt=$BUILD_RECEIPT"
