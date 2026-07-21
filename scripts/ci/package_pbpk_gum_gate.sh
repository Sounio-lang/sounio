#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# The PETAB baseline and PBPK/GUM workflow import stdlib + packages/* modules.
# Madaros owns package-import acceptance plus a focused PBPK/GUM runtime proof
# that also exercises bounded CSV string slicing. The two large PBPK witnesses
# cross the unresolved program layout-catalog boundary, so retain the bootstrap
# compiler only for those named witnesses until that parity blocker is closed.
OUT_DIR="$(mktemp -d /tmp/sounio-package-pbpk-gum.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

RAW_MADAROS="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW_MADAROS" ]]; then
  RAW_MADAROS="$OUT_DIR/madaros-current-source"
  bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW_MADAROS"
fi
if [[ ! -x "$RAW_MADAROS" || "$(head -c 2 "$RAW_MADAROS" 2>/dev/null)" == '#!' ]]; then
  echo '[package-pbpk-gum] FAIL: current-source raw Madaros ELF unavailable' >&2
  exit 1
fi
if ! "$RAW_MADAROS" --version | grep -qF 'Madaros'; then
  echo '[package-pbpk-gum] FAIL: compiler does not identify as Madaros' >&2
  exit 1
fi

LEAN_SOUC="${SOUNIO_SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
if [[ ! -x "$LEAN_SOUC" ]]; then
  LEAN_SOUC="$ROOT_DIR/bin/souc-linux-x86_64"
fi
if [[ ! -x "$LEAN_SOUC" ]]; then
  echo '[package-pbpk-gum] FAIL: lean_single compiler ELF not found' >&2
  exit 1
fi
export SOUNIO_SOUC_BIN="$LEAN_SOUC"
SOUC_BIN="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

BASELINE="tests/stdlib/darwin_pbpk/test_observed_petab_fit_e2e.sio"
WORKFLOW="tests/packages/package_pbpk_gum_workflow.sio"
MEASUREMENT_PROBE="tests/packages/package_pbpk_gum_measurement_probe.sio"

printf '[package-pbpk-gum] package-acceptance-madaros=%s\n' "$RAW_MADAROS"
printf '[package-pbpk-gum] legacy-runtime-souc=%s\n' "$SOUC_BIN"
printf '[package-pbpk-gum] stdlib=%s\n' "$SOUNIO_STDLIB_PATH"

printf '[package-pbpk-gum] run package import contract gate\n'
MADAROS_RAW_BIN="$RAW_MADAROS" \
  bash "$ROOT_DIR/scripts/ci/package_import_science_gate.sh" >"$OUT_DIR/package_import_science.log" 2>&1
cat "$OUT_DIR/package_import_science.log"

run_madaros_witness() {
  local source="$1"
  local marker="$2"
  local stem="$3"
  local elf="$OUT_DIR/$stem.elf"
  local log="$OUT_DIR/$stem.log"

  printf '[package-pbpk-gum] run focused Madaros witness %s\n' "$source"
  if ! "$RAW_MADAROS" --native-compile "$source" -o "$elf" >"$log" 2>&1; then
    cat "$log" >&2
    echo "[package-pbpk-gum] FAIL: Madaros could not compile $source" >&2
    exit 1
  fi
  chmod +x "$elf"
  if ! "$elf" >>"$log" 2>&1; then
    cat "$log" >&2
    echo "[package-pbpk-gum] FAIL: Madaros witness failed: $source" >&2
    exit 1
  fi
  if ! grep -qF "$marker" "$log"; then
    cat "$log" >&2
    echo "[package-pbpk-gum] FAIL: Madaros marker missing: $marker" >&2
    exit 1
  fi
}

run_madaros_witness "$MEASUREMENT_PROBE" 'PACKAGE_PBPK_GUM_MEASUREMENT_OK' 'measurement_probe'

printf '[package-pbpk-gum] run canonical observed PETAB baseline %s\n' "$BASELINE"
if ! "$SOUC_BIN" run "$BASELINE" >"$OUT_DIR/observed_petab.log" 2>&1; then
  cat "$OUT_DIR/observed_petab.log" >&2
  echo '[package-pbpk-gum] FAIL: canonical observed PETAB baseline failed' >&2
  exit 1
fi
if ! grep -qF 'OBSERVED_PETAB_FIT_OK' "$OUT_DIR/observed_petab.log"; then
  cat "$OUT_DIR/observed_petab.log" >&2
  echo '[package-pbpk-gum] FAIL: canonical observed PETAB marker missing' >&2
  exit 1
fi

printf '[package-pbpk-gum] run package-backed PBPK/GUM workflow %s\n' "$WORKFLOW"
if ! "$SOUC_BIN" run "$WORKFLOW" >"$OUT_DIR/package_pbpk_gum.log" 2>&1; then
  cat "$OUT_DIR/package_pbpk_gum.log" >&2
  echo '[package-pbpk-gum] FAIL: package-backed PBPK/GUM workflow failed' >&2
  exit 1
fi
cat "$OUT_DIR/package_pbpk_gum.log"
if ! grep -qF 'PACKAGE_PBPK_GUM_OK' "$OUT_DIR/package_pbpk_gum.log"; then
  echo '[package-pbpk-gum] FAIL: package-backed PBPK/GUM marker missing' >&2
  exit 1
fi

echo '[package-pbpk-gum] PASS: PBPK/GUM consumes epistemic-core as a package'
