#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

# The PETAB baseline and PBPK/GUM workflow import stdlib + packages/* modules.
# lean_single resolves those imports and enforces the full effect/ontology
# surface; the Madaros default (bin/souc since 2026-06-14) does not yet — it
# fails this workflow with a spurious "effect not declared (missing: GPU)"
# during multimodule thin-link. Pin the raw ELF to lean_single, matching the
# package import sub-gate (package_import_science_gate.sh). The raw lean_single
# ELF only accepts the positional `SRC OUT` CLI, so route the souc `run`/`compile`
# verbs this gate uses through souc-native-wrapper.sh. Override via SOUNIO_SOUC_BIN.
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
OUT_DIR="$(mktemp -d /tmp/sounio-package-pbpk-gum.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

printf '[package-pbpk-gum] souc=%s\n' "$SOUC_BIN"
printf '[package-pbpk-gum] stdlib=%s\n' "$SOUNIO_STDLIB_PATH"

printf '[package-pbpk-gum] run package import contract gate\n'
bash "$ROOT_DIR/scripts/ci/package_import_science_gate.sh" >"$OUT_DIR/package_import_science.log" 2>&1
cat "$OUT_DIR/package_import_science.log"

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
