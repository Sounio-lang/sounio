#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SUBCOMMAND_SOUC_BIN="$SOUC_BIN"
if [[ "$(head -c 2 "$SOUC_BIN" 2>/dev/null)" != "#!" ]]; then
  export SOUC_NATIVE_BIN="$SOUC_BIN"
  SUBCOMMAND_SOUC_BIN="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
  chmod +x "$SUBCOMMAND_SOUC_BIN"
fi

BASELINE="tests/stdlib/darwin_pbpk/test_observed_petab_fit_e2e.sio"
WORKFLOW="tests/packages/package_pbpk_gum_workflow.sio"
OUT_DIR="$(mktemp -d /tmp/sounio-package-pbpk-gum.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

printf '[package-pbpk-gum] souc=%s\n' "$SUBCOMMAND_SOUC_BIN"
printf '[package-pbpk-gum] stdlib=%s\n' "$SOUNIO_STDLIB_PATH"

printf '[package-pbpk-gum] run package import contract gate\n'
bash "$ROOT_DIR/scripts/ci/package_import_science_gate.sh" >"$OUT_DIR/package_import_science.log" 2>&1
cat "$OUT_DIR/package_import_science.log"

printf '[package-pbpk-gum] run canonical observed PETAB baseline %s\n' "$BASELINE"
if ! "$SUBCOMMAND_SOUC_BIN" run "$BASELINE" >"$OUT_DIR/observed_petab.log" 2>&1; then
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
if ! "$SUBCOMMAND_SOUC_BIN" run "$WORKFLOW" >"$OUT_DIR/package_pbpk_gum.log" 2>&1; then
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
