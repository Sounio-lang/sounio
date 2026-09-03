#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="${SOUNIO_WEBSITE_DOCS_SUPPORT_ARTIFACT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-website-docs-support.XXXXXX")}"
KEEP_ARTIFACT_ROOT="${SOUNIO_WEBSITE_DOCS_SUPPORT_KEEP_ARTIFACT_ROOT:-0}"

cleanup() {
  if [[ "$KEEP_ARTIFACT_ROOT" != "1" && -n "${ARTIFACT_ROOT:-}" ]]; then
    rm -rf "$ARTIFACT_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$ARTIFACT_ROOT"

run_step() {
  local label="$1"
  shift
  local log="$ARTIFACT_ROOT/$label.log"

  echo "[website-docs-support] >>> $label"
  if "$@" >"$log" 2>&1; then
    echo "[website-docs-support] <<< $label PASS log=$log"
  else
    local rc=$?
    echo "[website-docs-support] !!! $label FAIL rc=$rc log=$log" >&2
    tail -n 120 "$log" >&2 || true
    return "$rc"
  fi
}

echo "SOUNIO_WEBSITE_DOCS_SUPPORT_GATE_START"
echo "repo=$ROOT_DIR"
echo "artifact_root=$ARTIFACT_ROOT"

run_step docs-registry bash "$ROOT_DIR/scripts/dev/check_docs_registry.sh"
run_step docs-consistency bash "$ROOT_DIR/scripts/dev/check_docs_consistency.sh"
run_step website-i18n npm --prefix "$ROOT_DIR/website" run check:i18n
run_step website-docs-parity npm --prefix "$ROOT_DIR/website" run check:docs-parity
run_step website-render-assets env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  npm --prefix "$ROOT_DIR/website" run check:render-assets
run_step website-routes npm --prefix "$ROOT_DIR/website" run check:routes
run_step website-brand npm --prefix "$ROOT_DIR/website" run check:brand

if [[ "$KEEP_ARTIFACT_ROOT" == "1" ]]; then
  echo "SOUNIO_WEBSITE_DOCS_SUPPORT_ARTIFACT_ROOT=$ARTIFACT_ROOT"
fi
echo "SOUNIO_WEBSITE_DOCS_SUPPORT_GATE_PASS"
