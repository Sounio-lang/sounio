#!/usr/bin/env bash
# Verify that Visual IR HTML/SVG exports are passive markup.
#
# This runs the real Sounio programs that emit HTML/SVG, checks for expected
# audit/identity markers, and rejects active browser semantics.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUC_NATIVE_WRAPPER:-${ROOT_DIR}/scripts/ci/souc-native-wrapper.sh}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-${ROOT_DIR}/stdlib}"

TMP_DIR="$(mktemp -d /tmp/sounio-viz-html-passive.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

FORBIDDEN_REGEX=(
  '<[[:space:]]*script'
  '<[[:space:]]*foreignobject'
  '<[[:space:]]*iframe'
  '<[[:space:]]*object'
  '<[[:space:]]*embed'
  'javascript[[:space:]]*:'
  '[[:space:]]on[a-z0-9_-]+[[:space:]]*='
  '[[:space:]]href[[:space:]]*='
)

require_text() {
  local file="$1"
  local needle="$2"
  if ! grep -qF "$needle" "$file"; then
    echo "error: ${file}: missing required passive export marker: ${needle}" >&2
    tail -n 40 "$file" >&2 || true
    exit 1
  fi
}

reject_active_tokens() {
  local file="$1"
  local pattern
  for pattern in "${FORBIDDEN_REGEX[@]}"; do
    if grep -Eiq "$pattern" "$file"; then
      echo "error: ${file}: active HTML/SVG token matched forbidden pattern: ${pattern}" >&2
      grep -Ein "$pattern" "$file" >&2 || true
      exit 1
    fi
  done
}

run_case() {
  local label="$1"
  local src="$2"
  local marker="$3"
  local out="${TMP_DIR}/${label}.out"

  echo "[viz-html-passive] run ${src}"
  "$SOUC" run "${ROOT_DIR}/${src}" >"$out"

  require_text "$out" "$marker"
  require_text "$out" "<svg"
  require_text "$out" "data-viz-id="
  require_text "$out" "data-viz-kind="
  require_text "$out" "viz-audit"
  reject_active_tokens "$out"
  echo "[viz-html-passive] PASS ${label}"
}

run_case "viz_html_static" "tests/run-pass/viz_html_static.sio" "VIZ_HTML_STATIC_PASS"
run_case "viz_html_identity" "tests/run-pass/viz_html_identity.sio" "VIZ_HTML_IDENTITY_PASS"
run_case "viz_workbench_html_physchem" "tests/run-pass/viz_workbench_html_physchem.sio" "VIZ_WORKBENCH_HTML_PHYSCHEM_PASS"
run_case "viz_workbench_replay_html_archive" "tests/run-pass/viz_workbench_replay_html_archive.sio" "VIZ_WORKBENCH_REPLAY_HTML_ARCHIVE_PASS"

echo "VIZ_HTML_PASSIVE_EXPORT_PASS"
