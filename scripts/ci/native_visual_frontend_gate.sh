#!/usr/bin/env bash
# scripts/ci/native_visual_frontend_gate.sh
#
# Headless acceptance gate for Sounio's native Visual IR/frontend lane.
# This keeps the scientific/interaction semantics in Sounio data and checks
# Canvas, HTML/SVG, Workbench replay, physchem, and demo compile surfaces.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUC_NATIVE_WRAPPER:-${ROOT_DIR}/scripts/ci/souc-native-wrapper.sh}"
MANIFEST="${NATIVE_VIZ_GATE_MANIFEST:-${ROOT_DIR}/scripts/ci/native_visual_frontend_gate.manifest.tsv}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-${ROOT_DIR}/stdlib}"

TMP_DIR="$(mktemp -d /tmp/sounio-viz-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

run_expect() {
  local label="$1"
  local src="$2"
  local marker="$3"
  local out="${TMP_DIR}/${label}.out"
  echo "[native-viz-gate] run ${src}"
  "$SOUC" run "${ROOT_DIR}/${src}" >"$out"
  if ! grep -qF "$marker" "$out"; then
    echo "error: ${label}: missing marker ${marker}" >&2
    echo "--- ${label} output tail ---" >&2
    tail -n 40 "$out" >&2 || true
    exit 1
  fi
  echo "[native-viz-gate] PASS ${label}"
}

check_file() {
  local label="$1"
  local src="$2"
  echo "[native-viz-gate] check ${src}"
  "$SOUC" check "${ROOT_DIR}/${src}" >/tmp/sounio-viz-gate-check.out
  echo "[native-viz-gate] PASS ${label}"
}

compile_file() {
  local label="$1"
  local src="$2"
  local out="${TMP_DIR}/${label}.elf"
  echo "[native-viz-gate] compile ${src}"
  "$SOUC" compile "${ROOT_DIR}/${src}" -o "$out" >/tmp/sounio-viz-gate-compile.out
  test -s "$out"
  echo "[native-viz-gate] PASS ${label}"
}

script_file() {
  local label="$1"
  local src="$2"
  echo "[native-viz-gate] script ${src}"
  "${ROOT_DIR}/${src}"
  echo "[native-viz-gate] PASS ${label}"
}

echo "=== Native Visual Frontend Gate ==="
echo "[native-viz-gate] souc=${SOUC}"
echo "[native-viz-gate] stdlib=${SOUNIO_STDLIB_PATH}"
echo "[native-viz-gate] manifest=${MANIFEST}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "error: native visual frontend manifest missing: ${MANIFEST}" >&2
  exit 1
fi

python3 "${ROOT_DIR}/scripts/ci/check_native_visual_frontend_gate_manifest.py" \
  --root "$ROOT_DIR" \
  --manifest "$MANIFEST"

python3 "${ROOT_DIR}/scripts/ci/check_native_visual_frontend_plan_coverage.py" \
  --root "$ROOT_DIR" \
  --manifest "$MANIFEST"

while IFS=$'\t' read -r mode label path marker; do
  [[ -z "${mode}" || "${mode}" == \#* ]] && continue
  case "$mode" in
    check)
      check_file "$label" "$path"
      ;;
    compile)
      compile_file "$label" "$path"
      ;;
    run)
      run_expect "$label" "$path" "$marker"
      ;;
    script)
      script_file "$label" "$path"
      ;;
    *)
      echo "error: ${MANIFEST}: unknown mode '${mode}' for ${label}" >&2
      exit 1
      ;;
  esac
done <"$MANIFEST"

echo "=== Native Visual Frontend Gate: PASS ==="
