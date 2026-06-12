#!/usr/bin/env bash
# scripts/ci/native_visual_frontend_gate.sh
#
# Headless acceptance gate for Sounio's native Visual IR/frontend lane.
# This keeps the scientific/interaction semantics in Sounio data and checks
# Canvas, HTML/SVG, Workbench replay, physchem, and demo compile surfaces.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUC_NATIVE_WRAPPER:-${ROOT_DIR}/scripts/ci/souc-native-wrapper.sh}"
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

echo "=== Native Visual Frontend Gate ==="
echo "[native-viz-gate] souc=${SOUC}"
echo "[native-viz-gate] stdlib=${SOUNIO_STDLIB_PATH}"

check_file "viz_sci_module" "stdlib/viz/sci.sio"
check_file "viz_workbench_module" "stdlib/viz/workbench.sio"

run_expect "viz_module_surface" "tests/run-pass/viz_module_surface_gate.sio" "VIZ_MODULE_SURFACE_GATE_PASS"
run_expect "viz_headless" "tests/run-pass/viz_headless.sio" "VIZ_HEADLESS_PASS"
run_expect "viz_canvas_ext_surface" "tests/run-pass/viz_canvas_ext_surface.sio" "VIZ_CANVAS_EXT_SURFACE_PASS"
run_expect "viz_frontend_builders" "tests/run-pass/viz_frontend_builders.sio" "VIZ_FRONTEND_BUILDERS_PASS"
run_expect "viz_chart_builders" "tests/run-pass/viz_chart_builders.sio" "VIZ_CHART_BUILDERS_PASS"
run_expect "viz_event_reducer" "tests/run-pass/viz_event_reducer.sio" "VIZ_EVENT_REDUCER_PASS"
run_expect "viz_replay_deterministic" "tests/run-pass/viz_replay_deterministic.sio" "VIZ_REPLAY_DETERMINISTIC_PASS"
run_expect "viz_replay_trace" "tests/run-pass/viz_replay_trace.sio" "VIZ_REPLAY_TRACE_PASS"
run_expect "viz_html_static" "tests/run-pass/viz_html_static.sio" "VIZ_HTML_STATIC_PASS"
run_expect "viz_physchem" "tests/run-pass/viz_physchem.sio" "VIZ_PHYSCHEM_PASS"
run_expect "viz_physchem_dynamics" "tests/run-pass/viz_physchem_dynamics.sio" "VIZ_PHYSCHEM_DYNAMICS_PASS"
run_expect "viz_workbench_roundtrip" "tests/run-pass/viz_workbench_roundtrip.sio" "VIZ_WORKBENCH_ROUNDTRIP_PASS"
run_expect "viz_workbench_app_events" "tests/run-pass/viz_workbench_app_events.sio" "VIZ_WORKBENCH_APP_EVENTS_PASS"
run_expect "viz_workbench_physchem_dynamics" "tests/run-pass/viz_workbench_physchem_dynamics.sio" "VIZ_WORKBENCH_PHYSCHEM_DYNAMICS_PASS"
run_expect "viz_workbench_html_physchem" "tests/run-pass/viz_workbench_html_physchem.sio" "VIZ_WORKBENCH_HTML_PHYSCHEM_PASS"
run_expect "viz_workbench_mesh_dynamics" "tests/run-pass/viz_workbench_mesh_dynamics.sio" "VIZ_WORKBENCH_MESH_DYNAMICS_PASS"
run_expect "viz_workbench_timeline_playback" "tests/run-pass/viz_workbench_timeline_playback.sio" "VIZ_WORKBENCH_TIMELINE_PLAYBACK_PASS"
run_expect "viz_workbench_replay_studio" "tests/run-pass/viz_workbench_replay_studio.sio" "VIZ_WORKBENCH_REPLAY_STUDIO_PASS"
run_expect "viz_workbench_replay_session" "tests/run-pass/viz_workbench_replay_session.sio" "VIZ_WORKBENCH_REPLAY_SESSION_PASS"
run_expect "viz_workbench_replay_html_archive" "tests/run-pass/viz_workbench_replay_html_archive.sio" "VIZ_WORKBENCH_REPLAY_HTML_ARCHIVE_PASS"

compile_file "viz_hello_demo" "examples/viz_hello/main.sio"
compile_file "viz_lab_demo" "examples/viz_lab/main.sio"
compile_file "viz_physchem_demo" "examples/viz_physchem_demo/main.sio"
compile_file "viz_workbench_demo" "examples/viz_workbench/main.sio"

run_expect "viz_lab_run" "examples/viz_lab/main.sio" "VIZ_LAB_READY"
run_expect "viz_physchem_demo_run" "examples/viz_physchem_demo/main.sio" "VIZ_PHYSCHEM_DEMO_READY"
run_expect "viz_workbench_demo_run" "examples/viz_workbench/main.sio" "VIZ_WORKBENCH_READY"

echo "=== Native Visual Frontend Gate: PASS ==="
