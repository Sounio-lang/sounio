#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP:-0}"

fail() {
  echo "[madaros-f64-lowering] FAIL: $*" >&2
  exit 1
}

write_knowledge_receipt() {
  local gate_dir="$1" status="$2" rc="$3" tmp
  tmp="$(mktemp "$gate_dir/.status.XXXXXX")" || return 1
  printf 'version=1\nstatus=%s\nexit_status=%s\n' "$status" "$rc" > "$tmp" || return 1
  mv -f -- "$tmp" "$gate_dir/status.receipt"
}

knowledge_runtime_worker() {
  set -euo pipefail
  local gate_dir="${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:?}"
  local madaros_bin="${MADAROS_BIN:?}"
  local gate_script="${SOUNIO_MADAROS_F64_LOWERING_KNOWLEDGE_RUNTIME_SCRIPT:?}"
  local overall_rc=0 build_rc=0 verify_rc=0 gate_rc=0 driver driver_sha resolver_dir receipt_written=0
  gate_dir="$gate_dir/knowledge-runtime-threshold"
  mkdir -p "$gate_dir" || exit 65
  trap '[[ "$receipt_written" == 1 ]] || write_knowledge_receipt "$gate_dir" FAIL 1 || true' EXIT

  if [[ "$gate_script" != /* ]]; then gate_script="$ROOT_DIR/$gate_script"; fi
  if [[ ! -f "$gate_script" || ! -x "$madaros_bin" ]]; then
    echo "missing knowledge-runtime gate script or compiler" > "$gate_dir/worker.error"
    overall_rc=1
  fi
  resolver_dir="$ROOT_DIR/scripts/lib"
  if (( overall_rc == 0 )); then
    source "$resolver_dir/resolve_madaros.sh"
    if ! sounio_require_madaros; then overall_rc=1; fi
  fi
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
  if (( overall_rc == 0 )); then
    git -C "$ROOT_DIR" rev-parse HEAD > "$gate_dir/source-commit.txt" || overall_rc=1
    git -C "$ROOT_DIR" ls-files -z self-hosted stdlib tests/frontend/threshold-matrix |
      xargs -0 sha256sum > "$gate_dir/source.sha256" || overall_rc=1
    sha256sum "$madaros_bin" > "$gate_dir/madaros.sha256" || overall_rc=1
  fi
  driver="$gate_dir/driver"
  if (( overall_rc == 0 )); then
    rm -f "$driver"
    ulimit -S -s 65536
    timeout 900 "$madaros_bin" self-hosted/compiler/native_compile_driver.sio -o "$driver" \
      > "$gate_dir/driver-build.log" 2>&1 || build_rc=$?
    printf 'driver_build_rc=%s\n' "$build_rc" > "$gate_dir/phases.txt"
    cat "$gate_dir/driver-build.log"
    if [[ "$build_rc" != 0 || ! -s "$driver" ]]; then
      overall_rc=1
    else
      chmod +x "$driver"
      if ! sha256sum -c "$gate_dir/source.sha256" > "$gate_dir/source-verify.log" ||
         ! sha256sum -c "$gate_dir/madaros.sha256" > "$gate_dir/madaros-verify.log"; then
        verify_rc=1
        overall_rc=1
      fi
    fi
  fi
  if (( overall_rc == 0 )); then
    driver_sha="$(sha256sum "$driver" | cut -d ' ' -f1)"
    printf '%s  %s\n' "$driver_sha" "$driver" > "$gate_dir/driver.sha256"
    timeout 600 bash "$gate_script" "$driver" "$gate_dir/source.sha256" "$driver_sha" \
      > "$gate_dir/matrix.log" 2>&1 || gate_rc=$?
    cat "$gate_dir/matrix.log"
    printf 'threshold_matrix_rc=%s\n' "$gate_rc" >> "$gate_dir/phases.txt"
    [[ "$gate_rc" == 0 ]] || overall_rc=1
  fi
  write_knowledge_receipt "$gate_dir" "$([[ "$overall_rc" == 0 ]] && echo PASS || echo FAIL)" "$overall_rc" \
    || { echo "[madaros-f64-lowering] could not write knowledge-runtime receipt" >&2; exit 65; }
  receipt_written=1
  exit "$overall_rc"
}

start_knowledge_runtime() {
  local root="${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:?}"
  local gate_dir="$root/knowledge-runtime-threshold" pid
  [[ -x "${MADAROS_BIN:?}" ]] || fail "source-built Madaros is missing: $MADAROS_BIN"
  [[ -n "${SOUNIO_MADAROS_F64_LOWERING_KNOWLEDGE_RUNTIME_SCRIPT:-}" ]] \
    || fail "knowledge-runtime gate script input is missing"
  mkdir -p "$gate_dir" || fail "could not create knowledge-runtime output directory"
  [[ ! -e "$gate_dir/status.receipt" ]] || fail "refusing an existing knowledge-runtime receipt"
  nohup env \
    SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$root" \
    SOUNIO_MADAROS_F64_LOWERING_KNOWLEDGE_RUNTIME_SCRIPT="$SOUNIO_MADAROS_F64_LOWERING_KNOWLEDGE_RUNTIME_SCRIPT" \
    MADAROS_BIN="$MADAROS_BIN" \
    bash "$ROOT_DIR/scripts/ci/madaros_current_source_f64_lowering_gate.sh" --knowledge-runtime-worker \
    > "$gate_dir/worker.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$gate_dir/worker.pid"
  echo "[madaros-f64-lowering] knowledge-runtime worker started (pid $pid)"
}

wait_knowledge_runtime() {
  local gate_dir="${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:?}/knowledge-runtime-threshold"
  local timeout_seconds="${SOUNIO_MADAROS_F64_LOWERING_KNOWLEDGE_RUNTIME_WAIT_SECONDS:-1800}"
  local receipt="$gate_dir/status.receipt" started="$SECONDS" pid
  local -a lines
  [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || fail "invalid knowledge-runtime wait timeout"
  while [[ ! -f "$receipt" ]]; do
    if [[ ! -f "$gate_dir/worker.pid" ]]; then
      echo "[madaros-f64-lowering] FAIL: knowledge-runtime worker was not started" >&2
      return 1
    fi
    read -r pid < "$gate_dir/worker.pid"
    if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "$pid" 2>/dev/null; then
      echo "[madaros-f64-lowering] FAIL: knowledge-runtime worker ended without a receipt" >&2
      [[ ! -f "$gate_dir/worker.log" ]] || cat "$gate_dir/worker.log" >&2
      return 1
    fi
    if (( SECONDS - started >= timeout_seconds )); then
      echo "[madaros-f64-lowering] FAIL: timed out waiting for knowledge-runtime receipt" >&2
      [[ ! -f "$gate_dir/worker.log" ]] || cat "$gate_dir/worker.log" >&2
      return 1
    fi
    sleep 2
  done
  mapfile -t lines < "$receipt"
  if (( ${#lines[@]} != 3 )) || [[ "${lines[0]}" != version=1 ]] ||
     [[ ! "${lines[1]}" =~ ^status=(PASS|FAIL)$ ]] ||
     [[ ! "${lines[2]}" =~ ^exit_status=(0|[1-9][0-9]*)$ ]]; then
    echo "[madaros-f64-lowering] FAIL: malformed knowledge-runtime receipt" >&2
    return 1
  fi
  if [[ "${lines[1]}" != status=PASS || "${lines[2]}" != exit_status=0 ]]; then
    echo "[madaros-f64-lowering] FAIL: knowledge-runtime lane failed (${lines[*]})" >&2
    [[ ! -f "$gate_dir/worker.log" ]] || cat "$gate_dir/worker.log" >&2
    return 1
  fi
  echo "[madaros-f64-lowering] knowledge-runtime receipt: PASS"
}

case "${1:-}" in
  --knowledge-runtime-worker) knowledge_runtime_worker; exit $? ;;
  --start-knowledge-runtime) start_knowledge_runtime; exit $? ;;
  --wait-knowledge-runtime) wait_knowledge_runtime; exit $? ;;
esac

if [[ -n "${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-f64-lowering.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
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

MADAROS_RAW_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_CALL_ARITY_13_DIR="$WORK/imported-call-arity-13" \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_call_arity_13_gate.sh"

MADAROS_RAW_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_RECORD_OWNER_GATE_DIR="$WORK/nested-record-owner" \
  bash "$ROOT_DIR/scripts/ci/madaros_nested_if_record_owner_gate.sh"

echo "[madaros-f64-lowering] PASS: one shared Madaros ELF passed dereference, global f64, direct capacity, imported capacity, imported wide-call, and nested record-owner gates"
