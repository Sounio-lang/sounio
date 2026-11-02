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
      [[ -f "$receipt" ]] && break
      echo "[madaros-f64-lowering] FAIL: knowledge-runtime worker ended without a receipt" >&2
      [[ ! -f "$gate_dir/worker.log" ]] || cat "$gate_dir/worker.log" >&2
      return 1
    fi
    if (( SECONDS - started >= timeout_seconds )); then
      [[ -f "$receipt" ]] && break
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

# A directory that already holds only an executable `madaros` is the CI path:
# Witness Gate built the compiler, the receipt was verified, and this job
# copied that ELF here. Anything else already in the directory is leftover
# state. SOUNIO_MADAROS_F64_LOWERING_BUILD overrides the builder for the
# resolve-compiler selftest; CI must not set it.
# bash -L does not see a symlink when the path ends in / or /. ; those forms
# name the directory reached through the link.
gate_dir_lexical() {
  local p=$1
  while [[ "$p" == */ || "$p" == */. ]]; do
    [[ "$p" == / ]] && break
    if [[ "$p" == */. ]]; then p="${p%/.}"; else p="${p%/}"; fi
  done
  printf '%s\n' "$p"
}

prepare_compiler() {
  local created=0 preplaced=0 extra build_cmd
  compiler_origin=built
  if [[ -n "${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:-}" ]]; then
    WORK="$(gate_dir_lexical "$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR")"
    # -d follows a symlink, and find does not descend into one. A link to a
    # directory that also holds leftover files would pass the scan below.
    if [[ -L "$WORK" ]]; then
      fail "refusing existing gate directory: $WORK"
    fi
    if [[ -d "$WORK" && -f "$WORK/madaros" && -s "$WORK/madaros" && -x "$WORK/madaros" && ! -L "$WORK/madaros" ]]; then
      extra="$(find "$WORK" -mindepth 1 -maxdepth 1 ! -name madaros -print -quit)"
      if [[ -n "$extra" ]]; then
        fail "refusing existing gate directory: $WORK"
      fi
      preplaced=1
    elif [[ -e "$WORK" ]]; then
      fail "refusing existing gate directory: $WORK"
    else
      mkdir "$WORK" || fail "could not create gate directory: $WORK"
      created=1
    fi
  else
    WORK="$(mktemp -d /tmp/sounio-madaros-f64-lowering.XXXXXX)"
    created=1
  fi

  MADAROS_ELF="${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-$WORK/madaros}"
  if [[ "$KEEP_WORK" != "1" && "$created" == 1 ]]; then
    trap 'rm -rf "$WORK"' EXIT
  fi

  if [[ -n "${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-}" ]]; then
    compiler_origin=supplied
  elif [[ "$preplaced" == 1 ]]; then
    compiler_origin=preplaced
    echo "[madaros-f64-lowering] using pre-placed compiler: $MADAROS_ELF"
  else
    build_cmd="${SOUNIO_MADAROS_F64_LOWERING_BUILD:-$ROOT_DIR/scripts/ci/build_modular_madaros.sh}"
    compiler_origin=built
    if ! bash "$build_cmd" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
      tail -n 80 "$WORK/build.log" >&2 || true
      fail "current-source Madaros build failed"
    fi
  fi
  [[ -f "$MADAROS_ELF" && -s "$MADAROS_ELF" && -x "$MADAROS_ELF" && ! -L "$MADAROS_ELF" ]] \
    || fail "Madaros is missing or not executable: $MADAROS_ELF"
}

resolve_compiler_selftest() {
  local tmp stub rc out
  tmp="$(mktemp -d /tmp/sounio-f64-resolve-selftest.XXXXXX)"
  # EXIT runs after this function's locals are gone. The path has to live in a
  # global so a failing assertion still deletes the fixture.
  SOUNIO_F64_RESOLVE_SELFTEST_TMP=$tmp
  trap 'rm -rf "$SOUNIO_F64_RESOLVE_SELFTEST_TMP"' EXIT
  stub="$tmp/build.sh"
  cat >"$stub" <<'EOF'
#!/bin/bash
set -euo pipefail
printf 'built\n' > "$(dirname "$1")/build.marker"
printf '#!/bin/sh\nexit 0\n' > "$1"
chmod 755 "$1"
EOF
  chmod 755 "$stub"
  mkdir -p "$tmp/pre"
  printf '#!/bin/sh\nexit 0\n' > "$tmp/pre/madaros"
  chmod 755 "$tmp/pre/madaros"
  rc=0
  out="$(SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/pre" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    SOUNIO_MADAROS_F64_LOWERING_BUILD="$stub" \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only)" || rc=$?
  [[ "$rc" == 0 ]] || fail "preplaced compiler was refused ($rc): $out"
  grep -q 'RESOLVED preplaced' <<<"$out" || fail "preplaced marker missing: $out"
  [[ ! -e "$tmp/pre/build.marker" ]] || fail "preplaced compiler was rebuilt"
  mkdir -p "$tmp/dirty"
  printf '#!/bin/sh\nexit 0\n' > "$tmp/dirty/madaros"
  chmod 755 "$tmp/dirty/madaros"
  echo x > "$tmp/dirty/extra"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/dirty" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/dirty.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "contaminated directory was accepted ($rc)"
  grep -q 'refusing existing gate directory' "$tmp/dirty.log" || fail "contaminated directory lacked refusal"
  rc=0
  out="$(SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/fresh" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    SOUNIO_MADAROS_F64_LOWERING_BUILD="$stub" \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only)" || rc=$?
  [[ "$rc" == 0 ]] || fail "fresh build path failed ($rc): $out"
  grep -q 'RESOLVED built' <<<"$out" || fail "fresh build marker missing: $out"
  [[ -f "$tmp/fresh/build.marker" ]] || fail "fresh directory did not call the builder"
  printf '#!/bin/sh\nexit 0\n' > "$tmp/supplied"
  chmod 755 "$tmp/supplied"
  rc=0
  out="$(SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/supplied-work" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_BIN="$tmp/supplied" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    SOUNIO_MADAROS_F64_LOWERING_BUILD="$stub" \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only)" || rc=$?
  [[ "$rc" == 0 ]] || fail "supplied compiler was refused ($rc): $out"
  grep -q 'RESOLVED supplied' <<<"$out" || fail "supplied marker missing: $out"
  [[ ! -e "$tmp/supplied-work/build.marker" ]] || fail "supplied compiler was rebuilt"
  mkdir -p "$tmp/link"
  ln -s "$tmp/supplied" "$tmp/link/madaros"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/link" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/link.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "symlinked compiler was accepted ($rc)"
  grep -q 'refusing existing gate directory' "$tmp/link.log" || fail "symlink lacked refusal"
  mkdir -p "$tmp/via-target"
  printf '#!/bin/sh\nexit 0\n' > "$tmp/via-target/madaros"
  chmod 755 "$tmp/via-target/madaros"
  echo leftover > "$tmp/via-target/extra"
  ln -s "$tmp/via-target" "$tmp/via-link"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/via-link" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/via-link.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "symlinked gate directory was accepted ($rc)"
  grep -q 'refusing existing gate directory' "$tmp/via-link.log" || fail "symlinked gate directory lacked refusal"
  mkdir -p "$tmp/clean-target"
  printf '#!/bin/sh\nexit 0\n' > "$tmp/clean-target/madaros"
  chmod 755 "$tmp/clean-target/madaros"
  ln -s "$tmp/clean-target" "$tmp/clean-link"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/clean-link" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/clean-link.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "clean symlinked gate directory was accepted ($rc)"
  grep -q 'refusing existing gate directory' "$tmp/clean-link.log" || fail "clean symlink lacked refusal"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/clean-link/" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/clean-link-slash.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "trailing-slash symlink was accepted ($rc)"
  grep -q 'refusing existing gate directory' "$tmp/clean-link-slash.log" || fail "trailing-slash symlink lacked refusal"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/clean-link/." \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/clean-link-dot.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "dot-suffix symlink was accepted ($rc)"
  grep -q 'refusing existing gate directory' "$tmp/clean-link-dot.log" || fail "dot-suffix symlink lacked refusal"
  rc=0
  out="$(SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/pre/" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    SOUNIO_MADAROS_F64_LOWERING_BUILD="$stub" \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only)" || rc=$?
  [[ "$rc" == 0 ]] || fail "real directory with a trailing slash was refused ($rc): $out"
  grep -q 'RESOLVED preplaced' <<<"$out" || fail "trailing-slash real directory marker missing: $out"
  printf '#!/bin/bash\nexit 3\n' > "$tmp/failbuild"
  chmod 755 "$tmp/failbuild"
  rc=0
  SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$tmp/fail" \
    SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 \
    SOUNIO_MADAROS_F64_LOWERING_BUILD="$tmp/failbuild" \
    bash "${BASH_SOURCE[0]}" --resolve-compiler-only >"$tmp/fail.log" 2>&1 || rc=$?
  [[ "$rc" == 1 ]] || fail "failed builder was accepted ($rc)"
  grep -q 'current-source Madaros build failed' "$tmp/fail.log" || fail "failed builder lacked diagnostic"
  echo "[madaros-f64-lowering] resolve-compiler selftest PASS"
}

case "${1:-}" in
  --knowledge-runtime-worker) knowledge_runtime_worker; exit $? ;;
  --start-knowledge-runtime) start_knowledge_runtime; exit $? ;;
  --wait-knowledge-runtime) wait_knowledge_runtime; exit $? ;;
  --resolve-compiler-selftest) resolve_compiler_selftest; exit $? ;;
  --resolve-compiler-only) prepare_compiler; printf 'RESOLVED %s\n' "$compiler_origin"; exit 0 ;;
esac

prepare_compiler

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
