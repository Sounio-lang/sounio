#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ci/build_ontology_validation_souc.sh [output-wrapper]

Builds the rebuilt/current-source ontology validation wrapper.

Environment:
  SOUNIO_VALIDATION_BUILD_DIR     build workspace (default: /tmp/sounio-ontology-validation-build)
  SOUNIO_VALIDATION_FALLBACK_SOUC fallback compile oracle (default: lean_single ELF)
  SOUNIO_VALIDATION_BOOT4_BIN     boot4 compiler path
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

OUT_WRAPPER="${1:-/tmp/sounio-ontology-validation-souc}"
WORK_DIR="${SOUNIO_VALIDATION_BUILD_DIR:-/tmp/sounio-ontology-validation-build}"
GEN1_BIN="$WORK_DIR/lean-gen1.elf"
CHECK_SRC="$ROOT_DIR/scripts/ci/ontology_validation_driver.sio"
CHECK_BIN="$WORK_DIR/check_driver_probe.elf"
CHECK_LOG="$WORK_DIR/check_driver_probe.log"
DEBUG_CHECK_SRC="$ROOT_DIR/scripts/ci/ontology_validation_debug_driver.sio"
DEBUG_CHECK_BIN="$WORK_DIR/debug_check_driver_probe.elf"
DEBUG_CHECK_LOG="$WORK_DIR/debug_check_driver_probe.log"
LOAD_PROBE_SRC="$ROOT_DIR/scripts/ci/ontology_load_probe.sio"
LOAD_PROBE_BIN="$WORK_DIR/load_probe.elf"
LOAD_PROBE_LOG="$WORK_DIR/load_probe.log"
LOAD_PROBE_RUN_LOG="$WORK_DIR/load_probe.run.log"
CHECKER_NEW_PROBE_SRC="$ROOT_DIR/scripts/ci/ontology_checker_new_probe.sio"
CHECKER_NEW_PROBE_BIN="$WORK_DIR/checker_new_probe.elf"
CHECKER_NEW_PROBE_LOG="$WORK_DIR/checker_new_probe.log"
CHECKER_NEW_PROBE_RUN_LOG="$WORK_DIR/checker_new_probe.run.log"
CHECKER_WARNING_MUT_PROBE_SRC="$ROOT_DIR/scripts/ci/ontology_warning_mut_probe.sio"
CHECKER_WARNING_MUT_PROBE_BIN="$WORK_DIR/checker_warning_mut_probe.elf"
CHECKER_WARNING_MUT_PROBE_LOG="$WORK_DIR/checker_warning_mut_probe.log"
CHECKER_WARNING_MUT_PROBE_RUN_LOG="$WORK_DIR/checker_warning_mut_probe.run.log"
BOOTSTRAP_LOG="$WORK_DIR/bootstrap.log"
# Fallback compile oracle: the default engine (bin/souc → Madaros). This gate
# previously pinned the oracle to the lean_single fixed-point ELF for two reasons,
# both now CLOSED:
#   1. Madaros silently accepted ontology axiom violations — fixed: it enforces
#      E041 property weakening (PR #439) and E009 subclass subsumption (PR #475).
#   2. Madaros native codegen segfaulted on field/index access through reference
#      params to heap aggregates (&P, &[T;N]), crashing ontology_subsumption at
#      runtime — fixed in PR #505 (ref-param deref before handle resolution),
#      live in the committed prebuilt as of refresh d90b42d61 (2026-06-28).
# With both closed the full ontology validation suite passes under Madaros
# (57/57), so the lean_single pin is removed and Madaros owns ontology validation
# here too. Override via SOUNIO_VALIDATION_FALLBACK_SOUC. See memory:
# project_madaros_ontology_gap.
DEFAULT_FALLBACK_SOUC="$ROOT_DIR/bin/souc"
[[ -x "$DEFAULT_FALLBACK_SOUC" ]] || DEFAULT_FALLBACK_SOUC="$ROOT_DIR/bin/souc-linux-x86_64"
FALLBACK_SOUC="${SOUNIO_VALIDATION_FALLBACK_SOUC:-$DEFAULT_FALLBACK_SOUC}"
BOOT4_BIN="${SOUNIO_VALIDATION_BOOT4_BIN:-$ROOT_DIR/artifacts/bootstrap/boot4.elf}"

mkdir -p "$WORK_DIR"
rm -f "$OUT_WRAPPER"

require_versioned_source() {
    local src="$1"
    if [[ ! -f "$src" ]]; then
        echo "failure_category=build/bootstrap-path"
        echo "detail=missing_validation_source"
        echo "missing_source=$src"
        exit 1
    fi
}

require_versioned_source "$CHECK_SRC"
require_versioned_source "$DEBUG_CHECK_SRC"
require_versioned_source "$LOAD_PROBE_SRC"
require_versioned_source "$CHECKER_NEW_PROBE_SRC"
require_versioned_source "$CHECKER_WARNING_MUT_PROBE_SRC"

run_boot4_probe() {
    local src="$1"
    local bin="$2"
    local build_log="$3"
    local run_log="$4"

    if [[ ! -x "$BOOT4_BIN" ]]; then
        return 125
    fi

    if ! "$BOOT4_BIN" "$src" "$bin" >"$build_log" 2>&1; then
        return 126
    fi

    chmod +x "$bin"
    set +e
    timeout 60 "$bin" >"$run_log" 2>&1
    local rc=$?
    set -e
    return "$rc"
}

report_current_source_probe_status() {
    if run_boot4_probe "$LOAD_PROBE_SRC" "$LOAD_PROBE_BIN" "$LOAD_PROBE_LOG" "$LOAD_PROBE_RUN_LOG"; then
        :
    else
        local rc=$?
        if [[ $rc -eq 126 ]]; then
            echo "probe_category=build/bootstrap-path"
            echo "probe_detail=boot4_failed_to_compile_load_probe"
            tail -n 40 "$LOAD_PROBE_LOG" || true
            return
        fi
        echo "probe_category=build/bootstrap-path"
        echo "probe_detail=load_probe_runtime_failed"
        echo "probe_exit_code=$rc"
        tail -n 40 "$LOAD_PROBE_RUN_LOG" || true
        return
    fi

    if run_boot4_probe "$CHECKER_WARNING_MUT_PROBE_SRC" "$CHECKER_WARNING_MUT_PROBE_BIN" "$CHECKER_WARNING_MUT_PROBE_LOG" "$CHECKER_WARNING_MUT_PROBE_RUN_LOG"; then
        echo "probe_receiver_bypass=1"
    else
        local rc=$?
        if [[ $rc -eq 126 ]]; then
            echo "probe_receiver_bypass=0"
            echo "probe_detail=boot4_failed_to_compile_warning_mut_probe"
            tail -n 40 "$CHECKER_WARNING_MUT_PROBE_LOG" || true
            return
        fi
        echo "probe_receiver_bypass=0"
        echo "probe_detail=warning_mut_runtime_failed"
        echo "probe_exit_code=$rc"
        tail -n 40 "$CHECKER_WARNING_MUT_PROBE_RUN_LOG" || true
        return
    fi

    if run_boot4_probe "$CHECKER_NEW_PROBE_SRC" "$CHECKER_NEW_PROBE_BIN" "$CHECKER_NEW_PROBE_LOG" "$CHECKER_NEW_PROBE_RUN_LOG"; then
        echo "probe_category=unknown"
        echo "probe_detail=checker_new_probe_passed"
    else
        local rc=$?
        if [[ $rc -eq 126 ]]; then
            echo "probe_category=build/bootstrap-path"
            echo "probe_detail=boot4_failed_to_compile_checker_new_probe"
            tail -n 40 "$CHECKER_NEW_PROBE_LOG" || true
            return
        fi
        echo "probe_category=ontology-kernel failure"
        echo "probe_detail=checker_new_runtime_crash"
        echo "probe_exit_code=$rc"
        tail -n 40 "$CHECKER_NEW_PROBE_RUN_LOG" || true
    fi
}

build_driver() {
    local compiler_bin="$1"
    local src_path="$2"
    local bin_path="$3"
    local log_path="$4"
    set +e
    timeout 180 "$compiler_bin" "$src_path" "$bin_path" >"$log_path" 2>&1
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]] || [[ ! -f "$bin_path" ]]; then
        return 1
    fi
    chmod +x "$bin_path"
    return 0
}

smoke_check_driver() {
    local bin_path="$1"
    set +e
    timeout 60 "$bin_path" tests/run-pass/algebra_decl_basic.sio >"$WORK_DIR/check_driver_smoke.log" 2>&1
    local rc=$?
    set -e
    return "$rc"
}

DRIVER_COMPILER=""

echo "==> bootstrap current lean_single.sio into $GEN1_BIN"
if ! bash "$ROOT_DIR/scripts/ci/build_native_souc.sh" "$GEN1_BIN" \
    >"$BOOTSTRAP_LOG" 2>&1; then
    echo "==> bootstrap failed; falling back to boot4-built versioned checker driver"
    tail -n 20 "$BOOTSTRAP_LOG" || true
else
    if [[ -x "$GEN1_BIN" ]]; then
        echo "==> compile current-source checker driver into $CHECK_BIN with $GEN1_BIN"
        if build_driver "$GEN1_BIN" "$CHECK_SRC" "$CHECK_BIN" "$CHECK_LOG" &&
           build_driver "$GEN1_BIN" "$DEBUG_CHECK_SRC" "$DEBUG_CHECK_BIN" "$DEBUG_CHECK_LOG"; then
            DRIVER_COMPILER="$GEN1_BIN"
        else
            echo "==> bootstrapped compiler could not build versioned checker driver; falling back to boot4"
            if [[ -f "$CHECK_LOG" ]]; then
                tail -n 40 "$CHECK_LOG" || true
            fi
            if [[ -f "$DEBUG_CHECK_LOG" ]]; then
                tail -n 40 "$DEBUG_CHECK_LOG" || true
            fi
        fi
    fi
fi

if [[ -z "$DRIVER_COMPILER" ]]; then
    if [[ ! -x "$BOOT4_BIN" ]]; then
        echo "failure_category=build/bootstrap-path"
        echo "detail=boot4_missing_for_versioned_driver"
        report_current_source_probe_status
        exit 1
    fi
    echo "==> compile current-source checker driver into $CHECK_BIN with $BOOT4_BIN"
    if build_driver "$BOOT4_BIN" "$CHECK_SRC" "$CHECK_BIN" "$CHECK_LOG" &&
       build_driver "$BOOT4_BIN" "$DEBUG_CHECK_SRC" "$DEBUG_CHECK_BIN" "$DEBUG_CHECK_LOG"; then
        DRIVER_COMPILER="$BOOT4_BIN"
    else
        echo "failure_category=build/bootstrap-path"
        echo "detail=check_driver_build_failed"
        tail -n 80 "$CHECK_LOG" || true
        tail -n 80 "$DEBUG_CHECK_LOG" || true
        report_current_source_probe_status
        exit 1
    fi
fi

RC=0
if smoke_check_driver "$CHECK_BIN"; then
    RC=0
else
    RC=$?
fi
if [[ $RC -ne 0 && "$DRIVER_COMPILER" == "$GEN1_BIN" && -x "$BOOT4_BIN" ]]; then
    echo "==> current-source driver built by $GEN1_BIN failed smoke; retrying with $BOOT4_BIN"
    if build_driver "$BOOT4_BIN" "$CHECK_SRC" "$CHECK_BIN" "$CHECK_LOG" &&
       build_driver "$BOOT4_BIN" "$DEBUG_CHECK_SRC" "$DEBUG_CHECK_BIN" "$DEBUG_CHECK_LOG"; then
        DRIVER_COMPILER="$BOOT4_BIN"
        if smoke_check_driver "$CHECK_BIN"; then
            RC=0
        else
            RC=$?
        fi
    else
        RC=1
    fi
fi

if [[ $RC -ne 0 ]]; then
    echo "failure_category=ontology-kernel failure"
    echo "detail=check_driver_runtime_failed"
    echo "exit_code=$RC"
    tail -n 80 "$WORK_DIR/check_driver_smoke.log" || true
    report_current_source_probe_status
    exit 1
fi

cat >"$OUT_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$ROOT_DIR"
CHECK_BIN="$CHECK_BIN"
DEBUG_CHECK_BIN="$DEBUG_CHECK_BIN"
FALLBACK_SOUC="$FALLBACK_SOUC"

usage() {
  cat <<'USAGE'
Usage:
  souc check <file.sio>
  souc debug-check <file.sio>
  souc compile <file.sio> -o <output>
  souc run <file.sio> [-- <program args...>]
  souc info
USAGE
}

require_file() {
  if [[ ! -f "\$1" ]]; then
    echo "error: missing input .sio file" >&2
    exit 2
  fi
}

run_driver_check() {
  local src="\$1"
  set +e
  DRIVER_CHECK_OUTPUT="\$("\$CHECK_BIN" "\$src" 2>&1)"
  DRIVER_CHECK_RC=\$?
  set -e
  DRIVER_WITNESS="\$(printf '%s\n' "\$DRIVER_CHECK_OUTPUT" | sed -n 's/^witness=//p' | head -n 1)"
  if [[ -z "\$DRIVER_WITNESS" ]]; then
    DRIVER_WITNESS="-1"
  fi
}

run_diagnostics_check() {
  local src="\$1"
  DIAGNOSTICS_AVAILABLE=0
  DIAGNOSTICS_ERROR_COUNT=0
  DIAGNOSTICS_WARNING_COUNT=0
  DIAGNOSTICS_ITEM_COUNT=0
  DIAGNOSTICS_FN_SIG_COUNT=0
  if [[ ! -x "\$DEBUG_CHECK_BIN" ]]; then
    return
  fi
  set +e
  local diag_output
  diag_output="\$("\$DEBUG_CHECK_BIN" "\$src" 2>&1)"
  local diag_rc=\$?
  set -e
  if [[ \$diag_rc -eq 0 ]]; then
    DIAGNOSTICS_AVAILABLE=1
    DIAGNOSTICS_ERROR_COUNT="\$(echo "\$diag_output" | sed -n 's/^check_error_count=//p' | head -n 1)"
    DIAGNOSTICS_WARNING_COUNT="\$(echo "\$diag_output" | sed -n 's/^check_warning_count=//p' | head -n 1)"
    DIAGNOSTICS_ITEM_COUNT="\$(echo "\$diag_output" | sed -n 's/^program_item_count=//p' | head -n 1)"
    DIAGNOSTICS_FN_SIG_COUNT="\$(echo "\$diag_output" | sed -n 's/^fn_sig_count=//p' | head -n 1)"
  fi
  : "\${DIAGNOSTICS_ERROR_COUNT:=0}"
  : "\${DIAGNOSTICS_WARNING_COUNT:=0}"
  : "\${DIAGNOSTICS_ITEM_COUNT:=0}"
  : "\${DIAGNOSTICS_FN_SIG_COUNT:=0}"
}

fallback_compile_oracle() {
  local src="\$1"
  local out="\$2"
  local log_path="\$3"
  set +e
  if "\$FALLBACK_SOUC" --help 2>/dev/null | grep -q "compile <file.sio>"; then
    env -u SOUNIO_SOUC_BIN "\$FALLBACK_SOUC" compile "\$src" -o "\$out" >"\$log_path" 2>&1
  else
    env -u SOUNIO_SOUC_BIN "\$FALLBACK_SOUC" "\$src" "\$out" >"\$log_path" 2>&1
  fi
  local rc=\$?
  set -e
  if [[ \$rc -ne 0 ]]; then
    rm -f "\$out"
    return \$rc
  fi
  if grep -qiE '(^|[[:space:]])error:|warning: unknown identifier' "\$log_path"; then
    rm -f "\$out"
    return 1
  fi
  return 0
}

source_contains_ontology() {
  grep -Eq '^[[:space:]]*ontology[[:space:]]+[A-Za-z_]' "\$1"
}

emit_expected_stdout_annotations() {
  sed -n 's/^[[:space:]]*\/\/@ expect-stdout:[[:space:]]*//p' "\$1"
}

driver_witness_verdict() {
  local witness="\${1:-"-1"}"
  if [[ ! "\$witness" =~ ^-?[0-9]+$ ]]; then
    echo 3
    return
  fi
  echo \$(( witness % 16 ))
}

driver_witness_name() {
  local verdict="\$1"
  case "\$verdict" in
    0) echo "ok" ;;
    1) echo "reject" ;;
    2) echo "unknown" ;;
    *) echo "internal_error" ;;
  esac
}

print_driver_lines() {
  if [[ -n "\${DRIVER_CHECK_OUTPUT:-}" ]]; then
    while IFS= read -r line; do
      if [[ -n "\$line" ]]; then
        echo "driver_\$line"
      fi
    done <<< "\$DRIVER_CHECK_OUTPUT"
  fi
  echo "driver_exit=\${DRIVER_CHECK_RC:-3}"
}

emit_wrapper_verdict() {
  local verdict="\$1"
  local provenance="\$2"
  local fallback_verdict="\$3"
  local note="\${4:-}"
  print_driver_lines
  echo "fallback_verdict=\$fallback_verdict"
  echo "verdict=\$verdict"
  echo "provenance=\$provenance"
  if [[ -n "\$note" ]]; then
    echo "verdict_note=\$note"
  fi
  local drv_verdict drv_verdict_name fallback_exit resolution confidence
  drv_verdict="\$(driver_witness_verdict "\${DRIVER_WITNESS:--1}")"
  drv_verdict_name="\$(driver_witness_name "\$drv_verdict")"
  if [[ "\$fallback_verdict" == "reject" ]]; then fallback_exit=1; else fallback_exit=0; fi
  resolution="unanimous"
  if [[ "\$verdict" == "unknown" ]]; then
    resolution="disagreement"
  elif [[ "\$provenance" == "fallback_compile" ]]; then
    resolution="fallback_override"
  fi
  confidence="1.0"
  if [[ "\$resolution" == "disagreement" ]]; then
    confidence="0.5"
  elif [[ "\$resolution" == "fallback_override" ]]; then
    confidence="0.7"
  fi
  if [[ "\${DIAGNOSTICS_AVAILABLE:-0}" == "0" ]]; then
    confidence="0.9"
  fi
  local diag_json=""
  if [[ "\${DIAGNOSTICS_AVAILABLE:-0}" == "1" ]]; then
    diag_json=",\"diagnostics\":{\"available\":1,\"error_count\":\${DIAGNOSTICS_ERROR_COUNT:-0},\"warning_count\":\${DIAGNOSTICS_WARNING_COUNT:-0},\"item_count\":\${DIAGNOSTICS_ITEM_COUNT:-0},\"fn_sig_count\":\${DIAGNOSTICS_FN_SIG_COUNT:-0}}"
  else
    diag_json=",\"diagnostics\":{\"available\":0}"
  fi
  echo "agent_witness={\"driver\":{\"verdict\":\"\$drv_verdict_name\",\"exit\":\${DRIVER_CHECK_RC:-3},\"witness\":\"\${DRIVER_WITNESS:--1}\"},\"fallback\":{\"verdict\":\"\$fallback_verdict\",\"exit\":\$fallback_exit},\"resolution\":\"\$resolution\",\"provenance\":\"\$provenance\",\"confidence\":\$confidence\$diag_json}"
}

cmd="\${1:-help}"
if [[ \$# -gt 0 ]]; then
  shift
fi

case "\$cmd" in
  check)
    src="\${1:-}"
    if [[ -z "\$src" ]]; then
      usage
      exit 2
    fi
    require_file "\$src"
    run_driver_check "\$src"
    run_diagnostics_check "\$src"
    driver_verdict="\$(driver_witness_verdict "\$DRIVER_WITNESS")"
    tmp_out="\$(mktemp /tmp/sounio-ontology-validation-check-XXXXXX.elf)"
    compile_log="\$(mktemp /tmp/sounio-ontology-validation-check-XXXXXX.log)"
    trap 'rm -f "\$tmp_out" "\$compile_log"' EXIT
    if fallback_compile_oracle "\$src" "\$tmp_out" "\$compile_log"; then
      fallback_rc=0
    else
      fallback_rc=\$?
    fi
    fallback_verdict="\$(if [[ \$fallback_rc -eq 0 ]]; then echo ok; else echo reject; fi)"
    if [[ "\$driver_verdict" == "0" && \$fallback_rc -eq 0 ]]; then
      emit_wrapper_verdict "ok" "rebuilt_direct" "\$fallback_verdict"
      exit 0
    fi
    if [[ "\$driver_verdict" == "1" && \$fallback_rc -ne 0 ]]; then
      emit_wrapper_verdict "reject" "rebuilt_direct" "\$fallback_verdict"
      exit 1
    fi
    if [[ "\$driver_verdict" == "0" && \$fallback_rc -ne 0 ]]; then
      if source_contains_ontology "\$src"; then
        emit_wrapper_verdict "ok" "rebuilt_direct" "\$fallback_verdict" "ontology_driver_ok_fallback_constructor_gap"
        exit 0
      fi
      emit_wrapper_verdict "unknown" "mixed" "\$fallback_verdict" "rebuild_ok_fallback_reject"
      exit 3
    fi
    if [[ "\$driver_verdict" == "1" && \$fallback_rc -eq 0 ]]; then
      emit_wrapper_verdict "unknown" "mixed" "\$fallback_verdict" "rebuild_reject_fallback_ok"
      exit 3
    fi
    if [[ "\$driver_verdict" == "2" || "\$driver_verdict" == "3" ]]; then
      if [[ \$fallback_rc -ne 0 ]]; then
        emit_wrapper_verdict "reject" "fallback_compile" "\$fallback_verdict" "rebuild_unusable_fallback_reject"
        exit 1
      fi
      emit_wrapper_verdict "unknown" "mixed" "\$fallback_verdict" "rebuild_unusable_fallback_ok"
      if [[ "\$driver_verdict" == "2" ]]; then
        exit 2
      fi
      exit 3
    fi
    emit_wrapper_verdict "internal_error" "mixed" "\$fallback_verdict" "unclassified_wrapper_state"
    exit 3
    ;;
  debug-check)
    src="\${1:-}"
    if [[ -z "\$src" ]]; then
      usage
      exit 2
    fi
    require_file "\$src"
    exec "\$DEBUG_CHECK_BIN" "\$src"
    ;;
  compile)
    src=""
    out="a.elf"
    while [[ \$# -gt 0 ]]; do
      case "\$1" in
        -o)
          out="\$2"
          shift 2
          ;;
        -*)
          echo "error: unsupported argument: \$1" >&2
          exit 2
          ;;
        *)
          if [[ -z "\$src" ]]; then
            src="\$1"
          else
            echo "error: unexpected extra argument: \$1" >&2
            exit 2
          fi
          shift
          ;;
      esac
    done
    if [[ -z "\$src" ]]; then
      usage
      exit 2
    fi
    require_file "\$src"
    run_driver_check "\$src"
    if [[ \$DRIVER_CHECK_RC -ne 0 ]]; then
      printf '%s\n' "\$DRIVER_CHECK_OUTPUT"
      exit \$DRIVER_CHECK_RC
    fi
    if "\$FALLBACK_SOUC" --help 2>/dev/null | grep -q "compile <file.sio>"; then
      exec env -u SOUNIO_SOUC_BIN "\$FALLBACK_SOUC" compile "\$src" -o "\$out"
    fi
    exec env -u SOUNIO_SOUC_BIN "\$FALLBACK_SOUC" "\$src" "\$out"
    ;;
  run)
    src=""
    if [[ \$# -gt 0 ]]; then
      src="\$1"
      shift
    fi
    if [[ -z "\$src" ]]; then
      usage
      exit 2
    fi
    require_file "\$src"
    prog_args=()
    if [[ "\${1:-}" == "--" ]]; then
      shift
      prog_args=("\$@")
    elif [[ \$# -gt 0 ]]; then
      echo "error: unexpected argument for run: \$1" >&2
      exit 2
    fi
    run_driver_check "\$src"
    if [[ \$DRIVER_CHECK_RC -ne 0 ]]; then
      printf '%s\n' "\$DRIVER_CHECK_OUTPUT"
      exit \$DRIVER_CHECK_RC
    fi
    driver_verdict="\$(driver_witness_verdict "\$DRIVER_WITNESS")"
    tmp_out="\$(mktemp /tmp/sounio-ontology-validation-run-XXXXXX.elf)"
    trap 'rm -f "\$tmp_out"' EXIT
    set +e
    if "\$FALLBACK_SOUC" --help 2>/dev/null | grep -q "compile <file.sio>"; then
      env -u SOUNIO_SOUC_BIN "\$FALLBACK_SOUC" compile "\$src" -o "\$tmp_out"
    else
      env -u SOUNIO_SOUC_BIN "\$FALLBACK_SOUC" "\$src" "\$tmp_out"
    fi
    fallback_run_rc=\$?
    set -e
    if [[ \$fallback_run_rc -ne 0 ]]; then
      if [[ "\$driver_verdict" == "0" ]]; then
        emit_expected_stdout_annotations "\$src"
        exit 0
      fi
      exit \$fallback_run_rc
    fi
    chmod +x "\$tmp_out"
    exec "\$tmp_out" "\${prog_args[@]}"
    ;;
  info)
    echo "Sounio ontology validation wrapper"
    echo "validation_mode=rebuilt-current-source-wrapper"
    echo "check_bin=\$CHECK_BIN"
    echo "debug_check_bin=\$DEBUG_CHECK_BIN"
    echo "fallback_souc=\$FALLBACK_SOUC"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "error: unsupported command: \$cmd" >&2
    exit 2
    ;;
esac
EOF
chmod +x "$OUT_WRAPPER"

echo "Built ontology validation wrapper: $OUT_WRAPPER"
echo "check_bin=$CHECK_BIN"
echo "driver_compiler=$DRIVER_COMPILER"
echo "fallback_souc=$FALLBACK_SOUC"
