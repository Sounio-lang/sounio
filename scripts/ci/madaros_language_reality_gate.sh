#!/usr/bin/env bash
# Madaros language reality gate.
#
# This is the release-facing source-to-ELF proof surface for the modular
# compiler. It checks each selected source file, exercises IR trace loading,
# compiles through the public `madaros build` command and the raw native-v2
# path, runs both ELFs, and records classified results for every case.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-language] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-language] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

OUT_DIR="${SOUNIO_MADAROS_LANGUAGE_REALITY_DIR:-$(mktemp -d /tmp/sounio-madaros-language.XXXXXX)}"
MANIFEST_PATH="${SOUNIO_MADAROS_LANGUAGE_REALITY_MANIFEST:-tests/madaros/language_reality/release_manifest.tsv}"
ALLOW_FAIL="${SOUNIO_MADAROS_LANGUAGE_REALITY_ALLOW_FAIL:-0}"
TIMEOUT_SECS="${SOUNIO_MADAROS_LANGUAGE_REALITY_TIMEOUT:-30}"
KEEP_WORK="${SOUNIO_MADAROS_LANGUAGE_REALITY_KEEP:-1}"
LOG_DIR="$OUT_DIR/logs"
BIN_DIR="$OUT_DIR/bin"
RESULTS_TSV="$OUT_DIR/results.tsv"
SUMMARY_TXT="$OUT_DIR/summary.txt"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$OUT_DIR"' EXIT
fi

mkdir -p "$LOG_DIR" "$BIN_DIR"

echo "[madaros-language] START: language reality gate"
echo "[madaros-language] madaros=$MADAROS_BIN"
echo "[madaros-language] out=$OUT_DIR"
echo "[madaros-language] manifest=$MANIFEST_PATH"
echo "[madaros-language] allow_fail=$ALLOW_FAIL"
echo "[madaros-language] timeout_secs=$TIMEOUT_SECS"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[madaros-language] FAIL: missing manifest $MANIFEST_PATH" >&2
  exit 1
fi

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  "$@" &
  local pid=$!
  local elapsed=0
  while kill -0 "$pid" >/dev/null 2>&1; do
    if [[ "$elapsed" -ge "$seconds" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  wait "$pid"
}

assert_elf_x86_64() {
  local path="$1"
  local label="$2"
  local file_log="$LOG_DIR/${label}.file.txt"

  if command -v file >/dev/null 2>&1; then
    file "$path" >"$file_log"
    grep -q 'ELF 64-bit LSB executable, x86-64' "$file_log"
    return $?
  fi
  return 0
}

record_result() {
  local case_id="$1"
  local phase="$2"
  local mode="$3"
  local program_path="$4"
  local expected_exit="$5"
  local actual_exit="$6"
  local expected_stdout="$7"
  local stdout_status="$8"
  local claim_class="$9"
  local sha="${10}"
  local status="${11}"
  local log_path="${12}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$phase" "$mode" "$program_path" "$expected_exit" "$actual_exit" \
    "$expected_stdout" "$stdout_status" "$claim_class" "$sha" "$status" "$log_path" \
    >>"$RESULTS_TSV"
}

compare_stdout() {
  local expected_stdout="$1"
  local actual_stdout="$2"

  if [[ "$expected_stdout" == "-" ]]; then
    return 0
  fi
  cmp -s "$expected_stdout" "$actual_stdout"
}

compile_and_run_mode() {
  local case_id="$1"
  local mode="$2"
  local program_path="$3"
  local expected_exit="$4"
  local expected_stdout="$5"
  local claim_class="$6"

  local bin_path="$BIN_DIR/${case_id}.${mode}.elf"
  local compile_log="$LOG_DIR/${case_id}.${mode}.compile.log"
  local run_stdout="$LOG_DIR/${case_id}.${mode}.stdout"
  local run_stderr="$LOG_DIR/${case_id}.${mode}.stderr"
  local compile_rc=0
  local run_rc=0
  local sha="-"
  local status="ok"
  local stdout_status="skip"

  rm -f "$bin_path" "$compile_log" "$run_stdout" "$run_stderr"

  set +e
  if [[ "$mode" == "public_build" ]]; then
    run_with_timeout "$TIMEOUT_SECS" "$MADAROS_BIN" build "$program_path" -o "$bin_path" >"$compile_log" 2>&1
  elif [[ "$mode" == "raw_native_v2" ]]; then
    run_with_timeout "$TIMEOUT_SECS" "$MADAROS_BIN" --native-v2-compile "$program_path" "$bin_path" >"$compile_log" 2>&1
  else
    echo "internal error: unknown compile mode $mode" >"$compile_log"
    false
  fi
  compile_rc=$?
  set -e

  if [[ "$compile_rc" != "0" ]]; then
    record_result "$case_id" "compile" "$mode" "$program_path" "$expected_exit" "$compile_rc" \
      "$expected_stdout" "n/a" "$claim_class" "-" "compile_fail" "$compile_log"
    return 1
  fi

  if [[ ! -s "$bin_path" ]]; then
    record_result "$case_id" "compile" "$mode" "$program_path" "$expected_exit" "-" \
      "$expected_stdout" "n/a" "$claim_class" "-" "missing_elf" "$compile_log"
    return 1
  fi

  chmod +x "$bin_path" 2>/dev/null || true

  if ! assert_elf_x86_64 "$bin_path" "${case_id}.${mode}"; then
    record_result "$case_id" "compile" "$mode" "$program_path" "$expected_exit" "-" \
      "$expected_stdout" "n/a" "$claim_class" "-" "not_elf64_x86_64" "$LOG_DIR/${case_id}.${mode}.file.txt"
    return 1
  fi

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$bin_path" >"$run_stdout" 2>"$run_stderr"
  run_rc=$?
  set -e

  if [[ "$run_rc" != "$expected_exit" ]]; then
    status="exit_mismatch"
  fi

  if [[ "$expected_stdout" == "-" ]]; then
    stdout_status="skip"
  elif compare_stdout "$expected_stdout" "$run_stdout"; then
    stdout_status="ok"
  else
    stdout_status="mismatch"
    if [[ "$status" == "ok" ]]; then
      status="stdout_mismatch"
    fi
  fi

  sha="$(portable_sha256 "$bin_path")"
  record_result "$case_id" "run" "$mode" "$program_path" "$expected_exit" "$run_rc" \
    "$expected_stdout" "$stdout_status" "$claim_class" "$sha" "$status" "$run_stderr"

  [[ "$status" == "ok" ]]
}

printf 'case_id\tphase\tmode\tprogram\texpected_exit\tactual_exit\texpected_stdout\tstdout_status\tclaim_class\tsha256\tstatus\tlog\n' >"$RESULTS_TSV"

case_count=0
row_count=0
pass_count=0
fail_count=0

while IFS=$'\t' read -r case_id program_path expected_exit expected_stdout claim_class; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi

  case_count=$((case_count + 1))
  check_log="$LOG_DIR/$case_id.check.log"
  trace_log="$LOG_DIR/$case_id.trace.log"

  if [[ ! -f "$program_path" ]]; then
    record_result "$case_id" "preflight" "all" "$program_path" "$expected_exit" "-" \
      "$expected_stdout" "n/a" "${claim_class:-unknown}" "-" "missing_program" "-"
    fail_count=$((fail_count + 1))
    row_count=$((row_count + 1))
    continue
  fi

  if [[ "$expected_stdout" != "-" && ! -f "$expected_stdout" ]]; then
    record_result "$case_id" "preflight" "all" "$program_path" "$expected_exit" "-" \
      "$expected_stdout" "n/a" "$claim_class" "-" "missing_stdout_fixture" "-"
    fail_count=$((fail_count + 1))
    row_count=$((row_count + 1))
    continue
  fi

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$MADAROS_BIN" check "$program_path" >"$check_log" 2>&1
  check_rc=$?
  set -e
  if [[ "$check_rc" != "0" ]]; then
    record_result "$case_id" "check" "all" "$program_path" "$expected_exit" "$check_rc" \
      "$expected_stdout" "n/a" "$claim_class" "-" "check_fail" "$check_log"
    fail_count=$((fail_count + 1))
    row_count=$((row_count + 1))
    continue
  fi
  record_result "$case_id" "check" "all" "$program_path" "$expected_exit" "0" \
    "$expected_stdout" "n/a" "$claim_class" "-" "ok" "$check_log"
  pass_count=$((pass_count + 1))
  row_count=$((row_count + 1))

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$MADAROS_BIN" --probe-load-ir-trace "$program_path" >"$trace_log" 2>&1
  trace_rc=$?
  set -e
  if [[ "$trace_rc" != "0" ]]; then
    record_result "$case_id" "trace" "all" "$program_path" "$expected_exit" "$trace_rc" \
      "$expected_stdout" "n/a" "$claim_class" "-" "trace_fail" "$trace_log"
    fail_count=$((fail_count + 1))
    row_count=$((row_count + 1))
    continue
  fi
  record_result "$case_id" "trace" "all" "$program_path" "$expected_exit" "0" \
    "$expected_stdout" "n/a" "$claim_class" "-" "ok" "$trace_log"
  pass_count=$((pass_count + 1))
  row_count=$((row_count + 1))

  if compile_and_run_mode "$case_id" "public_build" "$program_path" "$expected_exit" "$expected_stdout" "$claim_class"; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
  row_count=$((row_count + 1))

  if compile_and_run_mode "$case_id" "raw_native_v2" "$program_path" "$expected_exit" "$expected_stdout" "$claim_class"; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
  row_count=$((row_count + 1))
done <"$MANIFEST_PATH"

{
  echo "summary_cases=$case_count"
  echo "summary_rows=$row_count"
  echo "summary_pass=$pass_count"
  echo "summary_fail=$fail_count"
  echo "manifest=$MANIFEST_PATH"
  echo "madaros=$MADAROS_BIN"
  echo "out_dir=$OUT_DIR"
  echo "results=$RESULTS_TSV"
} >"$SUMMARY_TXT"

echo "[madaros-language] SUMMARY cases=$case_count rows=$row_count pass=$pass_count fail=$fail_count"
echo "[madaros-language] results=$RESULTS_TSV"
echo "[madaros-language] summary=$SUMMARY_TXT"

if [[ "$fail_count" != "0" ]]; then
  echo "[madaros-language] FAILURES:" >&2
  awk -F '\t' 'NR == 1 { next } $11 != "ok" { printf "  %s phase=%s mode=%s class=%s status=%s log=%s\n", $1, $2, $3, $9, $11, $12 }' "$RESULTS_TSV" >&2
  if [[ "$ALLOW_FAIL" != "1" ]]; then
    exit 1
  fi
  echo "[madaros-language] CENSUS: failures allowed; see classified results TSV"
  exit 0
fi

echo "[madaros-language] PASS: check, IR trace, public build, raw native-v2 build, ELF run, exits, and stdout fixtures"
