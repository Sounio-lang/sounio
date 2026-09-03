#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
MATRIX_FILE="${MATRIX_FILE:-scripts/poseidon_compat_matrix.txt}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-poseidon-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
BUILD_SOUC="${BUILD_SOUC:-1}"
SELFHOST_BACKEND="${SOUNIO_PARITY_BACKEND:-${SOUNIO_SELFHOST_MODE:-driver}}"
if [ "$SELFHOST_BACKEND" = "vm" ]; then
  SELFHOST_BACKEND="driver"
fi
SELFHOST_EXECUTOR="${SOUNIO_SELFHOST_EXECUTOR:-}"
if [ -z "$SELFHOST_EXECUTOR" ]; then
  case "$SELFHOST_BACKEND" in
    driver|native-driver)
      SELFHOST_EXECUTOR="native-driver"
      ;;
  esac
fi
SKIP_FULL_SELFHOST="${POSEIDON_SKIP_FULL_SELFHOST:-0}"
RUN_ORACLE_PARITY="${POSEIDON_ORACLE_PARITY:-0}"
BASELINE_COMPARE="${POSEIDON_BASELINE_COMPARE:-1}"
BASELINE_COMMIT="${POSEIDON_BASELINE_COMMIT:-01ecf01}"
BASELINE_WORKTREE="$WORK_DIR/baseline-wt"
BASELINE_SOUC="$BASELINE_WORKTREE/target/debug/souc"
FULL_SELFHOST_TARGET="${FULL_SELFHOST_TARGET:-self-hosted/}"
WRAPPER_STEPS_TSV="$ARTIFACT_DIR/poseidon_wrapper_steps.tsv"
WRAPPER_STATUS_JSON="$ARTIFACT_DIR/poseidon_wrapper_status.v1.json"
WRAPPER_BENCH_JSON="$ARTIFACT_DIR/poseidon_wrapper_benchmarks.v1.json"

PASS_COUNT=0
FAIL_COUNT=0
NOT_RUN_COUNT=0

has_repo_cargo_manifest() {
  [ -f "$ROOT_DIR/Cargo.toml" ]
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "PASS [$1] $2"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "FAIL [$1] $2"
}

not_run() {
  NOT_RUN_COUNT=$((NOT_RUN_COUNT + 1))
  echo "NOT_RUN [$1] $2"
}

record_wrapper_step() {
  local name="$1"
  local status="$2"
  local reason="$3"
  local log_path="$4"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$reason" "$log_path" >>"$WRAPPER_STEPS_TSV"
}

summarize_criterion_benchmarks() {
  local bench_root="$1"
  local manifest_path="$2"
  local out_json="$3"
  python3 - "$bench_root" "$manifest_path" "$out_json" <<'PY'
import json
import sys
from pathlib import Path

bench_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
out_json = Path(sys.argv[3])
rows = []

if not bench_root.exists() or not manifest_path.exists():
    raise SystemExit(1)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {
    fixture["id"]
    for fixture in manifest.get("fixtures", [])
    if "bench" in fixture.get("tags", [])
}
if not required:
    raise SystemExit(1)

for estimates in sorted(bench_root.glob("load_execute/*/new/estimates.json")):
    data = json.loads(estimates.read_text(encoding="utf-8"))
    rows.append(
        {
            "name": estimates.parent.parent.name,
            "mean_ns": data.get("mean", {}).get("point_estimate"),
            "median_ns": data.get("median", {}).get("point_estimate"),
            "path": str(estimates),
        }
    )

names = {row["name"] for row in rows}
if not required.issubset(names):
    raise SystemExit(1)

payload = {
    "schema": "sounio.poseidon.wrapper.benchmarks.v1",
    "benchmarks": rows,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_wrapper_status_json() {
  python3 - "$WRAPPER_STEPS_TSV" "$WRAPPER_BENCH_JSON" "$WRAPPER_STATUS_JSON" <<'PY'
import json
import sys
from pathlib import Path

steps_path = Path(sys.argv[1])
bench_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
required_steps = {"c_lib_build", "c_cli_regression"}
steps = []

if steps_path.exists():
    for raw in steps_path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        name, status, reason, log_path = raw.split("\t")
        steps.append(
            {
                "name": name,
                "status": status,
                "reason": reason,
                "required": name in required_steps,
                "log_path": log_path,
            }
        )

overall = "pass"
for step in steps:
    if step["required"] and step["status"] == "fail":
        overall = "fail"
        break

payload = {
    "schema": "sounio.poseidon.wrapper.status.v1",
    "status": overall,
    "steps": steps,
}
if bench_path.exists():
    payload["benchmarks"] = json.loads(bench_path.read_text(encoding="utf-8")).get("benchmarks", [])

out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_poseidon_wrapper_checks() {
  local log_path=""
  local rc=0

  : >"$WRAPPER_STEPS_TSV"
  rm -f "$WRAPPER_BENCH_JSON" "$WRAPPER_STATUS_JSON"

  log_path="$LOG_DIR/wrapper_c_lib_build.log"
  set +e
  bash -lc "cd \"$ROOT_DIR/bootstrap/poseidon\" && make clean && make all poseidon_lib_tests" >"$log_path" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    pass "wrapper-c-lib-build" "built libposeidon.a, poseidon, and poseidon_lib_tests"
    record_wrapper_step "c_lib_build" "pass" "built libposeidon.a and test binary" "$log_path"
  else
    fail "wrapper-c-lib-build" "build failed (see $log_path)"
    record_wrapper_step "c_lib_build" "fail" "build_failed_rc_${rc}" "$log_path"
  fi

  log_path="$LOG_DIR/wrapper_c_cli_regression.log"
  set +e
  bash -lc "cd \"$ROOT_DIR/bootstrap/poseidon\" && make test" >"$log_path" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    pass "wrapper-c-cli-regression" "CLI and C library regression suite passed"
    record_wrapper_step "c_cli_regression" "pass" "make_test_passed" "$log_path"
  else
    fail "wrapper-c-cli-regression" "regression suite failed (see $log_path)"
    record_wrapper_step "c_cli_regression" "fail" "make_test_failed_rc_${rc}" "$log_path"
  fi

  if ! command -v cargo >/dev/null 2>&1 || [ ! -f "$ROOT_DIR/bootstrap/poseidon/rust/Cargo.toml" ]; then
    record_wrapper_step "rust_wrapper_tests" "not_run" "cargo_or_manifest_missing" ""
    record_wrapper_step "proptest" "not_run" "cargo_or_manifest_missing" ""
    record_wrapper_step "fuzz_build" "not_run" "cargo_or_manifest_missing" ""
    record_wrapper_step "bench_smoke" "not_run" "cargo_or_manifest_missing" ""
    write_wrapper_status_json
    return 0
  fi

  log_path="$LOG_DIR/wrapper_rust_wrapper_tests.log"
  set +e
  cargo test --manifest-path "$ROOT_DIR/bootstrap/poseidon/rust/Cargo.toml" --test api >"$log_path" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    pass "wrapper-rust-tests" "standalone Rust API tests passed"
    record_wrapper_step "rust_wrapper_tests" "pass" "cargo_test_api_passed" "$log_path"
  else
    fail "wrapper-rust-tests" "standalone Rust API tests failed (see $log_path)"
    record_wrapper_step "rust_wrapper_tests" "fail" "cargo_test_api_failed_rc_${rc}" "$log_path"
  fi

  log_path="$LOG_DIR/wrapper_proptest.log"
  set +e
  bash -lc "cargo test --manifest-path \"$ROOT_DIR/bootstrap/poseidon/rust/Cargo.toml\" --test prop_straight_line && cargo test --manifest-path \"$ROOT_DIR/bootstrap/poseidon/rust/Cargo.toml\" --test prop_invalid_bytes" >"$log_path" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    pass "wrapper-proptest" "property tests passed"
    record_wrapper_step "proptest" "pass" "cargo_test_proptest_passed" "$log_path"
  else
    fail "wrapper-proptest" "property tests failed (see $log_path)"
    record_wrapper_step "proptest" "fail" "cargo_test_proptest_failed_rc_${rc}" "$log_path"
  fi

  log_path="$LOG_DIR/wrapper_fuzz_build.log"
  if command -v cargo-fuzz >/dev/null 2>&1; then
    set +e
    bash -lc "cd \"$ROOT_DIR/bootstrap/poseidon/rust/fuzz\" && ./refresh_corpus.sh && cargo fuzz build load_bytes && cargo fuzz build run" >"$log_path" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
      pass "wrapper-fuzz-build" "cargo-fuzz targets built"
      record_wrapper_step "fuzz_build" "pass" "cargo_fuzz_build_passed" "$log_path"
    else
      fail "wrapper-fuzz-build" "cargo-fuzz build failed (see $log_path)"
      record_wrapper_step "fuzz_build" "fail" "cargo_fuzz_build_failed_rc_${rc}" "$log_path"
    fi
  else
    record_wrapper_step "fuzz_build" "not_run" "cargo_fuzz_missing" ""
  fi

  log_path="$LOG_DIR/wrapper_bench_smoke.log"
  rm -rf "$ROOT_DIR/bootstrap/poseidon/rust/target/criterion"
  set +e
  cargo bench --manifest-path "$ROOT_DIR/bootstrap/poseidon/rust/Cargo.toml" --bench poseidon_bench -- --noplot >"$log_path" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && summarize_criterion_benchmarks "$ROOT_DIR/bootstrap/poseidon/rust/target/criterion" "$ROOT_DIR/bootstrap/poseidon/tests/fixtures_manifest.v1.json" "$WRAPPER_BENCH_JSON"; then
    pass "wrapper-bench-smoke" "criterion benchmarks completed"
    record_wrapper_step "bench_smoke" "pass" "criterion_bench_passed" "$log_path"
  elif [ "$rc" -eq 0 ]; then
    fail "wrapper-bench-smoke" "criterion output missing required fixtures (see $log_path)"
    record_wrapper_step "bench_smoke" "fail" "criterion_output_incomplete" "$log_path"
  else
    fail "wrapper-bench-smoke" "criterion benchmark run failed (see $log_path)"
    record_wrapper_step "bench_smoke" "fail" "criterion_bench_failed_rc_${rc}" "$log_path"
  fi

  write_wrapper_status_json
}

run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
  else
    "$@"
  fi
}

count_matches() {
  local pattern="$1"
  local file="$2"
  local count="0"
  local status=0

  if [ ! -f "$file" ]; then
    echo "0"
    return 0
  fi

  if command -v rg >/dev/null 2>&1; then
    set +e
    count="$(rg -c -- "$pattern" "$file" 2>/dev/null)"
    status=$?
    set -e
  else
    set +e
    count="$(grep -E -c -- "$pattern" "$file" 2>/dev/null)"
    status=$?
    set -e
  fi

  if [ "$status" -ne 0 ] && [ "$status" -ne 1 ]; then
    count="0"
  fi

  if [[ ! "$count" =~ ^[0-9]+$ ]]; then
    count="0"
  fi

  echo "$count"
}

marker_value() {
  local marker_file="$1"
  local key="$2"
  local value

  if [ ! -f "$marker_file" ]; then
    echo "0"
    return 0
  fi

  value="$(awk -F= -v key="$key" '$1 == key { print $2; found=1; exit } END { if (!found) print "0" }' "$marker_file")"
  if [[ ! "$value" =~ ^-?[0-9]+$ ]]; then
    value="0"
  fi
  echo "$value"
}

run_full_selfhost() {
  local souc_bin="$1"
  local log_file="$2"

  run_with_timeout "$TIMEOUT_SECS" env \
    SOUNIO_PARITY_BACKEND="$SELFHOST_BACKEND" \
    SOUNIO_SELFHOST_EXECUTOR="$SELFHOST_EXECUTOR" \
    "$souc_bin" run "$FULL_SELFHOST_TARGET" >"$log_file" 2>&1
}

run_capture() {
  local out_file="$1"
  local err_file="$2"
  local code_file="$3"
  local env_prefix="$4"
  local cmd="$5"

  set +e
  if [ -n "$env_prefix" ]; then
    bash -lc "$env_prefix \"$SOUC_BIN\" $cmd" >"$out_file" 2>"$err_file"
  else
    bash -lc "\"$SOUC_BIN\" $cmd" >"$out_file" 2>"$err_file"
  fi
  local code=$?
  set -e

  echo "$code" >"$code_file"
}

detect_case_blocker_reason() {
  local stderr_file="$1"

  if [ ! -f "$stderr_file" ]; then
    return 0
  fi

  if [ "$(count_matches 'SELFHOST_DRIVER_HARNESS_UNAVAILABLE' "$stderr_file")" -gt 0 ]; then
    echo "candidate blocked: SELFHOST_DRIVER_HARNESS_UNAVAILABLE"
    return 0
  fi

  if [ "$(count_matches 'SELFHOST_BOOTSTRAP_ARTIFACTS_MISSING' "$stderr_file")" -gt 0 ]; then
    echo "candidate blocked: SELFHOST_BOOTSTRAP_ARTIFACTS_MISSING"
    return 0
  fi

  if [ "$(count_matches 'SELFHOST_SIGNED_HARNESS_REQUIRED|SELFHOST_SIGNED_HARNESS_MISSING' "$stderr_file")" -gt 0 ]; then
    echo "candidate blocked: SELFHOST_SIGNED_HARNESS_UNAVAILABLE"
    return 0
  fi
}

compare_case() {
  local case_id="$1"
  local mode="$2"
  local compare_spec="$3"
  local base_prefix="$4"
  local cand_prefix="$5"
  local base_exit=""
  local cand_exit=""
  local blocker_reason=""

  IFS=',' read -r -a checks <<<"$compare_spec"
  for check in "${checks[@]}"; do
    case "$check" in
      exit)
        if ! cmp -s "$base_prefix.exit" "$cand_prefix.exit"; then
          base_exit="$(tr -d '\n' <"$base_prefix.exit")"
          cand_exit="$(tr -d '\n' <"$cand_prefix.exit")"
          blocker_reason="$(detect_case_blocker_reason "$cand_prefix.stderr")"
          if [ -n "$blocker_reason" ] && [[ "$mode" =~ ^run_(selfhost|default_vs_compat)$ ]]; then
            not_run "$case_id" "baseline=$base_exit candidate=$cand_exit; $blocker_reason"
          elif [ -n "$blocker_reason" ]; then
            fail "$case_id" "exit code mismatch (baseline=$base_exit candidate=$cand_exit; $blocker_reason)"
          else
            fail "$case_id" "exit code mismatch (baseline=$base_exit candidate=$cand_exit)"
          fi
          return 1
        fi
        ;;
      stdout)
        if ! cmp -s "$base_prefix.stdout" "$cand_prefix.stdout"; then
          fail "$case_id" "stdout mismatch"
          return 1
        fi
        ;;
      stderr)
        if ! cmp -s "$base_prefix.stderr" "$cand_prefix.stderr"; then
          fail "$case_id" "stderr mismatch"
          return 1
        fi
        ;;
      *)
        fail "$case_id" "unknown compare token '$check'"
        return 1
        ;;
    esac
  done

  pass "$case_id" "baseline and candidate match ($compare_spec)"
}

extract_full_selfhost_markers() {
  local log_file="$1"
  local marker_file="$2"
  local exit_code="$3"

  local parse_all_header_count
  local parse_all_summary_count
  local parse_all_count
  local all_passed_count
  local suite_fail_detail_count
  local suite_fail_summary_count
  local suite_fail_count
  local rust_fallback_count
  local rust_oracle_count

  parse_all_header_count="$(count_matches '=== Parse All ===' "$log_file")"
  parse_all_summary_count="$(count_matches 'Parse All:' "$log_file")"
  parse_all_count=$((parse_all_header_count + parse_all_summary_count))
  all_passed_count="$(count_matches 'Suites: all passed' "$log_file")"
  suite_fail_detail_count="$(count_matches 'SUITE FAIL:' "$log_file")"
  suite_fail_summary_count="$(count_matches 'Suites: suite fail' "$log_file")"
  suite_fail_count=$((suite_fail_detail_count + suite_fail_summary_count))
  rust_fallback_count="$(count_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=fallback' "$log_file")"
  rust_oracle_count="$(count_matches 'SELFHOST=oracle backend=rust' "$log_file")"

  {
    echo "exit=$exit_code"
    echo "parse_all=$parse_all_count"
    echo "parse_all_header=$parse_all_header_count"
    echo "parse_all_summary=$parse_all_summary_count"
    echo "all_passed=$all_passed_count"
    echo "suite_fail=$suite_fail_count"
    echo "suite_fail_detail=$suite_fail_detail_count"
    echo "suite_fail_summary=$suite_fail_summary_count"
    echo "rust_fallback=$rust_fallback_count"
    echo "rust_oracle=$rust_oracle_count"
  } >"$marker_file"
}

is_full_selfhost_success() {
  local marker_file="$1"
  local exit_code
  local parse_all_header_count
  local parse_all_summary_count
  local parse_all_count
  local all_passed_count
  local suite_fail_count
  local rust_fallback_count
  local rust_oracle_count

  exit_code="$(marker_value "$marker_file" "exit")"
  parse_all_count="$(marker_value "$marker_file" "parse_all")"
  parse_all_header_count="$(marker_value "$marker_file" "parse_all_header")"
  parse_all_summary_count="$(marker_value "$marker_file" "parse_all_summary")"
  all_passed_count="$(marker_value "$marker_file" "all_passed")"
  suite_fail_count="$(marker_value "$marker_file" "suite_fail")"
  rust_fallback_count="$(marker_value "$marker_file" "rust_fallback")"
  rust_oracle_count="$(marker_value "$marker_file" "rust_oracle")"

  if [ "$exit_code" -ne 0 ]; then
    return 1
  fi

  # Keep marker checks strict to avoid false positives from partial output.
  if [ "$parse_all_count" -le 0 ] || [ "$parse_all_header_count" -le 0 ] || [ "$parse_all_summary_count" -le 0 ]; then
    return 1
  fi

  if [ "$all_passed_count" -le 0 ] || [ "$suite_fail_count" -ne 0 ]; then
    return 1
  fi

  if [ "$rust_fallback_count" -ne 0 ] || [ "$rust_oracle_count" -ne 0 ]; then
    return 1
  fi

  return 0
}

classify_full_selfhost_failure() {
  local current_marker_file="$1"
  local baseline_log="$LOG_DIR/full_selfhost.baseline.log"
  local baseline_marker_file="$LOG_DIR/full_selfhost.baseline.markers"

  if [ "$BASELINE_COMPARE" != "1" ]; then
    echo "classification=skipped reason=POSEIDON_BASELINE_COMPARE=$BASELINE_COMPARE"
    return 0
  fi

  if ! git cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null; then
    echo "classification=unknown reason=missing_commit commit=$BASELINE_COMMIT"
    return 0
  fi

  if [ -d "$BASELINE_WORKTREE" ]; then
    git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || rm -rf "$BASELINE_WORKTREE"
  fi

  if ! git worktree add --detach "$BASELINE_WORKTREE" "$BASELINE_COMMIT" >"$LOG_DIR/baseline_worktree.stdout" 2>"$LOG_DIR/baseline_worktree.stderr"; then
    echo "classification=unknown reason=worktree_add_failed commit=$BASELINE_COMMIT"
    return 0
  fi

  if [ ! -f "$BASELINE_WORKTREE/Cargo.toml" ]; then
    echo "classification=unknown reason=baseline_manifest_missing commit=$BASELINE_COMMIT"
    git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || true
    return 0
  fi

  if ! run_with_timeout 900 bash -lc "cd \"$BASELINE_WORKTREE\" && cargo build -p souc" >"$LOG_DIR/baseline_build.stdout" 2>"$LOG_DIR/baseline_build.stderr"; then
    echo "classification=unknown reason=baseline_build_failed commit=$BASELINE_COMMIT"
    git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || true
    return 0
  fi

  set +e
  run_full_selfhost "$BASELINE_SOUC" "$baseline_log"
  local baseline_exit=$?
  set -e
  extract_full_selfhost_markers "$baseline_log" "$baseline_marker_file" "$baseline_exit"

  local current_success=0
  local baseline_success=0
  if is_full_selfhost_success "$current_marker_file"; then
    current_success=1
  fi
  if is_full_selfhost_success "$baseline_marker_file"; then
    baseline_success=1
  fi

  git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || true

  if [ "$current_success" = "1" ]; then
    echo "classification=none reason=current_passed"
    return 0
  fi

  if [ "$baseline_success" = "1" ]; then
    echo "classification=branch_regression baseline_commit=$BASELINE_COMMIT"
    return 0
  fi

  local current_exit current_parse current_parse_header current_parse_summary current_all current_fail current_rust_fallback current_rust_oracle
  local baseline_exit_v baseline_parse baseline_parse_header baseline_parse_summary baseline_all baseline_fail baseline_rust_fallback baseline_rust_oracle
  current_exit="$(marker_value "$current_marker_file" "exit")"
  current_parse="$(marker_value "$current_marker_file" "parse_all")"
  current_parse_header="$(marker_value "$current_marker_file" "parse_all_header")"
  current_parse_summary="$(marker_value "$current_marker_file" "parse_all_summary")"
  current_all="$(marker_value "$current_marker_file" "all_passed")"
  current_fail="$(marker_value "$current_marker_file" "suite_fail")"
  current_rust_fallback="$(marker_value "$current_marker_file" "rust_fallback")"
  current_rust_oracle="$(marker_value "$current_marker_file" "rust_oracle")"
  baseline_exit_v="$(marker_value "$baseline_marker_file" "exit")"
  baseline_parse="$(marker_value "$baseline_marker_file" "parse_all")"
  baseline_parse_header="$(marker_value "$baseline_marker_file" "parse_all_header")"
  baseline_parse_summary="$(marker_value "$baseline_marker_file" "parse_all_summary")"
  baseline_all="$(marker_value "$baseline_marker_file" "all_passed")"
  baseline_fail="$(marker_value "$baseline_marker_file" "suite_fail")"
  baseline_rust_fallback="$(marker_value "$baseline_marker_file" "rust_fallback")"
  baseline_rust_oracle="$(marker_value "$baseline_marker_file" "rust_oracle")"

  echo "classification=pre_existing baseline_commit=$BASELINE_COMMIT current(exit=$current_exit parse_all=$current_parse parse_all_header=$current_parse_header parse_all_summary=$current_parse_summary all_passed=$current_all suite_fail=$current_fail rust_fallback=$current_rust_fallback rust_oracle=$current_rust_oracle) baseline(exit=$baseline_exit_v parse_all=$baseline_parse parse_all_header=$baseline_parse_header parse_all_summary=$baseline_parse_summary all_passed=$baseline_all suite_fail=$baseline_fail rust_fallback=$baseline_rust_fallback rust_oracle=$baseline_rust_oracle)"
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"
run_poseidon_wrapper_checks

if [ "$BUILD_SOUC" = "1" ]; then
  if ! has_repo_cargo_manifest; then
    pass "build" "skipped cargo build -p souc; repo Cargo.toml missing in this checkout"
  elif run_with_timeout 900 cargo build -p souc >"$LOG_DIR/build.stdout" 2>"$LOG_DIR/build.stderr"; then
    pass "build" "cargo build -p souc"
  else
    fail "build" "cargo build failed (see $LOG_DIR/build.stderr)"
  fi
fi

if [ ! -x "$SOUC_BIN" ]; then
  fail "preflight" "compiler binary not found at $SOUC_BIN"
  echo
  echo "Summary: PASS=$PASS_COUNT FAIL=$FAIL_COUNT NOT_RUN=$NOT_RUN_COUNT"
  exit 1
fi

if [ ! -f "$MATRIX_FILE" ]; then
  fail "preflight" "matrix file missing: $MATRIX_FILE"
  echo
  echo "Summary: PASS=$PASS_COUNT FAIL=$FAIL_COUNT NOT_RUN=$NOT_RUN_COUNT"
  exit 1
fi

while IFS='|' read -r case_id mode command compare_spec; do
  if [[ -z "${case_id:-}" ]] || [[ "$case_id" =~ ^# ]]; then
    continue
  fi

  base_prefix="$ARTIFACT_DIR/${case_id}.baseline"
  cand_prefix="$ARTIFACT_DIR/${case_id}.candidate"

  case "$mode" in
    run_selfhost|run_default_vs_compat)
      common_selfhost_env="SOUNIO_PARITY_BACKEND=$SELFHOST_BACKEND SOUNIO_SELFHOST_EXECUTOR=$SELFHOST_EXECUTOR"
      run_capture \
        "$base_prefix.stdout" \
        "$base_prefix.stderr" \
        "$base_prefix.exit" \
        "$common_selfhost_env" \
        "$command"

      run_capture \
        "$cand_prefix.stdout" \
        "$cand_prefix.stderr" \
        "$cand_prefix.exit" \
        "$common_selfhost_env" \
        "$command --use-sounio-compiler"
      ;;
    direct)
      run_capture \
        "$base_prefix.stdout" \
        "$base_prefix.stderr" \
        "$base_prefix.exit" \
        "" \
        "$command"

      run_capture \
        "$cand_prefix.stdout" \
        "$cand_prefix.stderr" \
        "$cand_prefix.exit" \
        "" \
        "$command"
      ;;
    *)
      fail "$case_id" "unknown mode '$mode'"
      continue
      ;;
  esac

  compare_case "$case_id" "$mode" "$compare_spec" "$base_prefix" "$cand_prefix" || true
done <"$MATRIX_FILE"

if [ "$RUN_ORACLE_PARITY" = "1" ]; then
  fail "oracle-parity" "POSEIDON_ORACLE_PARITY=1 is unsupported after no-rust cutover"
fi

if [ "$SKIP_FULL_SELFHOST" != "1" ]; then
  FULL_LOG="$LOG_DIR/full_selfhost.log"
  FULL_MARKERS="$LOG_DIR/full_selfhost.markers"
  set +e
  run_full_selfhost "$SOUC_BIN" "$FULL_LOG"
  full_code=$?
  set -e
  extract_full_selfhost_markers "$FULL_LOG" "$FULL_MARKERS" "$full_code"

  parse_all_count="$(marker_value "$FULL_MARKERS" "parse_all")"
  parse_all_header_count="$(marker_value "$FULL_MARKERS" "parse_all_header")"
  parse_all_summary_count="$(marker_value "$FULL_MARKERS" "parse_all_summary")"
  all_passed_count="$(marker_value "$FULL_MARKERS" "all_passed")"
  suite_fail_count="$(marker_value "$FULL_MARKERS" "suite_fail")"
  rust_fallback_count="$(marker_value "$FULL_MARKERS" "rust_fallback")"
  rust_oracle_count="$(marker_value "$FULL_MARKERS" "rust_oracle")"

  if is_full_selfhost_success "$FULL_MARKERS"; then
    pass "full-selfhost" "self-hosted suite completed with required markers and no rust fallback/oracle markers"
  else
    fail "full-selfhost" "gate failed (exit=$full_code parse_all=$parse_all_count parse_all_header=$parse_all_header_count parse_all_summary=$parse_all_summary_count all_passed=$all_passed_count suite_fail=$suite_fail_count rust_fallback=$rust_fallback_count rust_oracle=$rust_oracle_count)"
    classification="$(classify_full_selfhost_failure "$FULL_MARKERS")"
    if [ -n "$classification" ]; then
      echo "INFO [full-selfhost] $classification"
    fi
  fi
else
  pass "full-selfhost" "skipped by POSEIDON_SKIP_FULL_SELFHOST=1"
fi

echo
echo "Summary: PASS=$PASS_COUNT FAIL=$FAIL_COUNT NOT_RUN=$NOT_RUN_COUNT"
echo "Artifacts: $WORK_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
