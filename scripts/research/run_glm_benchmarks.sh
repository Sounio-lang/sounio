#!/usr/bin/env bash
# No-rust self-hosted benchmark runner used by GLM workflow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

OUTPUT_DIR="benchmark_results"
LOG_FILE="benchmark_execution.log"
REPORT_FILE="benchmark_report.md"

mkdir -p "$OUTPUT_DIR"
: > "$LOG_FILE"

BLUE='\033[0;34m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

now_ns() {
  python3 - <<'PY'
import time
print(time.time_ns())
PY
}

log() {
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

check_prerequisites() {
  log "Checking no-rust benchmark prerequisites..."
  sounio_require_souc
  log "Using SOUC_BIN=$SOUC_BIN"
}

write_result_json() {
  local out_json="$1"
  local name="$2"
  local mode="$3"
  local source_file="$4"
  local command_line="$5"
  local exit_code="$6"
  local duration_ms="$7"
  local status="$8"

  python3 - "$out_json" "$name" "$mode" "$source_file" "$command_line" "$exit_code" "$duration_ms" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone

out_json, name, mode, source_file, command_line, exit_code, duration_ms, status = sys.argv[1:]
payload = {
    "schema": "sounio.glm.benchmark.v2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "benchmark": name,
    "mode": mode,
    "source": source_file,
    "command": command_line,
    "exit_code": int(exit_code),
    "duration_ms": float(duration_ms),
    "status": status,
}
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY
}

run_case() {
  local name="$1"
  local mode="$2"
  local source_file="$3"
  local out_prefix="$OUTPUT_DIR/${name}_${mode}"
  local out_log="${out_prefix}.log"
  local out_json="${out_prefix}.json"

  if [[ ! -f "$source_file" ]]; then
    log "SKIP $name ($mode): missing source file $source_file"
    write_result_json "$out_json" "$name" "$mode" "$source_file" "missing" 0 0 "skipped"
    return 0
  fi

  local cmd=("$SOUC_BIN" "$mode" "$source_file")
  local command_line
  command_line="${cmd[*]}"

  echo -e "${YELLOW}Running benchmark ${name} (${mode})${NC}" | tee -a "$LOG_FILE"
  local start_ns end_ns elapsed_ns elapsed_ms rc
  start_ns="$(now_ns)"
  set +e
  "${cmd[@]}" >"$out_log" 2>&1
  rc=$?
  set -e
  end_ns="$(now_ns)"
  elapsed_ns=$((end_ns - start_ns))
  elapsed_ms="$(python3 - <<PY
print(round(${elapsed_ns} / 1_000_000.0, 3))
PY
)"

  local status="pass"
  if [[ "$rc" -ne 0 ]]; then
    status="fail"
    log "FAIL $name ($mode): rc=$rc"
  else
    log "PASS $name ($mode): ${elapsed_ms} ms"
  fi

  write_result_json "$out_json" "$name" "$mode" "$source_file" "$command_line" "$rc" "$elapsed_ms" "$status"
  return "$rc"
}

generate_report() {
  log "Generating benchmark report..."
  python3 - "$OUTPUT_DIR" "$REPORT_FILE" <<'PY'
import json
import pathlib
import statistics
import sys
from datetime import datetime, timezone

results_dir = pathlib.Path(sys.argv[1])
report_path = pathlib.Path(sys.argv[2])
results = []
for path in sorted(results_dir.glob("*.json")):
    try:
        results.append(json.loads(path.read_text()))
    except Exception:
        continue

passed = [r for r in results if r.get("status") == "pass"]
failed = [r for r in results if r.get("status") == "fail"]
skipped = [r for r in results if r.get("status") == "skipped"]
durations = [float(r.get("duration_ms", 0.0)) for r in passed]
avg = statistics.mean(durations) if durations else 0.0

lines = [
    "# GLM-4.7 Performance Benchmark Report (No-Rust Self-Hosted)",
    "",
    f"Generated on: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
    "",
    "## Summary",
    "",
    f"- Total cases: {len(results)}",
    f"- Passed: {len(passed)}",
    f"- Failed: {len(failed)}",
    f"- Skipped: {len(skipped)}",
    f"- Mean duration (pass): {avg:.3f} ms",
    "",
    "## Results",
    "",
    "| Benchmark | Mode | Status | Duration (ms) | Exit |",
    "|---|---|---|---:|---:|",
]

for row in results:
    lines.append(
        f"| {row.get('benchmark')} | {row.get('mode')} | {row.get('status')} | {float(row.get('duration_ms', 0.0)):.3f} | {int(row.get('exit_code', 0))} |"
    )

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

  cat > "$OUTPUT_DIR/trends.md" <<'EOF_TRENDS'
# Performance Trend Analysis

## No-Rust Self-Hosted Baseline

- Benchmarks are executed with pinned `SOUC_BIN`.
- Results are reported per `.sio` contract case.
- Historical trend comparison can be layered on top of `benchmark_results/*.json`.
EOF_TRENDS
}

run_suite() {
  local -a cases=(
    "hello:run:examples/hello.sio"
    "complex_arithmetic:run:tests/run-pass/complex_arithmetic.sio"
    "accumulator_bounds:run:tests/hardware/accumulator_bounds_test.sio"
    "epistemic_check:check:examples/epistemic_bmi.sio"
  )

  local limit="$1"
  local idx=0
  local failed=0

  for case_def in "${cases[@]}"; do
    idx=$((idx + 1))
    if [[ "$limit" -gt 0 && "$idx" -gt "$limit" ]]; then
      break
    fi
    IFS=':' read -r name mode source_file <<<"$case_def"
    if ! run_case "$name" "$mode" "$source_file"; then
      failed=1
    fi
  done

  generate_report
  return "$failed"
}

show_help() {
  cat <<'EOF_HELP'
GLM-4.7 Performance Benchmark Suite (No-Rust)

Usage: scripts/research/run_glm_benchmarks.sh [COMMAND]

Commands:
  check           Check prerequisites
  quick           Run a quick subset of benchmark cases
  comprehensive   Run the full benchmark suite
  regression      Run full suite (alias for comprehensive)
  monitor         Run one comprehensive cycle (compat mode)
  help            Show this help message
EOF_HELP
}

main() {
  local command="${1:-comprehensive}"

  echo -e "${BLUE}============================================${NC}"
  echo -e "${BLUE}  GLM-4.7 Benchmark Suite (No-Rust Mode)   ${NC}"
  echo -e "${BLUE}============================================${NC}"

  check_prerequisites

  case "$command" in
    check)
      log "Prerequisite check completed"
      ;;
    quick)
      run_suite 2
      ;;
    comprehensive)
      run_suite 0
      ;;
    regression)
      run_suite 0
      ;;
    monitor)
      run_suite 0
      ;;
    help|-h|--help)
      show_help
      ;;
    *)
      echo -e "${RED}Unknown command: $command${NC}"
      show_help
      exit 2
      ;;
  esac

  echo -e "${GREEN}Benchmark execution finished.${NC}"
}

main "$@"
