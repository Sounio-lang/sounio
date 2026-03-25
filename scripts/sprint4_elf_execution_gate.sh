#!/usr/bin/env bash
# sprint4_elf_execution_gate.sh — selfhosted native ELF execution correctness gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JIT_BIN="${SOUNIO_JIT_BIN:-$ROOT_DIR/bin/souc}"
[ ! -x "$JIT_BIN" ] && JIT_BIN="$ROOT_DIR/souc"

OUT_JSON="${SOUNIO_SPRINT4_ELF_GATE_OUT:-$ROOT_DIR/artifacts/sprint4/elf_execution_gate.v1.json}"
LOG_DIR="${SOUNIO_SPRINT4_ELF_LOG_DIR:-$ROOT_DIR/artifacts/sprint4/logs}"
TIMEOUT_SECS="${SOUNIO_SPRINT4_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR" artifacts/sprint4/out

CASES_TSV="$LOG_DIR/cases.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

run_case() {
  local name="$1" fixture="$2" expected_exit="$3"
  local out_elf="$ROOT_DIR/artifacts/sprint4/out/${name}.out"
  local log="$LOG_DIR/${name}.log"

  # Compile via selfhosted compiler
  set +e
  timeout "$TIMEOUT_SECS" "$JIT_BIN" run self-hosted/compiler/main.sio \
    -- --target selfhosted-native --output "$out_elf" "$fixture" \
    > "$log" 2>&1
  local compile_rc=$?
  set -e

  if [ $compile_rc -eq 124 ]; then
    printf '%s\tnot_run\ttimeout\t%d\t-1\t\n' "$name" "$expected_exit" >> "$CASES_TSV"
    return
  fi
  if [ $compile_rc -ne 0 ] || [ ! -f "$out_elf" ]; then
    printf '%s\tfail\tcompile_failed_rc_%d\t%d\t-1\t\n' "$name" "$compile_rc" "$expected_exit" >> "$CASES_TSV"
    return
  fi

  chmod +x "$out_elf"
  set +e
  timeout 10 "$out_elf" >> "$log" 2>&1
  local actual_exit=$?
  set -e

  if [ $actual_exit -eq "$expected_exit" ]; then
    printf '%s\tpass\tok\t%d\t%d\t\n' "$name" "$expected_exit" "$actual_exit" >> "$CASES_TSV"
  else
    printf '%s\tfail\twrong_exit_%d\t%d\t%d\t\n' "$name" "$actual_exit" "$expected_exit" "$actual_exit" >> "$CASES_TSV"
  fi
}

# Gate cases: (name, fixture, expected_exit_code)
run_case "main_returns_0"  "tests/frontend/minimal_pipeline_i64.sio"  0

python3 - "$OUT_JSON" "$LOG_DIR/cases.tsv" "$JIT_BIN" "$TIMEOUT_SECS" << 'PY'
import json, datetime as dt, sys
from pathlib import Path

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
jit_bin = sys.argv[3]
timeout_secs = int(sys.argv[4])

cases = []
counts = {"pass": 0, "fail": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    parts = raw.split("\t")
    name, status, reason, expected, actual = parts[0], parts[1], parts[2], parts[3], parts[4]
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({
        "name": name, "status": status, "reason": reason,
        "expected_exit": int(expected) if expected.strip().lstrip("-").isdigit() else None,
        "actual_exit": int(actual) if actual.strip().lstrip("-").isdigit() else None,
    })

overall = "pass" if counts["fail"] == 0 and counts["not_run"] == 0 else \
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else "fail"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint4.elf_execution_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall, "reason": reason,
    "config": {"jit_bin": jit_bin, "timeout_seconds": timeout_secs},
    "metrics": {"total": len(cases), "passed": counts["pass"],
                "failed": counts["fail"], "not_run": counts["not_run"]},
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint4_elf_execution_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
