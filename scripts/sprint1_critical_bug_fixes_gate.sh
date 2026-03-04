#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOUC_BIN="${SOUNIO_SOUC_BIN:-$ROOT_DIR/souc}"
OUT_DIR="$ROOT_DIR/artifacts/sprint1"
OUT_JSON="$OUT_DIR/critical_bug_fixes_gate.v1.json"

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

STEPS_TSV="$TMP_DIR/steps.tsv"
: > "$STEPS_TSV"

record_step() {
  local name="$1"
  local status="$2"
  local reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$STEPS_TSV"
}

run_cmd() {
  local name="$1"
  local timeout_s="$2"
  shift 2
  set +e
  timeout "${timeout_s}s" "$@" > "$TMP_DIR/${name}.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$TMP_DIR/${name}.rc"
}

last_nonempty_line() {
  awk 'NF { line=$0 } END { print line }' "$1"
}

if [[ ! -x "$SOUC_BIN" ]]; then
  record_step "souc_available" "fail" "souc binary is not executable at $SOUC_BIN"
  python3 - "$STEPS_TSV" "$OUT_JSON" <<'PY'
import datetime as dt
import json
import sys

steps_path, out_path = sys.argv[1:3]
steps = []
overall = "pass"
with open(steps_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        name, status, reason = line.split("\t", 2)
        if status != "pass":
            overall = "fail"
        steps.append({"name": name, "status": status, "reason": reason})

payload = {
    "schema": "sounio.sprint1.critical_bug_fixes_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "overall_status": overall,
    "steps": steps,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"wrote: {out_path}")
PY
  exit 1
fi

run_cmd "self_test" 180 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --self-test
self_rc="$(cat "$TMP_DIR/self_test.rc")"
if [[ "$self_rc" == "0" ]] && rg -F -q "compiler/main tests: 5/5 passed" "$TMP_DIR/self_test.out"; then
  record_step "self_test" "pass" "compiler/main self-tests passed"
else
  record_step "self_test" "fail" "self-test command failed or missing 5/5 marker"
fi

run_cmd "probe_char_valid" 120 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-char-from-i64 5
char_valid_last="$(last_nonempty_line "$TMP_DIR/probe_char_valid.out")"
if [[ "$(cat "$TMP_DIR/probe_char_valid.rc")" == "0" ]] && [[ "$char_valid_last" == "5" ]]; then
  record_step "char_from_i64_valid" "pass" "probe returned 5"
else
  record_step "char_from_i64_valid" "fail" "expected last line 5, got '$char_valid_last'"
fi

run_cmd "probe_char_neg" 120 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-char-from-i64 -1
if rg -F -q "panic: char_from_i64: digit must be 0-9, got -1" "$TMP_DIR/probe_char_neg.out"; then
  record_step "char_from_i64_neg1" "pass" "panic message matched"
else
  record_step "char_from_i64_neg1" "fail" "missing panic message for -1"
fi

run_cmd "probe_char_ten" 120 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-char-from-i64 10
if rg -F -q "panic: char_from_i64: digit must be 0-9, got 10" "$TMP_DIR/probe_char_ten.out"; then
  record_step "char_from_i64_10" "pass" "panic message matched"
else
  record_step "char_from_i64_10" "fail" "missing panic message for 10"
fi

run_cmd "probe_arg_valid" 120 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-arg-list-get 1
arg_valid_last="$(last_nonempty_line "$TMP_DIR/probe_arg_valid.out")"
if [[ "$(cat "$TMP_DIR/probe_arg_valid.rc")" == "0" ]] && [[ "$arg_valid_last" == "beta" ]]; then
  record_step "arg_list_get_valid" "pass" "probe returned beta"
else
  record_step "arg_list_get_valid" "fail" "expected last line beta, got '$arg_valid_last'"
fi

run_cmd "probe_arg_len" 120 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-arg-list-get 3
if rg -F -q "panic: arg_list_get: index 3 out of bounds (length 3)" "$TMP_DIR/probe_arg_len.out"; then
  record_step "arg_list_get_len" "pass" "panic message matched"
else
  record_step "arg_list_get_len" "fail" "missing panic message for index==len"
fi

cat > "$TMP_DIR/compile_capture_invalid.sio" <<'BADEOF'
fn main() {
    let x =
}
BADEOF

run_cmd "probe_compile_capture" 25 "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-compile-capture "$TMP_DIR/compile_capture_invalid.sio"
probe_compile_rc="$(cat "$TMP_DIR/probe_compile_capture.rc")"
if [[ "$probe_compile_rc" == "124" ]]; then
  record_step "compile_error_capture" "not_run" "probe timed out before producing structured errors"
elif rg -F -q "probe_compile_capture: fail" "$TMP_DIR/probe_compile_capture.out" && rg -F -q "error:" "$TMP_DIR/probe_compile_capture.out"; then
  record_step "compile_error_capture" "pass" "structured compile errors were emitted"
else
  record_step "compile_error_capture" "fail" "probe did not emit expected structured errors"
fi

PERF_JSON="$OUT_DIR/int_to_string_perf_gate.v1.json"
set +e
bash "$ROOT_DIR/scripts/sprint1_int_to_string_perf_gate.sh" "$PERF_JSON" > "$TMP_DIR/int_to_string_perf_gate.out" 2>&1
perf_rc=$?
set -e
if [[ "$perf_rc" != "0" ]]; then
  record_step "int_to_string_benchmark" "not_run" "perf gate execution failed (rc=${perf_rc})"
elif [[ ! -f "$PERF_JSON" ]]; then
  record_step "int_to_string_benchmark" "not_run" "perf gate did not emit JSON artifact"
else
  perf_status="$(python3 - "$PERF_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(str(data.get("status", "not_run")))
PY
)"
  perf_reason="$(python3 - "$PERF_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
reason = str(data.get("reason", "unknown"))
mode = str(data.get("mode", "none"))
runner = str(data.get("runner", ""))
metrics = data.get("metrics", {})
net = metrics.get("net_seconds")
if net is None:
    print(f"{reason} (mode={mode}, runner={runner})")
else:
    print(f"{reason} (mode={mode}, runner={runner}, net_seconds={net})")
PY
)"
  case "$perf_status" in
    pass|fail|not_run)
      record_step "int_to_string_benchmark" "$perf_status" "$perf_reason"
      ;;
    *)
      record_step "int_to_string_benchmark" "not_run" "unexpected perf status: $perf_status"
      ;;
  esac
fi

python3 - "$STEPS_TSV" "$OUT_JSON" <<'PY'
import datetime as dt
import json
import sys

steps_path, out_path = sys.argv[1:3]
steps = []
overall = "pass"
for raw in open(steps_path, "r", encoding="utf-8"):
    raw = raw.rstrip("\n")
    if not raw:
        continue
    name, status, reason = raw.split("\t", 2)
    if status != "pass":
        overall = "fail"
    steps.append({"name": name, "status": status, "reason": reason})

payload = {
    "schema": "sounio.sprint1.critical_bug_fixes_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "overall_status": overall,
    "steps": steps,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

print(f"wrote: {out_path}")
print(f"overall_status={overall}")
PY

if [[ "${SOUNIO_SPRINT1_GATE_REQUIRE_PASS:-0}" == "1" ]]; then
  overall_status="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(str(data.get("overall_status", "fail")))
PY
)"
  if [[ "$overall_status" != "pass" ]]; then
    echo "sprint1_critical_bug_fixes_gate: overall_status=${overall_status} (required pass mode)" >&2
    exit 1
  fi
fi
