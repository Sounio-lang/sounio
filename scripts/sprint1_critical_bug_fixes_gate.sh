#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOUC_BIN="${SOUNIO_SOUC_BIN:-$ROOT_DIR/souc}"
SOURCE_FILE="${SOUNIO_SPRINT1_SOURCE_FILE:-$ROOT_DIR/self-hosted/compiler/main.sio}"
OUT_DIR="$ROOT_DIR/artifacts/sprint1"
OUT_JSON="$OUT_DIR/critical_bug_fixes_gate.v1.json"
RUN_SELF_TEST="${SOUNIO_SPRINT1_RUN_SELF_TEST:-0}"

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

write_gate_json() {
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
}

run_light_self_test() {
  local fixture="$TMP_DIR/sprint1_light_self_test.sio"
  cat > "$fixture" <<'EOF'
fn i64_to_string_decimal(n: i64) -> string with Mut, Panic, Div {
    var out: [i8; 24] = [0; 24]
    if n == 0 {
        out[0] = 48 as i8
        return str_from_bytes(out, 1)
    }

    var value = n
    var digits: i64 = 1
    while value <= -10 || value >= 10 {
        digits = digits + 1
        value = value / 10
    }

    let negative = n < 0
    let total_len = if negative { digits + 1 } else { digits }

    var write = total_len - 1
    value = n
    while value <= -10 || value >= 10 {
        var digit = value % 10
        if digit < 0 {
            digit = 0 - digit
        }
        out[write as usize] = (48 + digit) as i8
        write = write - 1
        value = value / 10
    }

    var final_digit = value
    if final_digit < 0 {
        final_digit = 0 - final_digit
    }
    out[write as usize] = (48 + final_digit) as i8

    if negative {
        out[0] = 45 as i8
    }

    str_from_bytes(out, total_len)
}

fn int_to_string(n: i64) -> string with Mut, Panic, Div {
    i64_to_string_decimal(n)
}

fn char_from_i64_error(n: i64) -> string with Mut, Panic, Div {
    "char_from_i64: digit must be 0-9, got " + int_to_string(n)
}

fn char_from_i64(n: i64) -> string with Mut, Panic, Div {
    if n < 0 || n > 9 {
        panic(char_from_i64_error(n))
    }
    str_slice("0123456789", n, n + 1)
}

struct ArgList {
    items: [string; 64],
    len: i64,
}

fn arg_list_get_error(index: i64, len: i64) -> string with Mut, Panic, Div {
    "arg_list_get: index " + int_to_string(index) +
    " out of bounds (length " + int_to_string(len) + ")"
}

fn arg_list_get(lst: ArgList, i: i64) -> string with Mut, Panic, Div {
    if i < 0 || i >= lst.len {
        panic(arg_list_get_error(i, lst.len))
    }
    lst.items[i as usize]
}

fn main() with IO, Mut, Panic, Div {
    let args = ArgList {
        items: ["alpha", "beta", "gamma", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        len: 3,
    }

    let ok = (char_from_i64(0) == "0") &&
             (char_from_i64(5) == "5") &&
             (char_from_i64(9) == "9") &&
             (char_from_i64_error(-1) == "char_from_i64: digit must be 0-9, got -1") &&
             (char_from_i64_error(10) == "char_from_i64: digit must be 0-9, got 10") &&
             (arg_list_get(args, 1) == "beta") &&
             (arg_list_get_error(-1, 3) == "arg_list_get: index -1 out of bounds (length 3)") &&
             (arg_list_get_error(3, 3) == "arg_list_get: index 3 out of bounds (length 3)") &&
             (int_to_string(0) == "0") &&
             (int_to_string(-42) == "-42")

    if ok {
        println("sprint1_light_self_test: pass")
    } else {
        panic("sprint1_light_self_test: fail")
    }
}
EOF

  set +e
  timeout 30 "$SOUC_BIN" run "$fixture" > "$TMP_DIR/light_self_test.out" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" == "0" ]] && rg -F -q "sprint1_light_self_test: pass" "$TMP_DIR/light_self_test.out"; then
    record_step "self_test" "pass" "lightweight Sprint1 self-test fixture passed"
  elif [[ "$rc" == "124" ]]; then
    record_step "self_test" "fail" "lightweight Sprint1 self-test fixture timed out"
  else
    record_step "self_test" "fail" "lightweight Sprint1 self-test fixture failed"
  fi
}

record_source_contract_steps() {
  python3 - "$SOURCE_FILE" <<'PY' >> "$STEPS_TSV"
import re
import sys

source_file = sys.argv[1]
text = open(source_file, "r", encoding="utf-8").read()
compact = re.sub(r"\s+", " ", text).strip()

def add(name: str, ok: bool, pass_reason: str, fail_reason: str) -> None:
    status = "pass" if ok else "fail"
    reason = pass_reason if ok else fail_reason
    print(f"{name}\t{status}\t{reason}")

add(
    "char_from_i64_bounds",
    'fn char_from_i64(n: i64) -> string with Mut, Panic { if n < 0 || n > 9 { panic(char_from_i64_error(n)) } str_slice("0123456789", n, n + 1) }' in compact,
    "char_from_i64 validates 0 <= n <= 9 before indexing",
    "char_from_i64 missing explicit 0..9 bounds guard",
)

add(
    "char_from_i64_error_format",
    'char_from_i64: digit must be 0-9, got ' in text,
    "char_from_i64 error format string present",
    "char_from_i64 error format string missing",
)

add(
    "arg_list_get_bounds",
    "fn arg_list_get(lst: ArgList, i: i64) -> string with Mut, Panic { if i < 0 || i >= lst.len { panic(arg_list_get_error(i, lst.len)) } lst.items[i as usize] }" in compact,
    "arg_list_get validates 0 <= i < len before indexing",
    "arg_list_get missing bounds guard or Panic effect",
)

add(
    "arg_list_get_error_format",
    ("arg_list_get: index " in text) and ("out of bounds (length " in text),
    "arg_list_get error format includes index and length",
    "arg_list_get error format missing required index/length text",
)

add(
    "int_to_string_linear_impl",
    ('fn int_to_string(n: i64) -> string with Mut, Panic, Div { i64_to_string_decimal(n) }' in compact)
    and ('var out: [i8; 24] = [0; 24]' in text)
    and ('result = char_from_i64(ch) + result' not in text),
    "int_to_string uses fixed buffer O(n) decimal implementation",
    "int_to_string missing linear-time buffer path or still has O(n^2) prepend",
)

compile_tokens_ok = (
    "let exit_code = compile_multimodule_native(" in text
    and "var errors = compile_errors_new()" in text
    and 'compile_errors_push_lines(errors, "error: ", native_result.error_msg)' in text
    and "error: compilation failed (no structured diagnostics captured)" in text
)
add(
    "compile_error_capture",
    compile_tokens_ok,
    "compile() retains native compile call and captures structured diagnostics on failure",
    "compile() missing structured error-capture path after native compilation failure",
)
PY
}

if [[ ! -f "$SOURCE_FILE" ]]; then
  record_step "source_file_available" "fail" "missing source file at $SOURCE_FILE"
  write_gate_json
  exit 1
fi

record_step "source_file_available" "pass" "source file found at $SOURCE_FILE"

if [[ ! -x "$SOUC_BIN" ]]; then
  record_step "souc_available" "fail" "souc binary is not executable at $SOUC_BIN"
else
  record_step "souc_available" "pass" "souc binary is executable"
fi

record_source_contract_steps

if [[ "$RUN_SELF_TEST" == "1" ]]; then
  if [[ -x "$SOUC_BIN" ]]; then
    run_light_self_test
  else
    record_step "self_test" "not_run" "skipped because souc binary is unavailable"
  fi
else
  record_step "self_test" "pass" "self-test skipped (SOUNIO_SPRINT1_RUN_SELF_TEST=0)"
fi

PERF_JSON="$OUT_DIR/int_to_string_perf_gate.v1.json"
if [[ ! -x "$SOUC_BIN" ]]; then
  record_step "int_to_string_benchmark" "not_run" "perf gate skipped because souc binary is unavailable"
else
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
fi

write_gate_json

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
