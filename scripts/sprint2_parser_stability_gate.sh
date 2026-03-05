#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
OUT_JSON="${SOUNIO_SPRINT2_PARSER_GATE_OUT:-$ROOT_DIR/artifacts/sprint2/parser_stability_gate.v1.json}"
LOG_DIR="${SOUNIO_SPRINT2_PARSER_GATE_LOG_DIR:-$ROOT_DIR/artifacts/sprint2/logs/parser_stability}"
TIMEOUT_SECS="${SOUNIO_SPRINT2_PARSER_TIMEOUT_SECS:-8}"
FUZZ_CASE_COUNT="${SOUNIO_SPRINT2_PARSER_FUZZ_CASES:-10}"
FUZZ_SEED="${SOUNIO_SPRINT2_PARSER_FUZZ_SEED:-20260305}"

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR"

CASES_TSV="$LOG_DIR/cases.tsv"
FUZZ_MANIFEST="$LOG_DIR/fuzz_manifest.tsv"
FUZZ_DIR="$LOG_DIR/generated_fuzz"
: > "$CASES_TSV"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

append_case_result() {
  local name="$1"
  local suite="$2"
  local expectation="$3"
  local source_path="$4"
  local log_path="$5"
  local status="$6"
  local reason="$7"
  local exit_code="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" \
    "$suite" \
    "$expectation" \
    "$(to_rel "$source_path")" \
    "$(to_rel "$log_path")" \
    "$status" \
    "$reason" \
    "$exit_code" >> "$CASES_TSV"
}

CAN_RUN=1
CAN_RUN_REASON="ok"
if [ ! -x "$SOUC_BIN" ]; then
  CAN_RUN=0
  CAN_RUN_REASON="souc_bin_missing"
elif ! command -v timeout >/dev/null 2>&1; then
  CAN_RUN=0
  CAN_RUN_REASON="timeout_command_missing"
fi

run_case() {
  local name="$1"
  local suite="$2"
  local expectation="$3"
  local source_path="$4"
  local log_path="$5"

  if [ ! -f "$source_path" ]; then
    append_case_result "$name" "$suite" "$expectation" "$source_path" "$log_path" "not_run" "case_file_missing" "-1"
    return
  fi

  if [ "$CAN_RUN" -ne 1 ]; then
    append_case_result "$name" "$suite" "$expectation" "$source_path" "$log_path" "not_run" "$CAN_RUN_REASON" "-1"
    return
  fi

  set +e
  timeout "$TIMEOUT_SECS" "$SOUC_BIN" check "$source_path" >"$log_path" 2>&1
  local rc=$?
  set -e

  local status="fail"
  local reason="unexpected_case_result"
  if [ "$rc" -eq 124 ]; then
    status="timeout"
    reason="timeout"
  elif [ "$expectation" = "accept" ]; then
    if [ "$rc" -eq 0 ]; then
      status="pass"
      reason="accepted_expected_valid"
    else
      status="fail"
      reason="rejected_expected_valid"
    fi
  else
    if [ "$rc" -ne 0 ]; then
      status="pass"
      reason="rejected_expected_invalid"
    else
      status="fail"
      reason="accepted_expected_invalid"
    fi
  fi

  append_case_result "$name" "$suite" "$expectation" "$source_path" "$log_path" "$status" "$reason" "$rc"
}

corpus_valid=(
  "corpus_valid_basic_arith:tests/frontend/parser_stability/valid/basic_arith.sio"
  "corpus_valid_simple_fn_call:tests/frontend/parser_stability/valid/simple_fn_call.sio"
)

corpus_invalid=(
  "corpus_invalid_missing_paren:tests/frontend/parser_stability/invalid/missing_paren.sio"
  "corpus_invalid_bad_fn_signature:tests/frontend/parser_stability/invalid/bad_fn_signature.sio"
)

for entry in "${corpus_valid[@]}"; do
  name="${entry%%:*}"
  source_path="$ROOT_DIR/${entry#*:}"
  run_case "$name" "corpus_valid" "accept" "$source_path" "$LOG_DIR/${name}.log"
done

for entry in "${corpus_invalid[@]}"; do
  name="${entry%%:*}"
  source_path="$ROOT_DIR/${entry#*:}"
  run_case "$name" "corpus_invalid" "reject" "$source_path" "$LOG_DIR/${name}.log"
done

python3 - "$FUZZ_CASE_COUNT" "$FUZZ_SEED" "$FUZZ_DIR" "$FUZZ_MANIFEST" <<'PY'
from pathlib import Path
import random
import sys

count = int(sys.argv[1])
seed = int(sys.argv[2])
out_dir = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(seed)
valid_templates = [
    """fn main() -> i32 {{
    let a = {a}
    let b = {b}
    if a + b == {sum} {{
        return 0
    }}
    return 1
}}
""",
    """fn clamp(x: i32, lo: i32, hi: i32) -> i32 {{
    if x < lo {{
        return lo
    }}
    if x > hi {{
        return hi
    }}
    return x
}}

fn main() -> i32 {{
    let v = clamp({x}, {lo}, {hi})
    if v == {expected} {{
        return 0
    }}
    return 1
}}
""",
]
invalid_templates = [
    """fn main() -> i32 {{
    let broken = ({a} + {b}
    return broken
}}
""",
    """fn main( -> i32 {{
    return {ret}
}}
""",
    """fn helper(v: i32) -> i32 {{
    return v
}}

fn main() -> i32 {{
    let out = helper({a})
    return out
}} +
""",
]

with manifest_path.open("w", encoding="utf-8") as manifest:
    for idx in range(count):
        if idx % 2 == 0:
            a = rng.randint(0, 30)
            b = rng.randint(0, 30)
            x = rng.randint(-10, 20)
            lo = rng.randint(-10, 5)
            hi = rng.randint(6, 20)
            if lo > hi:
                lo, hi = hi, lo
            expected = min(max(x, lo), hi)
            template = valid_templates[rng.randrange(len(valid_templates))]
            text = template.format(a=a, b=b, sum=a + b, x=x, lo=lo, hi=hi, expected=expected)
            suite = "fuzz_valid"
            expectation = "accept"
            name = f"fuzz_valid_{idx:02d}"
        else:
            a = rng.randint(0, 30)
            b = rng.randint(0, 30)
            ret = rng.randint(0, 30)
            template = invalid_templates[rng.randrange(len(invalid_templates))]
            text = template.format(a=a, b=b, ret=ret)
            suite = "fuzz_invalid"
            expectation = "reject"
            name = f"fuzz_invalid_{idx:02d}"

        case_path = out_dir / f"{name}.sio"
        case_path.write_text(text, encoding="utf-8")
        manifest.write(f"{name}\t{suite}\t{expectation}\t{case_path}\n")
PY

while IFS=$'\t' read -r name suite expectation source_path; do
  [ -n "$name" ] || continue
  run_case "$name" "$suite" "$expectation" "$source_path" "$LOG_DIR/${name}.log"
done < "$FUZZ_MANIFEST"

python3 - "$ROOT_DIR" "$OUT_JSON" "$CASES_TSV" "$TIMEOUT_SECS" "$FUZZ_CASE_COUNT" "$FUZZ_SEED" "$SOUC_BIN" "$LOG_DIR" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
out_json = Path(sys.argv[2])
cases_tsv = Path(sys.argv[3])
timeout_secs = int(sys.argv[4])
fuzz_case_count = int(sys.argv[5])
fuzz_seed = int(sys.argv[6])
souc_bin = Path(sys.argv[7])
log_dir = Path(sys.argv[8])


def to_rel(raw: str) -> str:
    path = Path(raw)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return raw


cases = []
status_counts = {"pass": 0, "fail": 0, "timeout": 0, "not_run": 0}
suite_counts = {}
for line in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    fields = line.split("\t")
    if len(fields) != 8:
        continue
    name, suite, expectation, source_path, log_path, status, reason, rc = fields
    if status not in status_counts:
        status = "fail"
        reason = "invalid_status_token"
    status_counts[status] += 1
    suite_counts[suite] = suite_counts.get(suite, 0) + 1
    cases.append(
        {
            "name": name,
            "suite": suite,
            "expectation": expectation,
            "status": status,
            "reason": reason,
            "exit_code": int(rc),
            "source_path": source_path,
            "log_path": log_path,
        }
    )

overall_status = "pass"
overall_reason = "all_cases_passed"
if status_counts["fail"] > 0:
    overall_status = "fail"
    overall_reason = "one_or_more_cases_failed"
elif status_counts["timeout"] > 0:
    overall_status = "timeout"
    overall_reason = "one_or_more_cases_timed_out"
elif status_counts["not_run"] > 0:
    overall_status = "not_run"
    overall_reason = "one_or_more_cases_not_run"

payload = {
    "schema": "sounio.sprint2.parser_stability_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall_status,
    "reason": overall_reason,
    "config": {
        "souc_bin": to_rel(str(souc_bin)),
        "timeout_seconds": timeout_secs,
        "fuzz_case_count": fuzz_case_count,
        "fuzz_seed": fuzz_seed,
        "log_dir": to_rel(str(log_dir)),
    },
    "metrics": {
        # Canonical flat fields for CI jq contracts.
        "total": len(cases),
        "passed": status_counts["pass"],
        "failed": status_counts["fail"],
        "timeout": status_counts["timeout"],
        "not_run": status_counts["not_run"],
        # Compatibility aliases retained for local tooling.
        "total_cases": len(cases),
        "timed_out": status_counts["timeout"],
        "by_suite": suite_counts,
    },
    "cases": cases,
}

out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall_status}")
print(f"cases={len(cases)} pass={status_counts['pass']} fail={status_counts['fail']} timeout={status_counts['timeout']} not_run={status_counts['not_run']}")
PY

overall_status="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys

obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "fail"))
PY
)"

echo "parser_stability_gate: out_json=$(to_rel "$OUT_JSON") status=$overall_status"
if [ "$overall_status" = "pass" ]; then
  exit 0
fi
exit 2
