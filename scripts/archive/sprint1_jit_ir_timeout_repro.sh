#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${1:-$ROOT_DIR/artifacts/sprint1/jit_ir_timeout_repro.v1.json}"
SOUC_JIT_VERSION="${SOUNIO_SOUC_JIT_VERSION:-0.100.3-jit.2}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_JSON")"

contains_literal() {
  local needle="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q "$needle" "$file"
  else
    grep -F -q "$needle" "$file"
  fi
}

emit_not_run() {
  local reason="$1"
  python3 - "$OUT_JSON" "$reason" <<'PY'
import datetime as dt
import json
import sys

out_path, reason = sys.argv[1:3]
payload = {
    "schema": "sounio.sprint1.jit_ir_timeout_repro.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "not_run",
    "reason": reason,
    "runner": "",
    "lane_split": "unknown",
    "summary": {
        "total_probes": 0,
        "pass_count": 0,
        "timeout_count": 0,
        "stack_overflow_count": 0,
        "jit_panic_count": 0,
        "compile_error_count": 0,
    },
    "results": [],
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"wrote: {out_path}")
print(f"status=not_run reason={reason}")
PY
}

write_arith_probe_fixture() {
  local out_path="$1"
  cat > "$out_path" <<'PROBE'
fn main() -> i32 with IO {
    var i: i64 = 0
    var acc: i64 = 0
    while i < 1000 {
        acc = acc + i
        i = i + 1
    }
    if acc == 499500 {
        println("jit_probe_arith_ok")
        0
    } else {
        println("jit_probe_arith_bad")
        1
    }
}
PROBE
}

ARITH_PROBE_FIXTURE="$TMP_DIR/jit_probe_arith.sio"
write_arith_probe_fixture "$ARITH_PROBE_FIXTURE"

declare -a candidates=()
if [[ -n "${SOUNIO_SOUC_JIT_BIN:-}" ]]; then
  candidates+=("${SOUNIO_SOUC_JIT_BIN}")
fi
if [[ -n "${SOUNIO_SOUC_BIN:-}" ]]; then
  candidates+=("${SOUNIO_SOUC_BIN}")
fi
candidates+=(
  "$ROOT_DIR/bin/souc"
  "$ROOT_DIR/souc"
  "$ROOT_DIR/target/debug/souc"
  "$ROOT_DIR/target/release/souc"
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64"
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
)

resolver="$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh"
if [[ -x "$resolver" ]]; then
  set +e
  resolver_candidate="$(
    SOUNIO_SOUC_VERSION="$SOUC_JIT_VERSION" \
    SOUNIO_SOUC_VARIANT="jit" \
    OMEGA_SOUC_REQUIRE_PINNED=1 \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
      "$resolver" --print-path 2>/dev/null
  )"
  resolver_rc=$?
  set -e
  if [[ "$resolver_rc" == "0" && -n "$resolver_candidate" ]]; then
    candidates=("$resolver_candidate" "${candidates[@]}")
  fi
fi

declare -A seen_candidates=()
declare -a unique_candidates=()
for c in "${candidates[@]}"; do
  [[ -n "$c" ]] || continue
  [[ -x "$c" ]] || continue
  if [[ -n "${seen_candidates[$c]:-}" ]]; then
    continue
  fi
  seen_candidates["$c"]=1
  unique_candidates+=("$c")
done

jit_runner=""
for c in "${unique_candidates[@]}"; do
  set +e
  timeout 15 "$c" jit "$ARITH_PROBE_FIXTURE" > "$TMP_DIR/probe.out" 2>&1
  rc=$?
  set -e
  if [[ "$rc" == "0" ]] && contains_literal "jit_probe_arith_ok" "$TMP_DIR/probe.out"; then
    jit_runner="$c"
    break
  fi
done

if [[ -z "$jit_runner" ]]; then
  emit_not_run "jit_runner_unavailable"
  exit 0
fi

RESULTS_TSV="$TMP_DIR/results.tsv"
: > "$RESULTS_TSV"

probe_file() {
  local name="$1"
  local fixture="$2"
  local marker="$3"
  local timeout_s="$4"

  local out_file="$TMP_DIR/${name}.out"
  local err_file="$TMP_DIR/${name}.err"
  local time_file="$TMP_DIR/${name}.time"

  set +e
  /usr/bin/time -f '%e' -o "$time_file" \
    timeout "$timeout_s" "$jit_runner" jit "$fixture" > "$out_file" 2> "$err_file"
  local rc=$?
  set -e

  local status="fail"
  local reason="rc_nonzero"
  if contains_literal "JIT backend not enabled" "$out_file" || contains_literal "JIT backend not enabled" "$err_file"; then
    status="not_jit"
    reason="jit_backend_not_enabled"
  elif contains_literal "overflowed its stack" "$out_file" || contains_literal "overflowed its stack" "$err_file"; then
    status="fail"
    reason="stack_overflow"
  elif [[ "$rc" == "124" ]]; then
    status="fail"
    reason="timeout"
  elif contains_literal "panicked at" "$out_file" || contains_literal "panicked at" "$err_file"; then
    status="fail"
    reason="jit_panic"
  elif contains_literal "Error:" "$out_file" || contains_literal "Error:" "$err_file"; then
    status="fail"
    reason="compile_error"
  elif [[ "$rc" == "0" ]] && contains_literal "$marker" "$out_file"; then
    status="pass"
    reason="ok"
  elif [[ "$rc" == "0" ]]; then
    status="fail"
    reason="missing_marker"
  fi

  local elapsed_raw
  elapsed_raw="$(cat "$time_file" 2>/dev/null || true)"
  local elapsed
  elapsed="$(printf '%s' "$elapsed_raw" | tr '\n\t' '  ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$reason" "$rc" "$elapsed" >> "$RESULTS_TSV"
}

probe_file "baseline" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_timeout_baseline.sio" "jit_ir_baseline_ok" "30s"
probe_file "parser_ast_import" "$ROOT_DIR/self-hosted/compiler/jit_probe_parser_ast_import.sio" "jit_parser_ast_ok" "50s"
probe_file "parser_ast_name_import" "$ROOT_DIR/self-hosted/compiler/jit_probe_parser_ast_name_import.sio" "jit_parser_ast_name_ok" "40s"
probe_file "parser_ast_core_import" "$ROOT_DIR/self-hosted/compiler/jit_probe_parser_ast_core_import.sio" "jit_parser_ast_core_ok" "40s"
probe_file "parser_ast_ops_import" "$ROOT_DIR/self-hosted/compiler/jit_probe_parser_ast_ops_import.sio" "jit_parser_ast_ops_ok" "40s"
probe_file "ir_ir_import" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_ir_import.sio" "jit_ir_ir_ok" "70s"
probe_file "ir_ir_touch" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_ir_touch.sio" "jit_ir_ir_touch_ok" "70s"

python3 - "$RESULTS_TSV" "$OUT_JSON" "$jit_runner" <<'PY'
import datetime as dt
import json
import re
import sys

results_path, out_path, runner = sys.argv[1:4]
rows = []

def parse_elapsed(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    m = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None

for raw in open(results_path, "r", encoding="utf-8"):
    raw = raw.rstrip("\n")
    if not raw:
        continue
    name, status, reason, rc, elapsed = raw.split("\t", 4)
    rows.append(
        {
            "probe": name,
            "status": status,
            "reason": reason,
            "rc": int(rc),
            "elapsed_seconds": parse_elapsed(elapsed),
        }
    )

def probe(name: str):
    return next((r for r in rows if r["probe"] == name), {"status": "missing", "reason": "missing"})

baseline = probe("baseline")
parser_ast = probe("parser_ast_import")
parser_ast_name = probe("parser_ast_name_import")
parser_ast_core = probe("parser_ast_core_import")
parser_ast_ops = probe("parser_ast_ops_import")
ir_ir = probe("ir_ir_import")
ir_touch = probe("ir_ir_touch")

parser_probes = [parser_ast, parser_ast_name, parser_ast_core, parser_ast_ops]
parser_timeout = any(p.get("reason") == "timeout" for p in parser_probes)
parser_fail = any(p.get("status") != "pass" for p in parser_probes)
ir_timeout = ir_ir["reason"] == "timeout" or ir_touch["reason"] == "timeout"

timeouts = [r for r in rows if r["reason"] == "timeout"]
stack = [r for r in rows if r["reason"] == "stack_overflow"]
jit_panic = [r for r in rows if r["reason"] == "jit_panic"]
compile_error = [r for r in rows if r["reason"] == "compile_error"]
passes = [r for r in rows if r["status"] == "pass"]

lane_split = "unknown"
if parser_timeout and ir_timeout:
    lane_split = "frontend_and_ir_timeout"
elif (not parser_fail) and ir_timeout:
    lane_split = "ir_specific_timeout"
elif (not parser_fail) and ir_ir["status"] == "pass" and ir_touch["status"] == "pass":
    lane_split = "no_timeout"
elif parser_fail and ir_timeout:
    lane_split = "parser_dependency_issue_with_ir_timeout"
elif parser_fail:
    lane_split = "parser_dependency_issue_only"

status = "pass"
reason = "ok"
if ir_timeout:
    status = "fail"
    reason = "ir_module_timeout_detected"
elif parser_fail:
    status = "fail"
    reason = "parser_dependency_probe_failure"
elif stack:
    status = "fail"
    reason = "stack_overflow_detected"
elif jit_panic:
    status = "fail"
    reason = "jit_panic_detected"
elif compile_error:
    status = "fail"
    reason = "compile_error_detected"
elif timeouts:
    status = "fail"
    reason = "timeout_detected"
elif len(passes) != len(rows):
    status = "fail"
    reason = "probe_failure_detected"

payload = {
    "schema": "sounio.sprint1.jit_ir_timeout_repro.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "runner": runner,
    "lane_split": lane_split,
    "summary": {
        "total_probes": len(rows),
        "pass_count": len(passes),
        "timeout_count": len(timeouts),
        "stack_overflow_count": len(stack),
        "jit_panic_count": len(jit_panic),
        "compile_error_count": len(compile_error),
    },
    "pivot": {
        "baseline": baseline,
        "parser_ast_import": parser_ast,
        "parser_ast_name_import": parser_ast_name,
        "parser_ast_core_import": parser_ast_core,
        "parser_ast_ops_import": parser_ast_ops,
        "ir_ir_import": ir_ir,
        "ir_ir_touch": ir_touch,
    },
    "results": rows,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

print(f"wrote: {out_path}")
print(f"status={status} reason={reason} lane_split={lane_split}")
PY
