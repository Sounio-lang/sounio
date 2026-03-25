#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${1:-$ROOT_DIR/artifacts/sprint1/jit_root_cause_bigpush.v1.json}"
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
    "schema": "sounio.sprint1.jit_root_cause_bigpush.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "not_run",
    "reason": reason,
    "runner": "",
    "summary": {
        "total_probes": 0,
        "pass_count": 0,
        "stack_overflow_count": 0,
        "timeout_count": 0,
        "jit_panic_count": 0,
        "compile_error_count": 0,
    },
    "families": {},
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
  local family="$1"
  local name="$2"
  local fixture="$3"
  local marker="$4"
  local timeout_s="$5"

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
  elif [[ "$rc" == "0" ]]; then
    if [[ -z "$marker" ]] || contains_literal "$marker" "$out_file"; then
      status="pass"
      reason="ok"
    else
      status="fail"
      reason="missing_marker"
    fi
  fi

  local elapsed_raw
  elapsed_raw="$(cat "$time_file" 2>/dev/null || true)"
  local elapsed
  elapsed="$(printf '%s' "$elapsed_raw" | tr '\n\t' '  ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$family" "$name" "$status" "$reason" "$rc" "$elapsed" >> "$RESULTS_TSV"
}

probe_file "wasm" "wasm_real" "$ROOT_DIR/self-hosted/compiler/jit_probe_wasm_mod_import.sio" "jit_wasm_mod_ok" "50s"
probe_file "wasm" "wasm_shadow_stage0" "$ROOT_DIR/self-hosted/compiler/jit_probe_wasm_shadow_stage0_import.sio" "" "30s"
probe_file "wasm" "wasm_shadow_stage1" "$ROOT_DIR/self-hosted/compiler/jit_probe_wasm_shadow_stage1_import.sio" "" "30s"
probe_file "wasm" "wasm_shadow_stage2_len" "$ROOT_DIR/self-hosted/compiler/jit_probe_wasm_shadow_stage2_len_import.sio" "" "35s"
probe_file "wasm" "wasm_shadow_stage2_emit" "$ROOT_DIR/self-hosted/compiler/jit_probe_wasm_shadow_stage2_emit_import.sio" "" "45s"

probe_file "ir" "ir_real_import" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_ir_import.sio" "jit_ir_ir_ok" "80s"
probe_file "ir" "ir_real_touch" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_ir_touch.sio" "jit_ir_ir_touch_ok" "80s"
probe_file "ir" "ir_shadow_stage0" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_shadow_stage0_import.sio" "" "30s"
probe_file "ir" "ir_shadow_stage1" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_shadow_stage1_import.sio" "" "30s"
probe_file "ir" "ir_shadow_stage2_struct" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_shadow_stage2_struct_import.sio" "" "40s"
probe_file "ir" "ir_shadow_stage2_array" "$ROOT_DIR/self-hosted/compiler/jit_probe_ir_shadow_stage2_array_import.sio" "" "40s"

python3 - "$RESULTS_TSV" "$OUT_JSON" "$jit_runner" <<'PY'
import datetime as dt
import json
import re
import sys

results_path, out_path, runner = sys.argv[1:4]

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

rows = []
for raw in open(results_path, "r", encoding="utf-8"):
    raw = raw.rstrip("\n")
    if not raw:
        continue
    family, name, status, reason, rc, elapsed = raw.split("\t", 5)
    rows.append(
        {
            "family": family,
            "probe": name,
            "status": status,
            "reason": reason,
            "rc": int(rc),
            "elapsed_seconds": parse_elapsed(elapsed),
        }
    )

summary = {
    "total_probes": len(rows),
    "pass_count": sum(1 for r in rows if r["status"] == "pass"),
    "stack_overflow_count": sum(1 for r in rows if r["reason"] == "stack_overflow"),
    "timeout_count": sum(1 for r in rows if r["reason"] == "timeout"),
    "jit_panic_count": sum(1 for r in rows if r["reason"] == "jit_panic"),
    "compile_error_count": sum(1 for r in rows if r["reason"] == "compile_error"),
}

status = "pass"
reason = "ok"
if summary["stack_overflow_count"] > 0:
    status = "fail"
    reason = "stack_overflow_detected"
elif summary["timeout_count"] > 0:
    status = "fail"
    reason = "timeout_detected"
elif summary["jit_panic_count"] > 0:
    status = "fail"
    reason = "jit_panic_detected"
elif summary["compile_error_count"] > 0:
    status = "fail"
    reason = "compile_error_detected"
elif summary["pass_count"] != summary["total_probes"]:
    status = "fail"
    reason = "probe_failure_detected"

by_probe = {r["probe"]: r for r in rows}
family_order = {
    "wasm": ["wasm_real", "wasm_shadow_stage0", "wasm_shadow_stage1", "wasm_shadow_stage2_len", "wasm_shadow_stage2_emit"],
    "ir": ["ir_real_import", "ir_real_touch", "ir_shadow_stage0", "ir_shadow_stage1", "ir_shadow_stage2_struct", "ir_shadow_stage2_array"],
}
family_real_count = {
    "wasm": 1,
    "ir": 2,
}

families = {}
for fam, order in family_order.items():
    fam_rows = [by_probe[p] for p in order if p in by_probe]
    real_count = family_real_count.get(fam, 1)
    real_order = order[:real_count]
    shadow_order = order[real_count:]
    first_failure = ""
    first_shadow_failure = ""
    shadow_failure_count = 0
    first_real_failure = ""
    for p in order:
        row = by_probe.get(p)
        if row and row["status"] != "pass":
            first_failure = p
            break
    for p in real_order:
        row = by_probe.get(p)
        if row and row["status"] != "pass":
            first_real_failure = p
            break
    for p in shadow_order:
        row = by_probe.get(p)
        if row and row["status"] != "pass":
            shadow_failure_count += 1
            if not first_shadow_failure:
                first_shadow_failure = p
    real_probe_status = "pass"
    real_probe_reason = "ok"
    if first_real_failure:
        real_row = by_probe.get(first_real_failure, {})
        real_probe_status = real_row.get("status", "fail")
        real_probe_reason = real_row.get("reason", "unknown")
    families[fam] = {
        "order": order,
        "real_order": real_order,
        "shadow_order": shadow_order,
        "first_failure_probe": first_failure,
        "first_real_failure_probe": first_real_failure,
        "first_shadow_failure_probe": first_shadow_failure,
        "real_probe_status": real_probe_status,
        "real_probe_reason": real_probe_reason,
        "shadow_pass_count": sum(1 for p in shadow_order if by_probe.get(p, {}).get("status") == "pass"),
        "shadow_count": len(shadow_order),
        "shadow_failure_count": shadow_failure_count,
        "rows": fam_rows,
    }

hypotheses = []
wasm = families["wasm"]
ir = families["ir"]
if wasm["real_probe_status"] != "pass" and wasm["shadow_pass_count"] == wasm["shadow_count"]:
    hypotheses.append("wasm_real_only_failure")
elif wasm["shadow_failure_count"] > 0:
    hypotheses.append("wasm_shadow_failure_reproduced")
    if wasm["shadow_failure_count"] < wasm["shadow_count"]:
        hypotheses.append("wasm_shadow_partial_failure")
elif wasm["first_failure_probe"].startswith("wasm_shadow_"):
    hypotheses.append("wasm_shadow_failure_reproduced")

if ir["real_probe_status"] != "pass" and ir["shadow_pass_count"] == ir["shadow_count"]:
    hypotheses.append("ir_real_only_failure")
elif ir["shadow_failure_count"] > 0:
    hypotheses.append("ir_shadow_failure_reproduced")
    if ir["shadow_failure_count"] < ir["shadow_count"]:
        hypotheses.append("ir_shadow_partial_failure")
elif ir["first_failure_probe"].startswith("ir_shadow_"):
    hypotheses.append("ir_shadow_failure_reproduced")

payload = {
    "schema": "sounio.sprint1.jit_root_cause_bigpush.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "runner": runner,
    "summary": summary,
    "families": families,
    "hypotheses": hypotheses,
    "results": rows,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

print(f"wrote: {out_path}")
print(f"status={status} reason={reason}")
print(f"hypotheses={','.join(hypotheses) if hypotheses else 'none'}")
PY
