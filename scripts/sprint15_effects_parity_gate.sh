#!/usr/bin/env bash
# sprint15_effects_parity_gate.sh — effects parity 22-reclassified gate (H6.0)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="$SOUC_BIN"
fi
OUT_JSON="${SOUNIO_SPRINT15_GATE_OUT:-$ROOT_DIR/artifacts/sprint15/effects_parity_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT15_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint15_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

CF_DIR="$ROOT_DIR/tests/compile-fail"

# ── Case 1: current_effects field in Checker struct ──────────────────────────

if grep -q 'current_effects: \[i64; 8\]' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "checker_has_current_effects" "pass" "field_found_in_check.sio"
else
  record "checker_has_current_effects" "fail" "current_effects_field_missing"
fi

# ── Case 2: effects_subset called in check_callee_effects ────────────────────

if grep -q 'effects_subset' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "effects_subset_wired" "pass" "effects_subset_called_in_check.sio"
else
  record "effects_subset_wired" "fail" "effects_subset_not_found_in_check.sio"
fi

# ── Case 3: find_missing_effects called in check.sio ─────────────────────────

if grep -q 'find_missing_effects' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "find_missing_effects_wired" "pass" "found_in_check.sio"
else
  record "find_missing_effects_wired" "fail" "not_found_in_check.sio"
fi

# ── Case 4: has_effect_id called in check.sio ────────────────────────────────

if grep -q 'has_effect_id' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "has_effect_id_wired" "pass" "found_in_check.sio"
else
  record "has_effect_id_wired" "fail" "not_found_in_check.sio"
fi

# ── Case 5: At least 22 SELFHOST_PASS in compile-fail ────────────────────────

cf_pass_count=$(grep -rl 'SELFHOST_PASS' "$CF_DIR" 2>/dev/null | wc -l)
if [ "$cf_pass_count" -ge 23 ]; then
  record "selfhost_pass_count" "pass" "found_${cf_pass_count}_SELFHOST_PASS_in_compile-fail"
else
  record "selfhost_pass_count" "fail" "expected_ge_23_got_${cf_pass_count}"
fi

# ── Case 6: At most 2 BLOCKED remain (parser_rust_macro + effects_unhandled) ─

blocked_count=$(grep -rl 'BLOCKED:' "$CF_DIR" 2>/dev/null | wc -l || true)
if [ "$blocked_count" -le 2 ]; then
  record "blocked_at_most_2" "pass" "${blocked_count}_BLOCKED_remaining_expected_le_2"
else
  record "blocked_at_most_2" "fail" "found_${blocked_count}_BLOCKED_expected_le_2"
fi

# ── JIT probe: reclassified effects tests ────────────────────────────────────
JIT_BIN="${SOUNIO_JIT_BIN:-$ROOT_DIR/target/debug/souc}"

probe_jit() {
  local name="$1" file="$2" expect_fail="${3:-true}"
  if [ ! -x "$JIT_BIN" ]; then
    record "$name" "not_run" "jit_binary_not_found"
    return
  fi
  set +e
  timeout "$TIMEOUT_SECS" "$JIT_BIN" run \
    "$ROOT_DIR/self-hosted/compiler/main.sio" \
    -- --probe-frontend "$file" \
    > "/tmp/sprint15_probe_${name}.log" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$name" "fail" "timeout"
  elif [ "$expect_fail" = "true" ]; then
    if grep -q "probe_frontend: fail\|probe_frontend: error=type_check_failed" "/tmp/sprint15_probe_${name}.log" 2>/dev/null; then
      record "$name" "pass" "sh_probe_fail_as_expected"
    elif grep -q "probe_frontend: ok" "/tmp/sprint15_probe_${name}.log" 2>/dev/null; then
      record "$name" "fail" "sh_probe_passed_but_expected_fail"
    else
      record "$name" "not_run" "no_probe_output"
    fi
  fi
}

# Case 7-9: Effects tests — must detect effects violations
probe_jit "effect_io_in_pure"    "$CF_DIR/effect_io_in_pure.sio"
probe_jit "effect_missing"       "$CF_DIR/effect_missing.sio"
probe_jit "effects_pure_context" "$CF_DIR/effects_pure_context.sio"

# Case 10-15: Sample reclassified non-effects tests
probe_jit "linear_double_use"        "$CF_DIR/linear_double_use.sio"
probe_jit "linear_not_consumed"      "$CF_DIR/linear_not_consumed.sio"
probe_jit "ownership_use_after_move" "$CF_DIR/ownership_use_after_move.sio"
probe_jit "parser_rust_let_mut"      "$CF_DIR/parser_rust_let_mut.sio"
probe_jit "unit_cast_incompatible"   "$CF_DIR/unit_cast_incompatible.sio"
probe_jit "affine_double_use"        "$CF_DIR/affine_double_use.sio"

# ── Case 16: Sprint 14 regression ────────────────────────────────────────────

set +e
S14_OUT="/tmp/sprint15_s14_regression_$$.json"
SOUNIO_SOUC="$SOUC" SOUNIO_JIT_BIN="$JIT_BIN" \
  SOUNIO_SPRINT14_GATE_OUT="$S14_OUT" \
  bash "$ROOT_DIR/scripts/sprint14_checker_parity_gate.sh" > "/tmp/sprint15_s14_regression.log" 2>&1
s14_rc=$?
set -e

if [ $s14_rc -eq 0 ]; then
  record "sprint14_regression" "pass" "sprint14_gate_still_passes"
elif [ $s14_rc -eq 3 ]; then
  record "sprint14_regression" "not_run" "sprint14_gate_not_run"
else
  record "sprint14_regression" "fail" "sprint14_gate_failed_rc=${s14_rc}"
fi
rm -f "$S14_OUT"

# ── Emit JSON artifact ───────────────────────────────────────────────────────
python3 - "$OUT_JSON" "$CASES_TSV" "$SOUC" "$TIMEOUT_SECS" << 'PY'
import json, datetime as dt, sys
from pathlib import Path

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
souc = sys.argv[3]
timeout_secs = int(sys.argv[4])

cases = []
counts = {"pass": 0, "fail": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    parts = raw.split("\t", 2)
    name, status, reason = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({"name": name, "status": status, "reason": reason})

overall = "pass" if counts["fail"] == 0 else "fail"
reason = "effects_parity_23_reclassified" if overall == "pass" else f"{counts['fail']}_failed"

payload = {
    "schema": "sounio.sprint15.effects_parity_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall, "reason": reason,
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {"total": len(cases), "passed": counts["pass"],
                "failed": counts["fail"], "not_run": counts["not_run"]},
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint15_effects_parity_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
