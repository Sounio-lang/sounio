#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="${SOUNIO_DIRECT_DRIVER_ARTIFACT_ROOT:-/tmp/sounio-direct-driver-support-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$ARTIFACT_ROOT/logs"
BIN_DIR="$ARTIFACT_ROOT/bin"
MATRIX_TSV="$ARTIFACT_ROOT/direct_driver_matrix.tsv"
SUMMARY_JSON="$ARTIFACT_ROOT/direct_driver_support_status.v1.json"

mkdir -p "$LOG_DIR" "$BIN_DIR"

echo "== Sounio Direct Driver Support Gate =="
echo "repo=$ROOT_DIR"
echo "head=$(git rev-parse HEAD 2>/dev/null || true)"
echo "artifacts=$ARTIFACT_ROOT"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
if [[ ! -x "$SOUC_BIN" ]]; then
  echo "error: missing compiler entrypoint: $SOUC_BIN" >&2
  exit 1
fi

CASES=(
  "expr_add_7_35:42:"
  "expr_div_84_2:42:"
  "expr_mod_85_43:42:"
  "expr_mul_6_7:42:"
  "expr_sub_0_42:214:"
  "expr_sub_50_8:42:"
  "if_eq_2_2_42_0:42:"
  "if_ge_2_2_42_0:42:"
  "if_lt_1_2_42_0:42:"
  "if_ne_1_2_42_0:42:"
  "if_true_41_1:41:"
  "let_answer_10_plus_32:42:"
  "let_first_second_plus_32:42:"
  "let_three_bindings_chain:42:"
  "let_x_10_plus_32:42:"
  "let_x_42:42:"
  "let_x_plus_2:42:"
  "let_xy_add:42:"
  "let_xy_y_is_x_plus_32:42:"
  "print_boot:0:boot"
  "println_hi:0:hi"
  "ret_42:42:"
  "return_x_plus_2:42:"
  "two_prints:0:ab"
)

printf 'case_id\tsource\tcompile_status\trun_status\texit_code\texpected_exit\tstdout\texpected_stdout\n' >"$MATRIX_TSV"

pass=0
fail=0

for row in "${CASES[@]}"; do
  IFS=: read -r case_id expected_exit expected_stdout <<<"$row"
  src="tests/selfhost-driver-output/${case_id}.sio"
  out_bin="$BIN_DIR/${case_id}.elf"
  compile_log="$LOG_DIR/${case_id}.compile.log"
  stdout_file="$LOG_DIR/${case_id}.stdout"
  stderr_file="$LOG_DIR/${case_id}.stderr"
  compile_status="fail"
  run_status="fail"
  exit_code="NA"
  stdout_value=""

  if [[ ! -f "$src" ]]; then
    printf '%s\t%s\tmissing\tfail\tNA\t%s\t%s\t%s\n' \
      "$case_id" "$src" "$expected_exit" "$stdout_value" "$expected_stdout" >>"$MATRIX_TSV"
    echo "[direct-driver] FAIL missing fixture: $src" >&2
    fail=$((fail + 1))
    continue
  fi

  set +e
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    "$SOUC_BIN" compile "$src" -o "$out_bin" >"$compile_log" 2>&1
  compile_rc=$?
  set -e

  if [[ "$compile_rc" -eq 0 && -x "$out_bin" ]]; then
    compile_status="pass"
    set +e
    "$out_bin" >"$stdout_file" 2>"$stderr_file"
    exit_code=$?
    set -e
    stdout_value="$(cat "$stdout_file")"
    if [[ "$exit_code" == "$expected_exit" && "$stdout_value" == "$expected_stdout" ]]; then
      run_status="pass"
      pass=$((pass + 1))
      echo "[direct-driver] PASS $case_id exit=$exit_code"
    else
      fail=$((fail + 1))
      echo "[direct-driver] FAIL $case_id exit=$exit_code expected=$expected_exit stdout=[$stdout_value] expected_stdout=[$expected_stdout]" >&2
    fi
  else
    fail=$((fail + 1))
    echo "[direct-driver] FAIL $case_id compile_rc=$compile_rc log=$compile_log" >&2
    tail -n 40 "$compile_log" >&2 || true
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$src" "$compile_status" "$run_status" "$exit_code" "$expected_exit" "$stdout_value" "$expected_stdout" \
    >>"$MATRIX_TSV"
done

python3 - "$ROOT_DIR" "$MATRIX_TSV" "$SUMMARY_JSON" "$pass" "$fail" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
matrix_tsv = Path(sys.argv[2]).resolve()
summary_json = Path(sys.argv[3]).resolve()
pass_count = int(sys.argv[4])
fail_count = int(sys.argv[5])

registry_path = root / "docs/serious-language/public-claim-registry.v1.tsv"
known_limitations_path = root / "docs/compiler/KNOWN_LIMITATIONS.md"
readiness_ledger_path = root / "docs/serious-language/readiness-ledger.md"

gate_ref = "scripts/ci/sounio_direct_driver_support_gate.sh"
failures: list[dict[str, str]] = []

rows = list(csv.DictReader(matrix_tsv.open(newline="", encoding="utf-8"), delimiter="\t"))
if len(rows) != 24:
    failures.append({"kind": "matrix", "message": f"expected 24 direct-driver fixtures, got {len(rows)}"})
if fail_count:
    failures.append({"kind": "matrix", "message": f"direct-driver compile/run failures={fail_count}"})

registry_rows: dict[str, dict[str, str]] = {}
with registry_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        registry_rows[row["claim_id"]] = {key: (value or "").strip() for key, value in row.items()}

direct_row = registry_rows.get("direct_driver")
if not direct_row:
    failures.append({"kind": "claim_registry", "message": "missing broad direct_driver row"})
else:
    if direct_row.get("claim_level") != "prototype" or direct_row.get("closure_status") != "downgraded":
        failures.append({"kind": "claim_registry", "message": "broad direct_driver row must remain prototype/downgraded"})
    if "large-surface" not in direct_row.get("public_wording", ""):
        failures.append({"kind": "claim_registry", "message": "broad direct_driver wording must preserve large-surface boundary"})

support_row = registry_rows.get("direct_driver.support")
if not support_row:
    failures.append({"kind": "claim_registry", "message": "missing direct_driver.support row"})
else:
    if support_row.get("claim_level") != "validated_research":
        failures.append({"kind": "claim_registry", "message": f"direct_driver.support claim_level={support_row.get('claim_level')}"})
    if support_row.get("closure_status") != "closed":
        failures.append({"kind": "claim_registry", "message": f"direct_driver.support closure_status={support_row.get('closure_status')}"})
    if support_row.get("evidence_ref") != gate_ref:
        failures.append({"kind": "claim_registry", "message": f"direct_driver.support evidence_ref={support_row.get('evidence_ref')}"})
    wording = support_row.get("public_wording", "")
    for token in ["bounded direct-driver", "not large-surface", "not semantic authority"]:
        if token not in wording:
            failures.append({"kind": "claim_registry", "message": f"direct_driver.support wording missing token: {token}"})

known = known_limitations_path.read_text(encoding="utf-8", errors="replace")
ledger = readiness_ledger_path.read_text(encoding="utf-8", errors="replace")
for path, text, tokens in [
    (known_limitations_path, known, ["direct_driver.support = validated_research", "direct_driver = prototype", "24/24", "NOT PROVED"]),
    (readiness_ledger_path, ledger, ["Direct-driver support cohort", "24/24", "not large-surface"]),
]:
    for token in tokens:
        if token not in text:
            failures.append({"kind": "docs", "message": f"{path.relative_to(root)} missing token: {token}"})

not_proved = [
    "large-surface direct-driver execution",
    "ontology-sized direct-driver semantic truth",
    "replacement of wrapper provenance or fallback compile authority",
    "native-v2 driver self-compile/fixed-point closure",
    "direct-driver negative-truth restoration for ontology compile-fail fixtures",
    "API stability or broad production readiness outside this 24-fixture cohort",
]

summary = {
    "schema": "sounio.direct_driver.support_status.v1",
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "status_summary": "pass" if not failures else "fail",
    "claim": "direct_driver.support=validated_research",
    "claim_shape": "bounded direct-driver compile/run support cohort; not large-surface semantic authority",
    "matrix_tsv": str(matrix_tsv),
    "totals": {
        "pass": pass_count,
        "fail": fail_count,
        "total": len(rows),
    },
    "proved": {
        "compile_to_elf_and_execute": [row["case_id"] for row in rows if row["run_status"] == "pass"],
        "fixture_root": "tests/selfhost-driver-output",
    },
    "not_proved": not_proved,
    "failures": failures,
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"status_summary={summary['status_summary']}")
print(f"pass={pass_count}")
print(f"fail={fail_count}")
print(f"summary_json={summary_json}")
if failures:
    for failure in failures:
        print(f"failure {failure['kind']}: {failure['message']}")
    raise SystemExit(1)
PY

echo "[direct-driver] PASS: bounded direct-driver support cohort is checked"
