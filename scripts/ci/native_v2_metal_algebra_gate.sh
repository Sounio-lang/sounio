#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

OUT_DIR="${SOUNIO_NATIVE_V2_METAL_ALGEBRA_DIR:-$(mktemp -d /tmp/sounio-native-v2-metal-algebra.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_metal_algebra.v1.json"
MANIFEST="${SOUNIO_NATIVE_V2_METAL_ALGEBRA_MANIFEST:-tests/gpu/metal_first/manifest.tsv}"
ORACLE_OSSM="tests/run-pass/gpu_epistemic_f64_ossm_parity.sio"
ORACLE_SED="tests/run-pass/gpu_epistemic_f64_sedenion_parity.sio"
LOCAL_METAL_RUNNER="scripts/gpu-metal-validation/run_algebra_metal.sh"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

append_header() {
  cat >"$RESULTS_TSV" <<'EOF'
case_id	lane	path	status	detail
EOF
}

append_row() {
  local case_id="$1"
  local lane="$2"
  local path="$3"
  local status="$4"
  local detail="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "$lane" "$path" "$status" "$detail" >>"$RESULTS_TSV"
}

echo "[native-v2-metal-algebra] out=$OUT_DIR"
sounio_require_souc

append_header
pass_count=0
fail_count=0
not_run_count=0

run_oracle() {
  local case_id="$1"
  local program="$2"
  local expected="$3"
  local log="$LOG_DIR/$case_id.oracle.log"
  if [[ ! -f "$program" ]]; then
    append_row "$case_id" "cpu_oracle" "$program" "fail" "missing_oracle"
    fail_count=$((fail_count + 1))
    return 0
  fi
  if "$SOUC_BIN" run "$program" >"$log" 2>&1 && grep -qF "$expected" "$log"; then
    append_row "$case_id" "cpu_oracle" "$program" "pass" "oracle_stdout_ok"
    pass_count=$((pass_count + 1))
  else
    append_row "$case_id" "cpu_oracle" "$program" "fail" "oracle_run_failed"
    fail_count=$((fail_count + 1))
  fi
}

run_oracle "ossm_cpu_f64_oracle" "$ORACLE_OSSM" "PASS: GPU epistemic f64 O-SSM parity"
run_oracle "sedenion_cpu_f64_oracle" "$ORACLE_SED" "PASS: GPU epistemic f64 S-SSM parity"

# Pure-bash structural MSL walker (replaces python3 heredoc).
# Per-row checks: kernel_name, metal_f64_policy, source_semantics=f64,
# thread_position_in_grid, device float* present, no plain "double" or
# "device double*" anywhere. Append results.tsv rows with same fields
# (case_id, lane, path, status, detail) and same ordering as python.
set +e
structural_rc=0
if [[ ! -f "$MANIFEST" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\n' "metal_manifest" "structural" "$MANIFEST" "fail" "missing_manifest" >> "$RESULTS_TSV"
  structural_rc=1
else
  # Manifest row floor: the structural loop appends exactly one results row
  # per non-comment data row. An empty or header-only manifest runs the loop
  # zero times, every count stays 0, and status computes "pass" -- an
  # instrument that answered nothing certifying itself.
  manifest_rows="$(tail -n +2 "$MANIFEST" | grep -vE '^[[:space:]]*(#|$)' | grep -c . || true)"
  if [[ "$manifest_rows" -lt 1 ]]; then
    echo "[native-v2-metal-algebra] FAIL: manifest $MANIFEST has no data rows -- nothing to verify" >&2
    exit 1
  fi
  fail=0
  # Skip header, read each row positionally (manifest schema:
  # case_id, kernel_path, kernel_name, policy).
  while IFS=$'\t' read -r case_id kernel_path kernel_name policy _rest; do
    [[ -z "$case_id" ]] && continue
    [[ "$case_id" == \#* ]] && continue
    if [[ ! -f "$kernel_path" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "structural_msl" "$kernel_path" "fail" "missing_msl" >> "$RESULTS_TSV"
      fail=$((fail + 1))
      continue
    fi
    bad=()
    grep -qF "kernel void $kernel_name" "$kernel_path" || bad+=("kernel_name")
    grep -qF "sounio.metal_f64_policy=$policy" "$kernel_path" || bad+=("policy")
    grep -qF "sounio.source_semantics=f64" "$kernel_path" || bad+=("source_f64")
    grep -qF "thread_position_in_grid" "$kernel_path" || bad+=("thread_id")
    grep -qF "device float*" "$kernel_path" || bad+=("device_float")
    grep -qF "double" "$kernel_path" && bad+=("no_double")
    grep -qF "device double*" "$kernel_path" && bad+=("no_device_double")
    if [[ ${#bad[@]} -gt 0 ]]; then
      detail="missing_$(IFS=,; echo "${bad[*]}")"
      printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "structural_msl" "$kernel_path" "fail" "$detail" >> "$RESULTS_TSV"
      fail=$((fail + 1))
    else
      printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "structural_msl" "$kernel_path" "pass" "metal_f64_policy_structural_ok" >> "$RESULTS_TSV"
    fi
  done < <(tail -n +2 "$MANIFEST")
  # Every data row must have answered exactly once (the loop appends one
  # results row per case, pass or fail); a shortfall means rows were lost
  # between the manifest and the results, not that the corpus was smaller.
  structural_rows="$(awk -F'\t' '$2=="structural_msl"' "$RESULTS_TSV" | wc -l | tr -d ' ')"
  if [[ "$structural_rows" -ne "$manifest_rows" ]]; then
    echo "[native-v2-metal-algebra] FAIL: $structural_rows of $manifest_rows manifest rows produced results -- incomplete structural pass" >&2
    exit 1
  fi
  [[ "$fail" -gt 0 ]] && structural_rc=1
fi
set -e
if [[ "$structural_rc" -eq 0 ]]; then
  structural_passes="$(tail -n +2 "$RESULTS_TSV" | awk -F'\t' '$2=="structural_msl" && $4=="pass"{c++} END{print c+0}')"
  pass_count=$((pass_count + structural_passes))
else
  structural_fails="$(tail -n +2 "$RESULTS_TSV" | awk -F'\t' '$2=="structural_msl" && $4=="fail"{c++} END{print c+0}')"
  fail_count=$((fail_count + structural_fails))
fi

runtime_detail=""
if [[ "$fail_count" -eq 0 ]]; then
  runtime_log="$LOG_DIR/apple_metal_runtime.log"
  if command -v xcrun >/dev/null 2>&1 && command -v swiftc >/dev/null 2>&1; then
    if bash "$LOCAL_METAL_RUNNER" >"$runtime_log" 2>&1; then
      append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "pass" "local_metal_runtime_ok"
      pass_count=$((pass_count + 1))
    else
      append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "fail" "local_metal_runtime_failed"
      fail_count=$((fail_count + 1))
    fi
  else
    runtime_detail="xcrun_or_swiftc_unavailable"
    append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "not_run" "$runtime_detail"
    not_run_count=$((not_run_count + 1))
  fi
else
  runtime_detail="skipped_after_structural_or_oracle_failure"
  append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "not_run" "$runtime_detail"
  not_run_count=$((not_run_count + 1))
fi

# Pure-bash summary JSON emitter (replaces python3 csv.DictReader + json.dump heredoc).
# Status: "fail" if fail_count>0, "partial" if not_run_count>0 else "pass".
# cases: TSV rows -> JSON array of objects (keys sorted alphabetically per row).
# remaining_boundaries: 3 always-present + 1 conditional (only if not_run_count>0).
if [[ "$fail_count" -gt 0 ]]; then
  status_str="fail"
elif [[ "$not_run_count" -gt 0 ]]; then
  status_str="partial"
else
  status_str="pass"
fi

if [[ -f "$MANIFEST" ]]; then
  MANIFEST_SHA256="$(sha256sum "$MANIFEST" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$MANIFEST" | awk '{print $1}')"
else
  MANIFEST_SHA256=""
fi

# Convert results.tsv to JSON array of objects with keys sorted alphabetically
# (case_id, detail, lane, path, status — matches python json.dumps sort_keys=True).
# awk emits one '{ ... }' object per row, comma-separated, wrapped in [].
CASES_JSON="$(awk -F'\t' '
function jsesc(s,    out, i, c) {
  out = ""
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (c == "\\") out = out "\\\\"
    else if (c == "\"") out = out "\\\""
    else if (c == "\n") out = out "\\n"
    else if (c == "\r") out = out "\\r"
    else if (c == "\t") out = out "\\t"
    else out = out c
  }
  return out
}
NR==1 { next }   # skip header
NR>1 && NF>=5 {
  if (printed) printf ",\n"
  printed = 1
  printf "    {\n"
  printf "      \"case_id\": \"%s\",\n", jsesc($1)
  printf "      \"detail\": \"%s\",\n", jsesc($5)
  printf "      \"lane\": \"%s\",\n",    jsesc($2)
  printf "      \"path\": \"%s\",\n",    jsesc($3)
  printf "      \"status\": \"%s\"\n",   jsesc($4)
  printf "    }"
}
END { if (printed) printf "\n" }
' "$RESULTS_TSV")"
CASES_JSON="[
$CASES_JSON
  ]"

# remaining_boundaries: filter empty per python `[x for x in ... if x]`.
boundaries_args=()
if [[ "$not_run_count" -gt 0 ]]; then
  boundaries_args+=("Apple Metal runtime requires xcrun, swiftc, and an Apple GPU host")
fi
boundaries_args+=("Metal f64 is represented by explicit Sounio lowering policy, not native MSL double")
boundaries_args+=("float2 compensated f64 lowering is named but not yet the promoted runtime policy")
boundaries_args+=("this gate does not prove tensor-core performance, ROCm, WebGPU, CUDA PTX parity, or DDC")
boundaries_joined="$(IFS='|'; echo "${boundaries_args[*]}")"

"$ROOT_DIR/bin/kretikos" json-emit \
  --string "artifact_dir=$OUT_DIR" \
  --raw-json "cases=$CASES_JSON" \
  --int    "fail_count=$fail_count" \
  --string "fallback_path=none" \
  --string "host_callback=none" \
  --string "manifest=$MANIFEST" \
  --string "manifest_sha256=$MANIFEST_SHA256" \
  --int    "not_run_count=$not_run_count" \
  --int    "pass_count=$pass_count" \
  --array-strings "remaining_boundaries=$boundaries_joined" \
  --string "results_tsv=$RESULTS_TSV" \
  --string "schema=sounio.native_v2_metal_algebra.v1" \
  --string "souc_bin=$SOUC_BIN" \
  --string "status=$status_str" \
  --string "target=apple-metal-algebra" \
  > "$SUMMARY_JSON"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi

summary_status="$("$ROOT_DIR/bin/kretikos" kaxi-validate-evidence "$SUMMARY_JSON" --print "status")"

echo "[native-v2-metal-algebra] status=$summary_status summary=$SUMMARY_JSON"
if [[ "$summary_status" == "fail" ]]; then
  exit 1
fi
