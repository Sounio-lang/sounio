#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${1:-examples/kretikos/manifest.tsv}"
OUT_DIR="${SOUNIO_KRETIKOS_HPC_GATE_DIR:-$(mktemp -d /tmp/kretikos-hpc-slurm.XXXXXX)}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-420}"

mkdir -p "$OUT_DIR"

DOCTOR_LOG="$OUT_DIR/kretikos_hpc_doctor.out"
MANIFEST_LOG="$OUT_DIR/kretikos_hpc_manifest.out"
SUMMARY_COPY="$OUT_DIR/kretikos_hpc_manifest_summary.tsv"
GATE_JSON="$OUT_DIR/kretikos_hpc_slurm_runtime_gate.v1.json"

echo "kretikos_hpc_slurm_runtime_gate: doctor"
./bin/kretikos hpc doctor >"$DOCTOR_LOG" 2>&1
cat "$DOCTOR_LOG"

echo "kretikos_hpc_slurm_runtime_gate: submit manifest=$MANIFEST"
set +e
WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT_SECONDS" \
KRETIKOS_HPC_PUBLISH_RESULTS=1 \
KRETIKOS_HPC_FETCH_RESULTS=1 \
  ./bin/kretikos hpc manifest "$MANIFEST" >"$MANIFEST_LOG" 2>&1
manifest_rc=$?
set -e
cat "$MANIFEST_LOG"

if [[ "$manifest_rc" -ne 0 ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL manifest_rc=$manifest_rc log=$MANIFEST_LOG" >&2
  exit "$manifest_rc"
fi

result_line="$(awk '/^kretikos_manifest_result / { line=$0 } END { print line }' "$MANIFEST_LOG")"
if [[ -z "$result_line" ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL missing kretikos_manifest_result" >&2
  exit 1
fi

total="$(sed -n 's/.*total=\([0-9][0-9]*\).*/\1/p' <<<"$result_line")"
failed="$(sed -n 's/.*failed=\([0-9][0-9]*\).*/\1/p' <<<"$result_line")"
summary_path="$(sed -n 's/.*summary=\([^ ]*\).*/\1/p' <<<"$result_line")"

if [[ -z "$total" || -z "$failed" || -z "$summary_path" ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL malformed result line: $result_line" >&2
  exit 1
fi

if [[ "$failed" -ne 0 ]]; then
  echo "kretikos_hpc_slurm_runtime_gate: FAIL total=$total failed=$failed" >&2
  exit 1
fi

if [[ -f "$summary_path" ]]; then
  cp "$summary_path" "$SUMMARY_COPY"
else
  printf 'label\tsource\tstatus\tjob_id\tcomment\tartifact_dir\tartifact_json\n' >"$SUMMARY_COPY"
fi

# Per-case TSV aggregator → gate JSON via kretikos json-emit + jq -c.
# Replaces the prior python3 heredoc (csv.DictReader + json.dump).
KRETIKOS="$ROOT_DIR/bin/kretikos"
KAXI_SUPPORTED_REGEX='^(vec_add_f32|vec_sub_f32|vec_mul_f32|vec_div_f32|fma_f32|epistemic_elementwise_f32|epistemic_dual_output_f32)$'
emit_jsonl() { "$KRETIKOS" json-emit "$@" | jq -c .; }

WORK="$(mktemp -d /tmp/kretikos-hpc-gate-stage-b.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
for f in art_rt kaxi_certs missing_art missing_cert bad_cert jobs art_dirs art_jsons cases bad_arts; do
  : > "$WORK/$f.jsonl"
done

dedup_prefix() {
  local prefix="$1"
  awk -F'\t' -v p="${prefix}=" '
    NR==1 { for (i=1;i<=NF;i++) if ($i=="comment") col=i; next }
    {
      n = split($col, w, /[ \t]+/)
      for (i=1;i<=n;i++) if (substr(w[i],1,length(p))==p) print substr(w[i],length(p)+1)
    }
  ' "$SUMMARY_COPY" | LC_ALL=C sort -u
}
build_str_array() {
  awk 'BEGIN{printf "["} {if(NR>1) printf ","; gsub(/\\/,"\\\\"); gsub(/"/,"\\\""); printf "\"%s\"", $0} END{printf "]"}'
}
DEVICES_JSON=$(dedup_prefix device | build_str_array)
DRIVERS_JSON=$(dedup_prefix driver | build_str_array)
CCS_JSON=$(dedup_prefix cc | build_str_array)

HEADER=$(head -n1 "$SUMMARY_COPY")
IFS=$'\t' read -r -a HCOLS <<< "$HEADER"

while IFS=$'\t' read -r -a row; do
  declare -A C=()
  for i in "${!HCOLS[@]}"; do C["${HCOLS[$i]}"]="${row[$i]:-}"; done
  label="${C[label]:-}"
  artifact_json="${C[artifact_json]:-}"
  artifact_dir="${C[artifact_dir]:-}"
  job_id="${C[job_id]:-}"

  __case_args=()
  for col in "${HCOLS[@]}"; do __case_args+=(--string "${col}=${C[$col]:-}"); done
  emit_jsonl "${__case_args[@]}" >> "$WORK/cases.jsonl"

  [[ -n "$job_id" ]] && printf '"%s"\n' "$job_id" >> "$WORK/jobs.jsonl"
  [[ -n "$artifact_dir" ]] && printf '"%s"\n' "$artifact_dir" >> "$WORK/art_dirs.jsonl"
  [[ -n "$artifact_json" ]] && printf '"%s"\n' "$artifact_json" >> "$WORK/art_jsons.jsonl"

  if [[ -z "$artifact_json" ]]; then
    emit_jsonl --string "label=$label" --string reason=artifact_json_missing >> "$WORK/missing_art.jsonl"
    continue
  fi
  if [[ ! -f "$artifact_json" ]]; then
    emit_jsonl --string "label=$label" --string "path=$artifact_json" --string reason=artifact_json_not_found >> "$WORK/missing_art.jsonl"
    continue
  fi

  rt_status=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty runtime_validation.status)
  rt_reason=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty runtime_validation.reason)
  rt_rung=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty runtime_validation.rung)
  rt_kernel=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty runtime_validation.kernel)
  rt_device=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty runtime_validation.device_name)

  ART_RT_REC=$(emit_jsonl \
    --string "artifact_json=$artifact_json" \
    --string "device=$rt_device" \
    --string "kernel=$rt_kernel" \
    --string "label=$label" \
    --string "reason=$rt_reason" \
    --string "rung=$rt_rung" \
    --string "status=$rt_status")
  echo "$ART_RT_REC" >> "$WORK/art_rt.jsonl"
  [[ "$rt_status" != "pass" ]] && echo "$ART_RT_REC" >> "$WORK/bad_arts.jsonl"

  src_profile=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty source_profile.profile)
  profile="${src_profile:-$label}"

  cert_status=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.status)
  cert_reason=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.reason)
  cert_published=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.published_path)
  cert_pay_status=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.certificate.status)
  cert_pay_schema=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.certificate.schema)
  cert_kaxi_pat=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.kaxi_pattern)
  cert_sem=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.semantic_profile)
  cert_rt_status=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.runtime_validation.status)
  cert_rt_reason=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.runtime_validation.reason)
  cert_rt_rung=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.runtime_validation.rung)
  cert_rt_kernel=$("$KRETIKOS" kaxi-validate-evidence "$artifact_json" --print-or-empty kaxi_certificate.runtime_validation.kernel)

  fetched_bool=false
  fetched_status=""
  fetched_rt_status=""
  if [[ -n "$cert_published" ]]; then
    fetched_path="$(dirname "$artifact_json")/$cert_published"
    if [[ -f "$fetched_path" ]]; then
      fetched_bool=true
      fetched_status=$("$KRETIKOS" kaxi-validate-evidence "$fetched_path" --print-or-empty status)
      fetched_rt_status=$("$KRETIKOS" kaxi-validate-evidence "$fetched_path" --print-or-empty runtime.runtime_validation.status)
    elif [[ "$profile" =~ $KAXI_SUPPORTED_REGEX ]]; then
      emit_jsonl --string "label=$label" --string "path=$fetched_path" --string "profile=$profile" --string reason=published_certificate_not_found >> "$WORK/missing_cert.jsonl"
    fi
  fi

  if [[ "$fetched_bool" == "true" ]]; then
    if [[ "$fetched_status" != "$cert_pay_status" ]]; then
      emit_jsonl --string "embedded_status=$cert_pay_status" --string "fetched_status=$fetched_status" --string "label=$label" --string "profile=$profile" --string reason=embedded_and_fetched_certificate_status_mismatch >> "$WORK/bad_cert.jsonl"
    fi
    if [[ "$fetched_rt_status" != "$cert_rt_status" ]]; then
      emit_jsonl --string "embedded_runtime_status=$cert_rt_status" --string "fetched_runtime_status=$fetched_rt_status" --string "label=$label" --string "profile=$profile" --string reason=embedded_and_fetched_runtime_status_mismatch >> "$WORK/bad_cert.jsonl"
    fi
  fi

  emit_jsonl \
    --string "certificate_schema=$cert_pay_schema" \
    --string "certificate_status=$cert_pay_status" \
    --bool   "fetched=$fetched_bool" \
    --string "kaxi_pattern=$cert_kaxi_pat" \
    --string "label=$label" \
    --string "profile=$profile" \
    --string "published_path=$cert_published" \
    --string "reason=$cert_reason" \
    --string "runtime_kernel=$cert_rt_kernel" \
    --string "runtime_reason=$cert_rt_reason" \
    --string "runtime_rung=$cert_rt_rung" \
    --string "runtime_status=$cert_rt_status" \
    --string "semantic_profile=$cert_sem" \
    --string "status=$cert_status" >> "$WORK/kaxi_certs.jsonl"

  if [[ "$profile" =~ $KAXI_SUPPORTED_REGEX ]]; then
    [[ "$cert_status" != "pass" ]]    && emit_jsonl --string "detail=$cert_reason" --string "label=$label" --string "profile=$profile" --string reason=kaxi_certificate_status_not_pass --string "status=$cert_status" >> "$WORK/bad_cert.jsonl"
    [[ "$cert_pay_status" != "pass" ]] && emit_jsonl --string "label=$label" --string "profile=$profile" --string reason=kaxi_certificate_payload_not_pass --string "status=$cert_pay_status" >> "$WORK/bad_cert.jsonl"
    [[ "$cert_rt_status" != "pass" ]]  && emit_jsonl --string "label=$label" --string "profile=$profile" --string reason=kaxi_certificate_runtime_not_pass --string "runtime_reason=$cert_rt_reason" --string "status=$cert_rt_status" >> "$WORK/bad_cert.jsonl"
    [[ "$cert_rt_rung" != "$profile" ]] && emit_jsonl --string "label=$label" --string "profile=$profile" --string reason=kaxi_certificate_runtime_rung_mismatch --string "runtime_rung=$cert_rt_rung" >> "$WORK/bad_cert.jsonl"
  fi
done < <(tail -n +2 "$SUMMARY_COPY")

violation() {
  local why="$1" file="$2"
  if [[ "$failed" -eq 0 && -s "$file" ]]; then
    echo "$why $(tr '\n' ' ' < "$file")" >&2
    exit 1
  fi
}
violation "missing fetched artifacts:"      "$WORK/missing_art.jsonl"
violation "artifact runtime did not pass:"  "$WORK/bad_arts.jsonl"
violation "missing K-AXI certificates:"     "$WORK/missing_cert.jsonl"
violation "K-AXI certificate did not pass:" "$WORK/bad_cert.jsonl"

ART_RT_JSON=$(<"$WORK/art_rt.jsonl"        "$KRETIKOS" json-emit-array)
KAXI_CERTS_JSON=$(<"$WORK/kaxi_certs.jsonl" "$KRETIKOS" json-emit-array)
JOBS_JSON=$(<"$WORK/jobs.jsonl"             "$KRETIKOS" json-emit-array)
ART_DIRS_JSON=$(<"$WORK/art_dirs.jsonl"     "$KRETIKOS" json-emit-array)
ART_JSONS_JSON=$(<"$WORK/art_jsons.jsonl"   "$KRETIKOS" json-emit-array)
CASES_JSON=$(<"$WORK/cases.jsonl"           "$KRETIKOS" json-emit-array)

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STATUS="pass"; [[ "$failed" -ne 0 ]] && STATUS="fail"

# Top-level alphabetical (matches Python json.dump sort_keys=True).
"$KRETIKOS" json-emit \
  --raw-json "artifact_dirs=$ART_DIRS_JSON" \
  --raw-json "artifact_jsons=$ART_JSONS_JSON" \
  --raw-json "artifact_runtimes=$ART_RT_JSON" \
  --array-strings 'boundaries=local_workspace_is_build_and_inspection_host|slurm_worker_is_cuda_runtime_authority|runtime_acceptance_requires_cuda_driver_load_launch_and_copyback|supported_kaxi_profiles_must_publish_a_runtime_backed_kaxi_certificate|worker_side_ptxas_or_nvdisasm_missing_is_not_a_runtime_failure|each_worker_result_must_publish_and_fetch_a_json_artifact|does_not_claim_arbitrary_sounio_gpu_lowering' \
  --raw-json "cases=$CASES_JSON" \
  --raw-json "compute_capabilities=$CCS_JSON" \
  --raw-json "devices=$DEVICES_JSON" \
  --string "doctor_log=$(basename "$DOCTOR_LOG")" \
  --raw-json "drivers=$DRIVERS_JSON" \
  --int "failed=$failed" \
  --string "generated_at_utc=$NOW" \
  --raw-json "jobs=$JOBS_JSON" \
  --raw-json "kaxi_certificates=$KAXI_CERTS_JSON" \
  --string "manifest=$MANIFEST" \
  --string "manifest_log=$(basename "$MANIFEST_LOG")" \
  --string 'schema=sounio.kretikos.hpc-slurm-runtime-gate.v1' \
  --string "status=$STATUS" \
  --string "summary_tsv=$(basename "$SUMMARY_COPY")" \
  --int "total=$total" \
  > "$GATE_JSON"

echo "kretikos_hpc_slurm_runtime_gate: PASS out=$OUT_DIR total=$total failed=$failed"
