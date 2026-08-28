#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_KAXI_LOWERING_GATE_DIR:-$(mktemp -d /tmp/kretikos-kaxi-lowering.XXXXXX)}"
mkdir -p "$OUT_DIR"

PROFILE_SOURCE="examples/kretikos/real_vec_add.sio"
NEGATIVE_STDOUT="$OUT_DIR/profile_directive_negative.stdout"
NEGATIVE_STDERR="$OUT_DIR/profile_directive_negative.stderr"
GATE_JSON="$OUT_DIR/kretikos_kaxi_lowering_gate.v1.json"
LOWERING_CERTIFICATE="$OUT_DIR/epistemic_dual_output_f32.kaxi-lowering-certificate.json"
LOWERING_CERTIFICATE_WORK="$OUT_DIR/epistemic_dual_output_f32.lowering-certificate.d"
KNOWLEDGE_LOWERING_CERTIFICATE="$OUT_DIR/knowledge_dual_output_f32.kaxi-lowering-certificate.json"
KNOWLEDGE_LOWERING_CERTIFICATE_WORK="$OUT_DIR/knowledge_dual_output_f32.lowering-certificate.d"

lowering_cases=(
  "vec_add_f64|examples/kretikos/lower_vec_add_f64.sio|source_vec_add_f64|indexed_f64_vector_add|add"
  "vec_sub_f64|examples/kretikos/lower_vec_sub_f64.sio|source_vec_sub_f64|indexed_f64_vector_sub|sub"
  "vec_mul_f64|examples/kretikos/lower_vec_mul_f64.sio|source_vec_mul_f64|indexed_f64_vector_mul|mul"
  "vec_div_f64|examples/kretikos/lower_vec_div_f64.sio|source_vec_div_f64|indexed_f64_vector_div|div"
  "fma_f64|examples/kretikos/lower_fma_f64.sio|source_fma_f64|indexed_f64_affine_mad|mul,add"
  "vec_add_f32|examples/kretikos/lower_vec_add_f32.sio|source_vec_add_f32|indexed_f32_vector_add|add"
  "vec_sub_f32|examples/kretikos/lower_vec_sub_f32.sio|source_vec_sub_f32|indexed_f32_vector_sub|sub"
  "vec_mul_f32|examples/kretikos/lower_vec_mul_f32.sio|source_vec_mul_f32|indexed_f32_vector_mul|mul"
  "vec_div_f32|examples/kretikos/lower_vec_div_f32.sio|source_vec_div_f32|indexed_f32_vector_div|div"
  "fma_f32|examples/kretikos/lower_fma_f32.sio|source_fma_f32|indexed_f32_affine_mad|mul,add"
  "epistemic_dual_output_f32|examples/kretikos/lower_epistemic_dual_output_f32.sio|source_epistemic_dual_output_f32|indexed_f32_epistemic_dual_output|add,mul"
  "knowledge_dual_output_f32|examples/kretikos/lower_knowledge_dual_output_f32.sio|source_epistemic_dual_output_f32|indexed_f32_knowledge_dual_output|add,mul"
)

echo "kretikos_kaxi_lowering_gate: shell"
bash -n bin/kretikos "$0"

for item in "${lowering_cases[@]}"; do
  IFS='|' read -r label source kaxi_pattern recognized_pattern opcodes <<<"$item"
  asm_out="$OUT_DIR/${label}.kaxi"
  witness_out="$OUT_DIR/${label}.kaxi-witness.json"
  lowering_out="$OUT_DIR/${label}.kaxi-lowering.json"

  echo "kretikos_kaxi_lowering_gate: source-check label=$label source=$source"
  ./bin/kretikos check "$source"

  if grep -Eq 'kretikos:[[:space:]]*profile[[:space:]]*=' "$source"; then
    echo "kretikos_kaxi_lowering_gate: source unexpectedly contains a profile directive: $source" >&2
    exit 1
  fi

  echo "kretikos_kaxi_lowering_gate: kaxi-witness label=$label pattern=$kaxi_pattern"
  ./bin/kretikos kaxi-witness "$kaxi_pattern" -o "$witness_out" --asm-output "$asm_out"
  grep -q "get_tid r" "$asm_out"
  grep -q "load_global r" "$asm_out"
  grep -q "store_global r" "$asm_out"
  grep -q "ret seq=" "$asm_out"
  IFS=',' read -ra opcode_list <<<"$opcodes"
  for opcode in "${opcode_list[@]}"; do
    grep -q "$opcode r" "$asm_out"
  done

  echo "kretikos_kaxi_lowering_gate: source-lowering label=$label source=$source"
  ./bin/kretikos kaxi-lower-source "$source" -o "$lowering_out" --asm-output "$asm_out" --kaxi-witness-output "$witness_out"

  required_stores=1
  [[ "$kaxi_pattern" == "source_epistemic_dual_output_f32" ]] && required_stores=2
  ./bin/kretikos kaxi-validate-evidence "$lowering_out" \
    --expect "status=pass" \
    --expect "lowering.profile_hint=none" \
    --expect "lowering.fallback_path=none" \
    --expect "lowering.kaxi_pattern=$kaxi_pattern" \
    --expect "lowering.recognized_pattern=$recognized_pattern" \
    --expect "lowering.source_lowered_to_kaxi=true" \
    --expect "kaxi.epistemic_lanes.store_global_count>=$required_stores"
done

echo "kretikos_kaxi_lowering_gate: lowering-certificate label=epistemic_dual_output_f32"
./bin/kretikos kaxi-lowering-certificate \
  examples/kretikos/lower_epistemic_dual_output_f32.sio \
  -o "$LOWERING_CERTIFICATE" \
  --work-dir "$LOWERING_CERTIFICATE_WORK" \
  --force

./bin/kretikos kaxi-validate-evidence "$LOWERING_CERTIFICATE" \
  --expect "status=pass" \
  --expect "schema=sounio.kretikos.kaxi-lowering-certificate.v1" \
  --expect "lowering.profile_hint=none" \
  --expect "lowering.fallback_path=none" \
  --expect "lowering.recognized_pattern=indexed_f32_epistemic_dual_output" \
  --expect "lowering.kaxi_pattern=source_epistemic_dual_output_f32" \
  --expect "profile.name=epistemic_dual_output_f32" \
  --expect "runtime.runtime_validation.rung=epistemic_dual_output_f32" \
  --expect "kaxi.epistemic_lanes.store_global_count>=2"

echo "kretikos_kaxi_lowering_gate: lowering-certificate label=knowledge_dual_output_f32"
./bin/kretikos kaxi-lowering-certificate \
  examples/kretikos/lower_knowledge_dual_output_f32.sio \
  -o "$KNOWLEDGE_LOWERING_CERTIFICATE" \
  --work-dir "$KNOWLEDGE_LOWERING_CERTIFICATE_WORK" \
  --force

./bin/kretikos kaxi-validate-evidence "$KNOWLEDGE_LOWERING_CERTIFICATE" \
  --expect "status=pass" \
  --expect "schema=sounio.kretikos.kaxi-lowering-certificate.v1" \
  --expect "lowering.profile_hint=none" \
  --expect "lowering.fallback_path=none" \
  --expect "lowering.recognized_pattern=indexed_f32_knowledge_dual_output" \
  --expect "lowering.kaxi_pattern=source_epistemic_dual_output_f32" \
  --expect "profile.name=epistemic_dual_output_f32" \
  --expect "runtime.runtime_validation.rung=epistemic_dual_output_f32" \
  --expect "kaxi.epistemic_lanes.store_global_count>=2"

echo "kretikos_kaxi_lowering_gate: negative profile-directive rejection source=$PROFILE_SOURCE"
if ./bin/kretikos kaxi-lower-source "$PROFILE_SOURCE" -o "$OUT_DIR/profile_directive_negative.json" >"$NEGATIVE_STDOUT" 2>"$NEGATIVE_STDERR"; then
  cat "$NEGATIVE_STDOUT"
  cat "$NEGATIVE_STDERR" >&2
  echo "kretikos_kaxi_lowering_gate: profile directive source unexpectedly lowered" >&2
  exit 1
fi
grep -q "profile_directive_present" "$NEGATIVE_STDERR"

# Aggregate gate JSON via kretikos json-emit (replaces python json.dump heredoc).
# All upstream `kaxi-validate-evidence` calls have already halted on any
# validation mismatch (set -euo pipefail), so by the time we reach this point
# every case + both certificates are known-pass. failures stays []; status=pass.

__sha() { sha256sum "$1" | awk '{print $1}'; }
__wf() { ./bin/kretikos kaxi-validate-evidence "$1" --print-or-empty "$2"; }
__subtree() { ./bin/kretikos kaxi-validate-evidence "$1" --print-subtree-or "$2" '[]'; }
__opt_string() {
  local p="$1" v="$2"
  if [ -z "$v" ]; then printf '%s\0%s\0' "--null" "$p"
  else printf '%s\0%s\0' "--string" "${p}=${v}"
  fi
}
__opt_int() {
  local p="$1" v="$2"
  if [ -z "$v" ]; then printf '%s\0%s\0' "--null" "$p"
  else printf '%s\0%s\0' "--int" "${p}=${v}"
  fi
}
__opt_bool() {
  local p="$1" v="$2"
  if [ -z "$v" ]; then printf '%s\0%s\0' "--null" "$p"
  else printf '%s\0%s\0' "--bool" "${p}=${v}"
  fi
}
__emit_obj() { xargs -0 ./bin/kretikos json-emit "$@"; }

__case_objs=()
for __spec in "${lowering_cases[@]}"; do
  IFS='|' read -r __label __source __kaxi __recog __opcodes <<<"$__spec"
  __asm="$OUT_DIR/${__label}.kaxi"
  __wit="$OUT_DIR/${__label}.kaxi-witness.json"
  __low="$OUT_DIR/${__label}.kaxi-lowering.json"

  __ph=$(__wf "$__low" lowering.profile_hint)
  __fb=$(__wf "$__low" lowering.fallback_path)
  __rec=$(__wf "$__low" lowering.recognized_pattern)
  __kp=$(__wf "$__low" lowering.kaxi_pattern)
  __slk=$(__wf "$__low" lowering.source_lowered_to_kaxi)
  __sops=$(__subtree "$__low" lowering.semantic_ops)
  __sgc=$(__wf "$__wit" epistemic_lanes.store_global_count)
  __sd=$(__wf "$__wit" assembly.seq_dense_zero_based)
  __ops=$(__subtree "$__wit" assembly.opcodes)
  __req=1
  [[ "$__kaxi" == "source_epistemic_dual_output_f32" ]] && __req=2

  __obj=$({
    __opt_string profile_hint "$__ph"
    __opt_string fallback_path "$__fb"
    __opt_string recognized_pattern "$__rec"
    __opt_string kaxi_pattern "$__kp"
    __opt_bool source_lowered_to_kaxi "$__slk"
    __opt_bool seq_dense_zero_based "$__sd"
    __opt_int store_global_count "$__sgc"
  } | __emit_obj \
    --string "label=$__label" \
    --string "source=$__source" \
    --string "source_sha256=$(__sha "$__source")" \
    --raw-json "semantic_ops=$__sops" \
    --string "assembly=$(basename "$__asm")" \
    --string "assembly_sha256=$(__sha "$__asm")" \
    --string "witness=$(basename "$__wit")" \
    --string "witness_sha256=$(__sha "$__wit")" \
    --raw-json "opcodes=$__ops" \
    --int "required_store_global_count=$__req")
  __case_objs+=("$__obj")
done

# Build lowering_certificate sub-object.
__cert_obj() {
  local cert="$1"
  local schema status profile rec kp rs rr ru rk
  schema=$(__wf "$cert" schema)
  status=$(__wf "$cert" status)
  profile=$(__wf "$cert" profile.name)
  rec=$(__wf "$cert" lowering.recognized_pattern)
  kp=$(__wf "$cert" lowering.kaxi_pattern)
  rs=$(__wf "$cert" runtime.runtime_validation.status)
  rr=$(__wf "$cert" runtime.runtime_validation.reason)
  ru=$(__wf "$cert" runtime.runtime_validation.rung)
  rk=$(__wf "$cert" runtime.runtime_validation.kernel)
  {
    __opt_string schema "$schema"
    __opt_string status "$status"
    __opt_string profile "$profile"
    __opt_string recognized_pattern "$rec"
    __opt_string kaxi_pattern "$kp"
    __opt_string runtime_status "$rs"
    __opt_string runtime_reason "$rr"
    __opt_string runtime_rung "$ru"
    __opt_string runtime_kernel "$rk"
  } | __emit_obj \
    --string "path=$(basename "$cert")" \
    --string "sha256=$(__sha "$cert")"
}
__lc_obj=$(__cert_obj "$LOWERING_CERTIFICATE")
__klc_obj=$(__cert_obj "$KNOWLEDGE_LOWERING_CERTIFICATE")

# negative_cases is fixed shape: one entry recording the rejection check.
__neg_obj=$(./bin/kretikos json-emit \
  --string "name=profile_directive_rejected" \
  --string "source=examples/kretikos/real_vec_add.sio" \
  --string "stderr=$(basename "$NEGATIVE_STDERR")" \
  --string "status=pass")

__cases_arr="[$(IFS=,; echo "${__case_objs[*]:-}")]"
__neg_arr="[$__neg_obj]"

./bin/kretikos json-emit \
  --string "schema=sounio.kretikos.kaxi-source-lowering-gate.v1" \
  --string "generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --string "status=pass" \
  --string "out_dir=$OUT_DIR" \
  --int "case_count=${#__case_objs[@]}" \
  --raw-json "cases=$__cases_arr" \
  --raw-json "lowering_certificate=$__lc_obj" \
  --raw-json "knowledge_lowering_certificate=$__klc_obj" \
  --raw-json "negative_cases=$__neg_arr" \
  --raw-json "failures=[]" \
  --array-strings "boundaries=gate_proves_checked_source_to_kaxi_lowering_for_f32_arithmetic_corpus|gate_requires_profile_hint_none|gate_requires_fallback_path_none|gate_rejects_legacy_profile_directive_path|gate_proves_source_lowering_certificate_braids_lowering_kaxi_and_runtime_bundle_evidence|gate_does_not_claim_arbitrary_sounio_gpu_lowering|gate_does_not_replace_slurm_cuda_runtime_authority" \
  > "$GATE_JSON"

echo "kretikos_kaxi_lowering_gate: PASS out=$OUT_DIR"
