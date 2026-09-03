#!/usr/bin/env bash
set -euo pipefail

# C.0 closure: KaxiAsmBuf at 256 KB makes per-fn frames exceed the 8 MB
# default Linux stack rlimit. Raise it for the ELF spawns below.
ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_KAXI_GATE_DIR:-$(mktemp -d /tmp/kretikos-kaxi.XXXXXX)}"
mkdir -p "$OUT_DIR"

SELF_CHECK_LOG="$OUT_DIR/kaxi_self_check.out"
GATE_JSON="$OUT_DIR/kretikos_kaxi_gate.v1.json"

echo "kretikos_kaxi_gate: shell"
bash -n bin/kretikos

echo "kretikos_kaxi_gate: self-check"
./bin/kretikos emit-kaxi self-check >"$SELF_CHECK_LOG" 2>&1
cat "$SELF_CHECK_LOG"
grep -q "PASS kaxi_self_check" "$SELF_CHECK_LOG"

echo "kretikos_kaxi_gate: end-to-end-sim"
E2E_LOG="$OUT_DIR/kaxi_end_to_end.out"
E2E_ELF="$OUT_DIR/kaxi_end_to_end.elf"
"$ROOT_DIR/bin/souc-linux-x86_64" "$ROOT_DIR/self-hosted/gpu/test_kaxi_driver.sio" "$E2E_ELF" >"$E2E_LOG" 2>&1 || {
  cat "$E2E_LOG"
  echo "kretikos_kaxi_gate: end-to-end-sim compile FAILED"
  exit 1
}
chmod +x "$E2E_ELF"
"$E2E_ELF" >"$E2E_LOG" 2>&1 || {
  cat "$E2E_LOG"
  echo "kretikos_kaxi_gate: end-to-end-sim run FAILED"
  exit 1
}
cat "$E2E_LOG"
grep -q "PASS kaxi_end_to_end" "$E2E_LOG"

echo "kretikos_kaxi_gate: differential-validation"
DIFF_LOG="$OUT_DIR/kaxi_differential.out"
"$ROOT_DIR/scripts/dev/run_kaxi_differential.sh" >"$DIFF_LOG" 2>&1 || {
  cat "$DIFF_LOG"
  echo "kretikos_kaxi_gate: differential-validation FAILED"
  exit 1
}
cat "$DIFF_LOG"
grep -q "PASS differential_validation" "$DIFF_LOG"

echo "kretikos_kaxi_gate: fuzz-validation"
FUZZ_LOG="$OUT_DIR/kaxi_fuzz.out"
"$ROOT_DIR/scripts/dev/run_kaxi_fuzz.sh" >"$FUZZ_LOG" 2>&1 || {
  cat "$FUZZ_LOG"
  echo "kretikos_kaxi_gate: fuzz-validation FAILED"
  exit 1
}
cat "$FUZZ_LOG"
grep -q "PASS fuzz_validation" "$FUZZ_LOG"

patterns=(
  exit_only
  vec_add
  vec_sub
  vec_mul
  vec_div
  fma
  source_vec_add_f64
  source_vec_sub_f64
  source_vec_mul_f64
  source_vec_div_f64
  source_fma_f64
  epistemic_elementwise_f32
  epistemic_dual_output_f32
  octonion_assoc
  octonion_mul
)

for pattern in "${patterns[@]}"; do
  out="$OUT_DIR/${pattern}.kaxi"
  witness="$OUT_DIR/${pattern}.kaxi-witness.json"
  log="$OUT_DIR/${pattern}.emit.out"
  echo "kretikos_kaxi_gate: witness pattern=$pattern out=$out witness=$witness"
  ./bin/kretikos kaxi-witness "$pattern" -o "$witness" --asm-output "$out" >"$log" 2>&1
  cat "$log"
  grep -q "K-AXI epistemic kernel assembly" "$out"
  grep -q "ret seq=" "$out"
  ./bin/kretikos kaxi-validate-evidence "$witness" \
      --expect "status=pass" \
      --expect "assembly.seq_dense_zero_based=true"
done

grep -q "add r" "$OUT_DIR/vec_add.kaxi"
grep -q "sub r" "$OUT_DIR/vec_sub.kaxi"
grep -q "mul r" "$OUT_DIR/vec_mul.kaxi"
grep -q "div r" "$OUT_DIR/vec_div.kaxi"
grep -q "fma r" "$OUT_DIR/fma.kaxi"
grep -q "add r" "$OUT_DIR/source_vec_add_f64.kaxi"
grep -q "type=f64" "$OUT_DIR/source_vec_add_f64.kaxi"
grep -q "store_global r" "$OUT_DIR/source_vec_add_f64.kaxi"
grep -q "sub r" "$OUT_DIR/source_vec_sub_f64.kaxi"
grep -q "type=f64" "$OUT_DIR/source_vec_sub_f64.kaxi"
grep -q "store_global r" "$OUT_DIR/source_vec_sub_f64.kaxi"
grep -q "mul r" "$OUT_DIR/source_vec_mul_f64.kaxi"
grep -q "type=f64" "$OUT_DIR/source_vec_mul_f64.kaxi"
grep -q "store_global r" "$OUT_DIR/source_vec_mul_f64.kaxi"
grep -q "div r" "$OUT_DIR/source_vec_div_f64.kaxi"
grep -q "type=f64" "$OUT_DIR/source_vec_div_f64.kaxi"
grep -q "store_global r" "$OUT_DIR/source_vec_div_f64.kaxi"
grep -q "mul r" "$OUT_DIR/source_fma_f64.kaxi"
grep -q "add r" "$OUT_DIR/source_fma_f64.kaxi"
grep -q "type=f64" "$OUT_DIR/source_fma_f64.kaxi"
grep -q "store_global r" "$OUT_DIR/source_fma_f64.kaxi"
grep -q "fma r" "$OUT_DIR/epistemic_elementwise_f32.kaxi"
grep -q "store_global r" "$OUT_DIR/epistemic_elementwise_f32.kaxi"
grep -q "add r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "mul r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "store_global r" "$OUT_DIR/epistemic_dual_output_f32.kaxi"
grep -q "mul r" "$OUT_DIR/octonion_assoc.kaxi"
grep -q "store_global r" "$OUT_DIR/octonion_assoc.kaxi"
grep -q "sub r" "$OUT_DIR/octonion_mul.kaxi"
grep -q "store_global r" "$OUT_DIR/octonion_mul.kaxi"

source_witnesses=(
  "vec_add_f32:examples/kretikos/real_vec_add.sio"
  "vec_sub_f32:examples/kretikos/real_vec_sub.sio"
  "vec_mul_f32:examples/kretikos/real_vec_mul.sio"
  "vec_div_f32:examples/kretikos/real_vec_div.sio"
  "fma_f32:examples/kretikos/real_fma_f32.sio"
  "epistemic_elementwise_f32:examples/kretikos/real_epistemic_elementwise.sio"
  "epistemic_dual_output_f32:examples/kretikos/real_epistemic_dual_output.sio"
  "vec_add_f64:examples/kretikos/real_vec_add_f64.sio"
  "vec_sub_f64:examples/kretikos/real_vec_sub_f64.sio"
  "vec_mul_f64:examples/kretikos/real_vec_mul_f64.sio"
  "vec_div_f64:examples/kretikos/real_vec_div_f64.sio"
  "fma_f64:examples/kretikos/real_fma_f64.sio"
)

for item in "${source_witnesses[@]}"; do
  label="${item%%:*}"
  src="${item#*:}"
  source_witness="$OUT_DIR/${label}.kaxi-source-witness.json"
  source_log="$OUT_DIR/${label}.source-witness.out"
  echo "kretikos_kaxi_gate: source-witness label=$label source=$src witness=$source_witness"
  ./bin/kretikos kaxi-source-witness "$src" -o "$source_witness" >"$source_log" 2>&1
  cat "$source_log"
  ./bin/kretikos kaxi-validate-evidence "$source_witness" \
      --expect "status=pass" \
      --expect "profile.name=$label"
done

for item in "${source_witnesses[@]}"; do
  label="${item%%:*}"
  src="${item#*:}"
  certificate="$OUT_DIR/${label}.kaxi-certificate.json"
  cert_work="$OUT_DIR/${label}.certificate.d"
  cert_log="$OUT_DIR/${label}.certificate.out"
  echo "kretikos_kaxi_gate: certificate label=$label source=$src certificate=$certificate"
  ./bin/kretikos kaxi-certificate "$src" -o "$certificate" --work-dir "$cert_work" --force >"$cert_log" 2>&1
  cat "$cert_log"
  ./bin/kretikos kaxi-validate-evidence "$certificate" \
      --expect "status=pass" \
      --expect "profile.name=$label" \
      --expect "runtime.runtime_validation.rung=$label"
done

# ── gate JSON aggregation (replaces python3 heredoc) ──────────────────────
__gate_sha256() { sha256sum "$1" 2>/dev/null | cut -c1-64 || echo ""; }
__gate_file_size() { stat -c%s "$1" 2>/dev/null || wc -c < "$1" 2>/dev/null || echo 0; }
__null_or_str() { local v="$1"; if [[ -z "$v" || "$v" == "null" ]]; then echo "null"; else printf '"%s"' "$v"; fi; }
__null_or_num() { local v="$1"; if [[ -z "$v" || "$v" == "null" ]]; then echo "null"; else echo "$v"; fi; }

CASES_JSON="$(
  {
    for pattern in "${patterns[@]}"; do
      path="$OUT_DIR/${pattern}.kaxi"
      witness_path="$OUT_DIR/${pattern}.kaxi-witness.json"
      artifact="$(basename "$path")"
      witness_name="$(basename "$witness_path")"
      sha256="$(__gate_sha256 "$path")"
      wsha256="$(__gate_sha256 "$witness_path")"
      bytes="$(__gate_file_size "$path")"
      instr_count="$(awk '!/^;/ && NF' "$path" | wc -l)"
      has_add="$(grep -qF "add r" "$path" 2>/dev/null && echo true || echo false)"
      has_div="$(grep -qF "div r" "$path" 2>/dev/null && echo true || echo false)"
      has_fma="$(grep -qF "fma r" "$path" 2>/dev/null && echo true || echo false)"
      has_mul="$(grep -qF "mul r" "$path" 2>/dev/null && echo true || echo false)"
      has_store_global="$(grep -qF "store_global r" "$path" 2>/dev/null && echo true || echo false)"
      has_sub="$(grep -qF "sub r" "$path" 2>/dev/null && echo true || echo false)"
      if grep -qF "var=+" "$path" 2>/dev/null || grep -qF "var=0%" "$path" 2>/dev/null; then
        has_variance=true
      else
        has_variance=false
      fi
      mapfile -t __wf < <(./bin/kretikos kaxi-validate-evidence "$witness_path" \
          --print-or-empty semantic_profile \
          --print-or-empty assembly.seq_dense_zero_based \
          --print-or-empty epistemic_lanes.store_global_count)
      sp_lit="$(__null_or_str "${__wf[0]:-}")"
      seq_lit="${__wf[1]:-null}"
      sc_lit="$(__null_or_num "${__wf[2]:-}")"
      printf '{"artifact":"%s","bytes":%s,"has_add":%s,"has_div":%s,"has_fma":%s,"has_mul":%s,"has_store_global":%s,"has_sub":%s,"has_variance_annotation":%s,"instruction_count":%s,"pattern":"%s","semantic_profile":%s,"seq_dense_zero_based":%s,"sha256":"%s","store_global_count":%s,"witness":"%s","witness_sha256":"%s"}\n' \
          "$artifact" "$bytes" \
          "$has_add" "$has_div" "$has_fma" "$has_mul" "$has_store_global" "$has_sub" "$has_variance" \
          "$instr_count" "$pattern" "$sp_lit" "$seq_lit" "$sha256" "$sc_lit" \
          "$witness_name" "$wsha256"
    done
  } | ./bin/kretikos json-emit-array
)"

SOURCE_CASES_JSON="$(
  {
    for f in $(ls "$OUT_DIR"/*.kaxi-source-witness.json 2>/dev/null | sort); do
      wname="$(basename "$f")"
      wsha256="$(__gate_sha256 "$f")"
      mapfile -t __sp < <(./bin/kretikos kaxi-validate-evidence "$f" \
          --print-or-empty kaxi.pattern \
          --print-or-empty kaxi.semantic_profile \
          --print-or-empty profile.name \
          --print-or-empty source.path \
          --print-or-empty source.sha256 \
          --print-or-empty status)
      sp_lit="$(__null_or_str "${__sp[1]:-}")"
      printf '{"kaxi_pattern":"%s","profile":"%s","semantic_profile":%s,"source":"%s","source_sha256":"%s","status":"%s","witness":"%s","witness_sha256":"%s"}\n' \
          "${__sp[0]:-}" "${__sp[2]:-}" "$sp_lit" "${__sp[3]:-}" "${__sp[4]:-}" "${__sp[5]:-}" \
          "$wname" "$wsha256"
    done
  } | ./bin/kretikos json-emit-array
)"

CERTIFICATE_CASES_JSON="$(
  {
    for f in $(ls "$OUT_DIR"/*.kaxi-certificate.json 2>/dev/null | sort); do
      cname="$(basename "$f")"
      csha256="$(__gate_sha256 "$f")"
      mapfile -t __cp < <(./bin/kretikos kaxi-validate-evidence "$f" \
          --print-or-empty kaxi.pattern \
          --print-or-empty kaxi.semantic_profile \
          --print-or-empty profile.name \
          --print-or-empty runtime.runtime_validation.kernel \
          --print-or-empty runtime.runtime_validation.reason \
          --print-or-empty runtime.runtime_validation.rung \
          --print-or-empty runtime.runtime_validation.status \
          --print-or-empty source.path \
          --print-or-empty source.sha256 \
          --print-or-empty status)
      sp_lit="$(__null_or_str "${__cp[1]:-}")"
      printf '{"certificate":"%s","certificate_sha256":"%s","kaxi_pattern":"%s","profile":"%s","runtime_kernel":%s,"runtime_reason":%s,"runtime_rung":%s,"runtime_status":%s,"semantic_profile":%s,"source":"%s","source_sha256":"%s","status":"%s"}\n' \
          "$cname" "$csha256" "${__cp[0]:-}" "${__cp[2]:-}" \
          "$(__null_or_str "${__cp[3]:-}")" "$(__null_or_str "${__cp[4]:-}")" \
          "$(__null_or_str "${__cp[5]:-}")" "$(__null_or_str "${__cp[6]:-}")" \
          "$sp_lit" "${__cp[7]:-}" "${__cp[8]:-}" "${__cp[9]:-}"
    done
  } | ./bin/kretikos json-emit-array
)"

GENERATED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
./bin/kretikos json-emit \
    --array-strings "boundaries=kaxi_is_sounio_owned_epistemic_gpu_assembly|gate_proves_self_hosted_emitter_compiles_and_emits_structural_artifacts|gate_proves_kaxi_witness_json_for_each_profile|gate_proves_checked_source_to_kaxi_witness_link_for_supported_profiles|gate_proves_kaxi_certificate_braids_source_kaxi_and_runtime_bundle_evidence|gate_proves_profile_level_lowering_for_explicit_kretikos_kaxi_profiles|gate_does_not_claim_arbitrary_sounio_gpu_lowering|gate_does_not_replace_slurm_cuda_runtime_authority" \
    --raw-json "cases=$CASES_JSON" \
    --raw-json "certificate_cases=$CERTIFICATE_CASES_JSON" \
    --string "generated_at_utc=$GENERATED_AT_UTC" \
    --string "schema=sounio.kretikos.kaxi-gate.v1" \
    --string "self_check_log=$(basename "$SELF_CHECK_LOG")" \
    --raw-json "source_cases=$SOURCE_CASES_JSON" \
    --string "status=pass" \
    > "$GATE_JSON" || exit 1

echo ""
echo "--- kaxi_ptx golden drift oracle (Phase C) ---"
bash "$ROOT_DIR/scripts/ci/kaxi_ptx_golden_gate.sh"

echo "kretikos_kaxi_gate: PASS out=$OUT_DIR"
