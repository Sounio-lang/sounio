#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# -eq 10 ]] || {
  printf 'usage: %s SHUFFLE_PTX GLOBAL_PTX ARM64_LOADER SOUNIO_SOURCE FROZEN COMPILER POLICY CONTRACT ITERATIONS RECEIPT\n' "$0" >&2
  exit 64
}
shuffle_ptx="$1"; global_ptx="$2"; loader="$3"; source_file="$4"; frozen="$5"
compiler="$6"; policy="$7"; contract="$8"; iterations="$9"; receipt="${10}"
nodes='gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54'
sha(){ sha256sum "$1"|cut -d' ' -f1; }
fail(){ printf 'pireus-operator-foundry: FAIL: %s\n' "$*" >&2;exit 1; }
for artifact in "$shuffle_ptx" "$global_ptx" "$loader" "$source_file" "$frozen" "$compiler" "$policy" "$contract"; do [[ -s "$artifact" ]]||fail "missing artifact $artifact"; done
[[ -x "$loader" && -x "$compiler" ]]||fail 'loader/compiler is not executable'
[[ "$iterations" =~ ^[1-9][0-9]*$ ]]||fail 'iterations must be positive'
[[ "$(wc -l <"$frozen")" -eq 16 ]]||fail 'frozen Sounio output must have 16 lanes'
grep -qx 'required_nodes=2' "$policy"||fail 'policy does not require both Sparks'
grep -qx 'required_candidates=warp-shuffle,global-load' "$policy"||fail 'policy candidate set drifted'
grep -qx 'metric=sum_ns_per_launch_across_nodes' "$policy"||fail 'policy metric drifted'
grep -q '^\.target sm_121' "$shuffle_ptx"||fail 'shuffle PTX target mismatch'
grep -q 'shfl.sync.bfly.b32' "$shuffle_ptx"||fail 'shuffle candidate lacks warp shuffle'
grep -q '^\.target sm_121' "$global_ptx"||fail 'global PTX target mismatch'
if grep -q 'shfl.sync.bfly.b32' "$global_ptx"; then fail 'global candidate contains warp shuffle'; fi
source_sha="$(sha "$source_file")"; frozen_sha="$(sha "$frozen")"; work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-foundry.XXXXXX")"; output="$work/output"
salloc --partition=gpu-orangefs --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist="$nodes" --gres=gpu:gb10:1 --exclusive bash -lc "
  sbcast --force '$shuffle_ptx' /tmp/pireus-xor-shuffle.ptx
  sbcast --force '$global_ptx' /tmp/pireus-xor-global.ptx
  sbcast --force '$frozen' /tmp/pireus-xor-semantics.txt
  srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 --label --bcast=/tmp/pireus-foundry-loader '$loader' /tmp/pireus-xor-shuffle.ptx sedenion_xor_product /tmp/pireus-xor-semantics.txt '$source_sha' '$frozen_sha' '$iterations'
  srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 --label /tmp/pireus-foundry-loader /tmp/pireus-xor-global.ptx sedenion_xor_product_global /tmp/pireus-xor-semantics.txt '$source_sha' '$frozen_sha' '$iterations'
" >"$output" 2>&1
shuffle_count=0; global_count=0; shuffle_sum=0; global_sum=0; shuffle_values=''; global_values=''
while IFS= read -r line; do
  if [[ "$line" == *'result=PASS candidate=sedenion_xor_product '* && "$line" =~ ns_per_launch=([0-9]+) ]]; then
    value="${BASH_REMATCH[1]}"; shuffle_sum=$((shuffle_sum+value)); shuffle_count=$((shuffle_count+1)); shuffle_values="${shuffle_values}${shuffle_values:+,}${value}"
  elif [[ "$line" == *'result=PASS candidate=sedenion_xor_product_global '* && "$line" =~ ns_per_launch=([0-9]+) ]]; then
    value="${BASH_REMATCH[1]}"; global_sum=$((global_sum+value)); global_count=$((global_count+1)); global_values="${global_values}${global_values:+,}${value}"
  fi
done <"$output"
[[ "$shuffle_count" -eq 2 && "$global_count" -eq 2 ]]||{ cat "$output" >&2;fail 'both candidates must pass on both nodes';}
winner='warp-shuffle'; winner_sum="$shuffle_sum"
if (( global_sum < shuffle_sum )); then winner='global-load'; winner_sum="$global_sum"; fi
job_id="$(sed -n 's/^salloc: Granted job allocation \([0-9][0-9]*\)$/\1/p' "$output"|tail -1)"
{
  printf 'schema=pireus-dgx-operator-foundry-receipt-v1\noperator=XorConvolution\nbits=4\ntwist=CayleyDicksonSign\n'
  printf 'semantic_authority_language=Sounio\nsounio_source_sha256=%s\nfrozen_semantics_sha256=%s\n' "$source_sha" "$frozen_sha"
  printf 'material_language=C++\nmaterial_role=MATERIAL_PARITY\ncompiler_sha256=%s\nloader_sha256=%s\n' "$(sha "$compiler")" "$(sha "$loader")"
  printf 'operator_contract_sha256=%s\npolicy_sha256=%s\nshuffle_ptx_sha256=%s\nglobal_ptx_sha256=%s\n' "$(sha "$contract")" "$(sha "$policy")" "$(sha "$shuffle_ptx")" "$(sha "$global_ptx")"
  printf 'toolchain=Ubuntu-24.04-aarch64-linux-gnu-g++-13.3.0\nhardware=2x-NVIDIA-DGX-Spark-GB10-sm121\n'
  printf 'shuffle_compile_command=bin/souc build tests/gpu/pireus_sed_xor_convolution_f64.sio --backend gpu --gpu-target dgx-sm121\n'
  printf 'global_compile_command=bin/souc build tests/gpu/pireus_sed_xor_convolution_global_f64.sio --backend gpu --gpu-target dgx-sm121\n'
  printf 'material_command=salloc --partition=gpu-orangefs --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:gb10:1 --exclusive; sbcast; srun\n'
  printf 'slurm_job_id=%s\nnodes=%s\niterations_per_candidate_per_node=%s\n' "${job_id:-unknown}" "$nodes" "$iterations"
  printf 'warp_shuffle_ns_per_launch=%s\nglobal_load_ns_per_launch=%s\n' "$shuffle_values" "$global_values"
  printf 'warp_shuffle_metric_sum=%s\nglobal_load_metric_sum=%s\nselected_lowering=%s\nselected_metric_sum=%s\nresult=PASS\n' "$shuffle_sum" "$global_sum" "$winner" "$winner_sum"
} >"$receipt"
cat "$output"
printf 'PIREUS_OPERATOR_FOUNDRY_PASS winner=%s receipt=%s sha256=%s\n' "$winner" "$receipt" "$(sha "$receipt")"
