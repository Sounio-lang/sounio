#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# -eq 6 ]] || { printf 'usage: %s PTX ARM64_LOADER SOUNIO_SOURCE FROZEN_OUTPUT COMPILER RECEIPT\n' "$0" >&2; exit 64; }
ptx="$1"; loader="$2"; source_file="$3"; frozen="$4"; compiler="$5"; receipt="$6"
nodes='gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54'
sha(){ sha256sum "$1"|cut -d' ' -f1; }
fail(){ printf 'pireus-dgx-ptx-xor: FAIL: %s\n' "$*" >&2;exit 1; }
[[ -s "$ptx" && -x "$loader" && -s "$source_file" && -s "$frozen" && -x "$compiler" ]]||fail 'missing input artifact'
[[ "$(wc -l <"$frozen")" -eq 16 ]]||fail 'frozen Sounio output must contain 16 lanes'
file "$loader"|grep -q 'ARM aarch64'||fail 'loader is not Linux/aarch64'
grep -q '^\.target sm_121' "$ptx"||fail 'PTX target mismatch'
grep -q '^\.visible \.entry sedenion_xor_product' "$ptx"||fail 'PTX entry absent'
source_sha="$(sha "$source_file")"; semantics_sha="$(sha "$frozen")"; ptx_sha="$(sha "$ptx")"
loader_sha="$(sha "$loader")"; compiler_sha="$(sha "$compiler")"
work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-xor-slurm.XXXXXX")"; output="$work/output"
salloc --partition=gpu-orangefs --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --nodelist="$nodes" --gres=gpu:gb10:1 --exclusive bash -lc "
  sbcast --force '$ptx' /tmp/pireus-sedenion-xor.ptx
  sbcast --force '$frozen' /tmp/pireus-xor-semantics.txt
  srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 --label \\
    --bcast=/tmp/pireus-xor-loader '$loader' /tmp/pireus-sedenion-xor.ptx \\
    /tmp/pireus-xor-semantics.txt '$source_sha' '$semantics_sha'
" >"$output" 2>&1
[[ "$(grep -c 'result=PASS lanes=16' "$output")" -eq 2 ]]||{ cat "$output" >&2;fail 'both DGX Spark ranks did not pass';}
job_id="$(sed -n 's/^salloc: Granted job allocation \([0-9][0-9]*\)$/\1/p' "$output"|tail -1)"
{
  printf 'schema=pireus-dgx-ptx-xor-slurm-receipt-v1\nsemantic_authority_language=Sounio\nsemantic_authority_role=SEMANTIC_AUTHORITY\n'
  printf 'sounio_source_sha256=%s\nfrozen_semantics_sha256=%s\nproducer_language=C++\nproducer_role=MATERIAL_PARITY\n' "$source_sha" "$semantics_sha"
  printf 'compiler_sha256=%s\nptx_sha256=%s\nloader_sha256=%s\ntoolchain=Ubuntu-24.04-aarch64-linux-gnu-g++-13.3.0\nhardware=2x-NVIDIA-DGX-Spark-GB10-sm121\n' "$compiler_sha" "$ptx_sha" "$loader_sha"
  printf 'compiler_command=bin/souc build tests/gpu/pireus_sed_xor_convolution_f64.sio --backend gpu --gpu-target dgx-sm121\n'
  printf 'material_command=salloc --partition=gpu-orangefs --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:gb10:1 --exclusive; sbcast; srun\n'
  printf 'slurm_job_id=%s\nnodes=%s\nmatched_ranks=2\nmatched_lanes_per_rank=16\nresult=PASS\n' "${job_id:-unknown}" "$nodes"
} >"$receipt"
cat "$output"
printf 'PIREUS_DGX_PTX_XOR_PASS receipt=%s sha256=%s\n' "$receipt" "$(sha "$receipt")"
