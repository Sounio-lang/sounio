#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# -eq 4 ]] || {
  printf 'usage: %s CUDA_SOURCE HEADER ARM64_BINARY RECEIPT\n' "$0" >&2
  exit 64
}

source_file="$1"
header_file="$2"
binary="$3"
receipt="$4"
nodes='gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54'
expected_source='6820fa05ff91cb89012bb0a7651896e196d5ff379be8f90afdc0b8ae08a8688a'
expected_header='ae54c8f455d5ef057f182212aacd466bdf5e014898872706e80e51f6b16e7782'
expected_digest='76ff9d84a5537e93850fde22392e2aaebea56d796a2906527783dfd376bdc631'

fail() { printf 'pireus-dgx-material-slurm: FAIL: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

[[ "$(sha "$source_file")" == "$expected_source" ]] || fail 'CUDA source drifted'
[[ "$(sha "$header_file")" == "$expected_header" ]] || fail 'material header drifted'
[[ -x "$binary" ]] || fail 'ARM64 material comparator is not executable'
command -v salloc >/dev/null && command -v sbcast >/dev/null && command -v srun >/dev/null || \
  fail 'required Slurm transport is unavailable'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-dgx-slurm.XXXXXX")"
output="$work/output"
binary_sha="$(sha "$binary")"

salloc --partition=gpu-orangefs --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --nodelist="$nodes" --gres=gpu:gb10:1 --exclusive bash -lc "
    sbcast --force '$source_file' /tmp/pireus-dgx-material-parity.cu
    sbcast --force '$header_file' /tmp/material_sha256.hpp
    srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 --label \
      --bcast=/tmp/pireus-dgx-material-parity '$binary'
  " >"$output" 2>&1

[[ "$(grep -c 'result=PASS' "$output")" == 2 ]] || fail 'both ranks did not pass'
[[ "$(grep -c "candidate_digest_sha256=$expected_digest" "$output")" == 2 ]] || \
  fail 'rank digest mismatch'

job_id="$(sed -n 's/^salloc: Granted job allocation \([0-9][0-9]*\)$/\1/p' "$output" | tail -1)"
{
  printf 'schema=pireus-dgx-material-slurm-receipt-v1\n'
  printf 'producer_language=C++\nproducer_role=MATERIAL_PARITY\nsemantic_authority_language=Sounio\n'
  printf 'sounio_source_sha256=4be23864a14274d7996dd890473a5b3356a88441a589e509080c9978ba1cf404\n'
  printf 'frozen_semantics_sha256=a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336\n'
  printf 'cuda_source_sha256=%s\nheader_sha256=%s\nbinary_sha256=%s\n' "$expected_source" "$expected_header" "$binary_sha"
  printf 'transport=salloc+sbcast+srun\nslurm_job_id=%s\nnodes=%s\n' "${job_id:-captured-by-slurm}" "$nodes"
  printf 'candidate_digest_sha256=%s\nmatched_ranks=2\nresult=PASS\n' "$expected_digest"
} >"$receipt"
cat "$output"
printf 'PIREUS_DGX_MATERIAL_SLURM_PASS receipt=%s sha256=%s\n' "$receipt" "$(sha "$receipt")"
