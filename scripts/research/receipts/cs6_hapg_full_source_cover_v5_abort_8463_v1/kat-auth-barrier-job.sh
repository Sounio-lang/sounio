#!/usr/bin/env bash
#SBATCH --job-name=cs6-hapg-kat-auth-v5
#SBATCH --partition=gpu-orangefs
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodelist=gpuorangefs-r770-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --export=NIL

set -euo pipefail

[[ $# -eq 3 ]]
root=/orangefs/training/cs6-hapg-cover/7e09b89b94a773c6
expected_auth_sha=$1
expected_guard_sha=$2
expected_barrier_sha=$3
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
[[ ${SLURMD_NODENAME:-} == gpuorangefs-r770-proxmox ]]
[[ $(sha256sum "$root/kat-authorization.txt" | awk '{print $1}') == "$expected_auth_sha" ]]
[[ $(sha256sum "$root/adaptive_submit_guard.py" | awk '{print $1}') == "$expected_guard_sha" ]]
[[ $(sha256sum "$root/kat_auth_barrier_job.sh" | awk '{print $1}') == "$expected_barrier_sha" ]]

guard_output=$(python3 -B "$root/adaptive_submit_guard.py")
expected_guard_output=$(printf '%s\n%s' \
  "HAPG_V5_KAT_AUTH_SHA256=$expected_auth_sha" \
  'HAPG_V5_ADAPTIVE_SUBMISSION_AUTHORIZED=true')
[[ $guard_output == "$expected_guard_output" ]]

guard_path="$root/logs/kat-auth-barrier-${SLURM_JOB_ID}.guard.txt"
record_path="$root/logs/kat-auth-barrier-${SLURM_JOB_ID}.scontrol.txt"
attestation_path="$root/kat-auth-barrier-job${SLURM_JOB_ID}.txt"
printf '%s\n' "$guard_output" > "${guard_path}.tmp"
scontrol -o show job "$SLURM_JOB_ID" > "${record_path}.tmp"
{
  echo 'SCHEMA=sounio.cs6.hapg-full-source-cover-kat-auth-barrier.v1'
  echo "BARRIER_JOB_ID=$SLURM_JOB_ID"
  echo "BARRIER_NODE=$SLURMD_NODENAME"
  echo "KAT_JOB_ID=8458"
  echo "KAT_AUTHORIZATION_SHA256=$expected_auth_sha"
  echo "KAT_GUARD_SHA256=$expected_guard_sha"
  echo "BARRIER_JOB_SCRIPT_SHA256=$expected_barrier_sha"
  echo "BARRIER_GUARD_OUTPUT_SHA256=$(sha256sum "${guard_path}.tmp" | awk '{print $1}')"
  echo "BARRIER_JOB_RECORD_SHA256=$(sha256sum "${record_path}.tmp" | awk '{print $1}')"
  echo 'KAT_GUARD_PASS=true'
} > "${attestation_path}.tmp"
chmod 0440 "${guard_path}.tmp" "${record_path}.tmp" "${attestation_path}.tmp"
mv "${guard_path}.tmp" "$guard_path"
mv "${record_path}.tmp" "$record_path"
mv "${attestation_path}.tmp" "$attestation_path"
printf '%s\n' "$guard_output"
echo "HAPG_V5_KAT_AUTH_BARRIER_JOB=$SLURM_JOB_ID"

# Keep the completed KAT authorization job visible while its dependent is submitted.
sleep 20
