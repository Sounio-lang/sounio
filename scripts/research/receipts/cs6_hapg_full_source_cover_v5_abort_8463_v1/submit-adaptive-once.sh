#!/usr/bin/env bash
set -euo pipefail

root=/orangefs/training/cs6-hapg-cover/7e09b89b94a773c6
auth_sha=7073311d87709fc0d583ca2b037280b374308371c8e2301c4fe0a1d4fe1fb61d
guard_sha=f676ab45f3178c55a763ddbe43bca9e867aca60341c2d45f508137479cd6d5d3
barrier_sha=dc6e79e3dd04f7b90ff2c1487f1a2446f44d2b9d8a79c8a78cca78a08fd4ab53
config_sha=2154027e1fcea4e5e21375427355fce045a20da4975008f9f710291c310f94ce
kat_job=8458
kat_comment="HAPG_V5_KAT_AUTH_SHA256=${auth_sha},HAPG_V5_KAT_JOB=${kat_job}"

[[ ! -e $root/adaptive-submission.txt ]]
[[ $(sha256sum "$root/kat-authorization.txt" | awk '{print $1}') == "$auth_sha" ]]
[[ $(sha256sum "$root/adaptive_submit_guard.py" | awk '{print $1}') == "$guard_sha" ]]
[[ $(sha256sum "$root/kat_auth_barrier_job.sh" | awk '{print $1}') == "$barrier_sha" ]]
[[ $(sha256sum "$root/adaptive-config.txt" | awk '{print $1}') == "$config_sha" ]]

guard_output=$(python3 -B "$root/adaptive_submit_guard.py")
expected_guard_output=$(printf '%s\n%s' \
  "HAPG_V5_KAT_AUTH_SHA256=$auth_sha" \
  'HAPG_V5_ADAPTIVE_SUBMISSION_AUTHORIZED=true')
[[ $guard_output == "$expected_guard_output" ]]

barrier_submission=$(sbatch --parsable \
  --comment="$kat_comment" \
  --output="$root/logs/kat-auth-barrier-%j.out" \
  --error="$root/logs/kat-auth-barrier-%j.err" \
  "$root/kat_auth_barrier_job.sh" "$auth_sha" "$guard_sha" "$barrier_sha")
barrier_job=${barrier_submission%%;*}
barrier_job=${barrier_job//$'\r'/}
[[ $barrier_job =~ ^[0-9]+$ ]]

barrier_attestation="$root/kat-auth-barrier-job${barrier_job}.txt"
for _ in $(seq 1 60); do
  [[ -f $barrier_attestation ]] && break
  barrier_state=$(squeue -h -j "$barrier_job" -o '%T')
  [[ $barrier_state != FAILED && $barrier_state != CANCELLED && $barrier_state != TIMEOUT ]]
  sleep 1
done
[[ -f $barrier_attestation ]]
[[ $(awk -F= '$1=="BARRIER_JOB_ID" {print $2}' "$barrier_attestation") == "$barrier_job" ]]
[[ $(awk -F= '$1=="KAT_JOB_ID" {print $2}' "$barrier_attestation") == "$kat_job" ]]
[[ $(awk -F= '$1=="KAT_AUTHORIZATION_SHA256" {print $2}' "$barrier_attestation") == "$auth_sha" ]]
[[ $(awk -F= '$1=="KAT_GUARD_SHA256" {print $2}' "$barrier_attestation") == "$guard_sha" ]]
[[ $(awk -F= '$1=="BARRIER_JOB_SCRIPT_SHA256" {print $2}' "$barrier_attestation") == "$barrier_sha" ]]
[[ $(awk -F= '$1=="KAT_GUARD_PASS" {print $2}' "$barrier_attestation") == true ]]
barrier_attestation_sha=$(sha256sum "$barrier_attestation" | awk '{print $1}')

adaptive_comment="${kat_comment},HAPG_V5_KAT_BARRIER_JOB=${barrier_job}"
submission=$(sbatch --parsable \
  --dependency="afterok:$barrier_job" \
  --comment="$adaptive_comment" \
  --job-name=cs6-hapg-adaptive-v5 \
  --output="$root/logs/adaptive-%j.out" \
  --error="$root/logs/adaptive-%j.err" \
  "$root/hapg-job.sh" "$root/adaptive-config.txt" "$config_sha")
job_id=${submission%%;*}
job_id=${job_id//$'\r'/}
[[ $job_id =~ ^[0-9]+$ ]]

marker_tmp="$root/.adaptive-submission-${job_id}.tmp"
{
  echo 'SCHEMA=sounio.cs6.hapg-full-source-cover-adaptive-submission.v1'
  echo "ADAPTIVE_JOB_ID=$job_id"
  echo "ADAPTIVE_CONFIG_SHA256=$config_sha"
  echo "KAT_JOB_ID=$kat_job"
  echo "KAT_AUTHORIZATION_SHA256=$auth_sha"
  echo "KAT_GUARD_SHA256=$guard_sha"
  echo "KAT_AUTH_BARRIER_JOB_ID=$barrier_job"
  echo "KAT_AUTH_BARRIER_SCRIPT_SHA256=$barrier_sha"
  echo "KAT_AUTH_BARRIER_ATTESTATION_SHA256=$barrier_attestation_sha"
  echo "SLURM_DEPENDENCY=afterok:$barrier_job"
  echo "SLURM_COMMENT=$adaptive_comment"
  echo 'KAT_GUARD_PASS=true'
} > "$marker_tmp"
chmod 0440 "$marker_tmp"
mv "$marker_tmp" "$root/adaptive-submission.txt"
printf '%s\n' "$guard_output"
echo "KAT_AUTH_BARRIER_JOB_ID=$barrier_job"
echo "ADAPTIVE_JOB_ID=$job_id"
