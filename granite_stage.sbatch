#!/usr/bin/env bash
#SBATCH --job-name=granite-stage
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=02:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-stage-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
work=/tmp/granite-stage-$SLURM_JOB_ID
mkdir -p "$work"
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
if ! grep -q 'BEGIN CERTIFICATE' "$work/ca.crt"; then cp /etc/ssl/certs/ca-certificates.crt "$work/ca.crt"; fi
export GIT_SSL_CAINFO="$work/ca.crt"
export SSL_CERT_FILE="$work/ca.crt"
[[ -d /orangefs/training && -w /orangefs/training ]]
mkdir -p "$stage"
exec > >(tee "$stage/stage-$SLURM_JOB_ID.log") 2>&1
hostname
findmnt -T "$stage"
df -h "$stage"
model="$stage/granite-3.1-8b-instruct"
if [[ -s "$model/SHA256SUMS" ]]; then (cd "$model" && sha256sum -c SHA256SUMS); exit 0; fi
command -v git-lfs
if [[ ! -d "$model/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://huggingface.co/ibm-granite/granite-3.1-8b-instruct "$model"
fi
cd "$model"
git lfs pull --include='*.safetensors' --exclude=''
git lfs fsck
printf '%s\n' "$(git rev-parse HEAD)" > MODEL_REVISION
sha256sum *.safetensors *.json MODEL_REVISION > SHA256SUMS.tmp
mv SHA256SUMS.tmp SHA256SUMS
sha256sum -c SHA256SUMS
exit 0
__CA_BUNDLE__
