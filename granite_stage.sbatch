#!/usr/bin/env bash
#SBATCH --job-name=granite-stage
#SBATCH --comment=lane:llm-granite
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --exclude=dl380p-dl380p-proxmox,gpuorangefs-5860-proxmox
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
if ! command -v git-lfs >/dev/null; then
  tool="$stage/tooling"
  [[ -d "$tool" ]]
  if [[ ! -x "$tool/git-lfs" ]]; then
    cat "$tool"/part-0[0-4] > "$work/git-lfs.gz"
    printf '%s  %s\n' '2d3445a9420c15b9ebe44595b83af257fcbee741e3454ff7bab194478d973db7' "$work/git-lfs.gz" | sha256sum -c -
    gzip -dc "$work/git-lfs.gz" > "$tool/git-lfs.tmp-$SLURM_JOB_ID"
    printf '%s  %s\n' '95795bacd7040d12e1d510309ea8abfe8d3007ff60b7d996af0df19298096fb8' "$tool/git-lfs.tmp-$SLURM_JOB_ID" | sha256sum -c -
    chmod +x "$tool/git-lfs.tmp-$SLURM_JOB_ID"
    mv "$tool/git-lfs.tmp-$SLURM_JOB_ID" "$tool/git-lfs"
  fi
  export PATH="$tool:$PATH"
fi
git lfs version
if [[ ! -d "$model/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://huggingface.co/ibm-granite/granite-3.1-8b-instruct "$model"
fi
cd "$model"
git lfs install --local
git lfs pull --include='*.safetensors,tokenizer.json,tokenizer_config.json,config.json,model.safetensors.index.json' --exclude=''
git lfs fsck
printf '%s\n' "$(git rev-parse HEAD)" > MODEL_REVISION
sha256sum *.safetensors *.json MODEL_REVISION > SHA256SUMS.tmp
mv SHA256SUMS.tmp SHA256SUMS
sha256sum -c SHA256SUMS
exit 0
__CA_BUNDLE__
