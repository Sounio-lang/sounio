#!/usr/bin/env bash
#SBATCH --job-name=granite-cpp-kit-repair
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-cpp-kit-repair-%j.log
set -euo pipefail
kit=/orangefs/training/llm-granite-n10/tooling/cpp-kit
work=/tmp/granite-cpp-kit-repair-$SLURM_JOB_ID
mkdir -p "$work"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; rec "kit-repair-exit-$rc"; for f in "$work/"*.log; do if [[ -f "$f" ]]; then while IFS= read -r line; do rec "$line"; done < "$f"; fi; done' EXIT
cd "$kit"
# 29286 accidentally hashed the manifest itself while it was empty. Verify
# the exact failed manifest change and all seven original payload digests.
printf '%s  %s\n' 0917d6bd7d281c51dd2e921a6292dd6ba08b4207bebe6465c071a61f6a35d07f SHA256SUMS | sha256sum -c -
awk '$2=="./SHA256SUMS" {if($1!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") exit 1; n++;next} {print; m++} END {if(n!=1 || m!=7) exit 1}' SHA256SUMS > "$work/fixed.sha256"
sha256sum -c "$work/fixed.sha256" > "$work/payload-check.log" 2>&1
cp SHA256SUMS "SHA256SUMS.failed-29286"
cp "$work/fixed.sha256" "SHA256SUMS.tmp-$SLURM_JOB_ID"
mv "SHA256SUMS.tmp-$SLURM_JOB_ID" SHA256SUMS
sha256sum -c SHA256SUMS
rec "kit-manifest-fixed-$(sha256sum SHA256SUMS | cut -d' ' -f1)"
export LD_LIBRARY_PATH="$kit/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$kit/bin/curl" --version > "$work/curl.log" 2>&1
"$kit/bin/jq" --version > "$work/jq.log" 2>&1
"$kit/bin/xz" --version > "$work/xz.log" 2>&1
rec KIT_PAYLOAD_AND_TOOLS_PASS
exit 0
__CA_BUNDLE__
