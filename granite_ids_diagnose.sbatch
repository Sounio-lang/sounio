#!/usr/bin/env bash
#SBATCH --job-name=granite-ids-diagnose
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-ids-diagnose-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
kit=$stage/tooling/cpp-kit
model=$stage/granite-3.1-8b-instruct/tokenizer.json
export PATH="$kit/bin:$PATH" LD_LIBRARY_PATH="$kit/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
art=$stage/ids-diagnose-$SLURM_JOB_ID
mkdir -p "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
jq -c '.pre_tokenizer' "$model" | tee "$art/pretokenizer.json" | while IFS= read -r l; do rec "$l"; done
jq -c '.model.vocab as $v | [.model.merges[] | select(($v[.[0]] == null) or ($v[.[1]] == null) or ($v[.[0]+.[1]] == null))] | {count:length, examples:.[0:8]}' "$model" | tee "$art/missing-merges.json" | while IFS= read -r l; do rec "$l"; done
jq -c '.model.vocab as $v | [range(0;256) | if (.>=33 and .<=126) or (.>=161 and .<=172) or .>=174 then . else (if .<33 then 256+. elif .<161 then 289+.-127 elif .==173 then 323 else 0 end) end | [.]|implode|select($v[.] == null)]' "$model" | tee "$art/missing-bytes.json" | while IFS= read -r l; do rec "$l"; done
rec ids-diagnosis-done
exit 0
__CA_BUNDLE__
