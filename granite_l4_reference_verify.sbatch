#!/usr/bin/env bash
#SBATCH --job-name=granite-l4-reference-verify
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-l4-reference-verify-%j.log
set -euo pipefail
ref=/orangefs/training/llm-granite-n10/l4-ref-29553
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
cd "$ref"
sha256sum -c SHA256SUMS
[[ $(sha256sum cuda.tsv | cut -d' ' -f1) == 74b22db3f82221c5550593f2ce73f9b8430e8723ca219e45ce73b62b6c7c5fc5 ]]
[[ $(wc -l < cuda.tsv) == 481 ]]
awk -F '\t' 'BEGIN{n=0;h=0} /^#/ {h++;next} {if(NF!=5 || $1!=n || $3!=128) exit 1; n++} END{if(n!=480 || h!=1) exit 1}' cuda.tsv
if [[ -f run.log ]]; then grep -qx 'AUDIT_61440_DONE backend=cuda new_prompts=480 TOTAL_CUDA_FMA_MIN_NORM_HITS=0' run.log; rec L4_AUDIT_DONE_CONFIRMED; fi
head -1 cuda.tsv | grep -o 'binary_sha256=[0-9a-f]*' | while IFS= read -r l; do rec "$l"; done
while IFS= read -r l; do rec "$l"; done < SHA256SUMS
rec L4_PRESERVED_REF_SHA256_PASS
exit 0
__CA_BUNDLE__
