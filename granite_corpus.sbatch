#!/usr/bin/env bash
#SBATCH --job-name=granite-corpus
#SBATCH --no-requeue
#SBATCH --comment=lane:llm-granite
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-corpus-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
work=/tmp/granite-corpus-$SLURM_JOB_ID
mkdir -p "$work"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$1" true </dev/null >/dev/null 2>&1; }
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt" ZIG_GLOBAL_CACHE_DIR="$work/cache" ZIG_LOCAL_CACHE_DIR="$work/local-cache"
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
[[ $(git -C src rev-parse HEAD) == "$1" ]]
"$stage/tooling/zig-x86_64-linux-0.15.2/zig" c++ -std=c++23 -O2 -Wall -Wextra -Werror src/tests/llm/data/generate_granite_480.cpp -o generate
./generate > granite_480_prompts.txt
[[ $(wc -l < granite_480_prompts.txt) == 480 && $(LC_ALL=C sort -u granite_480_prompts.txt | wc -l) == 480 ]]
[[ $(grep -cve '^[[:space:]]*$' granite_480_prompts.txt) == 480 ]]
hash=$(sha256sum granite_480_prompts.txt | cut -d' ' -f1)
rec "corpus-sha-$hash"
# Readback over Slurm accounting: compressed exact generator bytes, numbered.
# The same hash must match the committed corpus before any audit submission.
gzip -n -c granite_480_prompts.txt | base64 -w 0 | fold -w 96 > chunks
n=0
while IFS= read -r chunk || [[ -n "$chunk" ]]; do
    rec "corpus-$(printf '%03d' "$n")-$chunk"
    n=$((n+1))
done < chunks
rec "corpus-chunks-$n"
cp granite_480_prompts.txt "$stage/granite_480_prompts.txt.new-$SLURM_JOB_ID"
if [[ -e "$stage/granite_480_prompts.txt" ]]; then cmp granite_480_prompts.txt "$stage/granite_480_prompts.txt"; rm "$stage/granite_480_prompts.txt.new-$SLURM_JOB_ID"; else mv "$stage/granite_480_prompts.txt.new-$SLURM_JOB_ID" "$stage/granite_480_prompts.txt"; fi
sha256sum "$stage/granite_480_prompts.txt" > "$stage/granite_480_prompts.txt.sha256"
rec GRANITE_CORPUS_GENERATOR_PASS
exit 0
__CA_BUNDLE__
