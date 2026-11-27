#!/usr/bin/env bash
#SBATCH --job-name=granite-reference-ids
#SBATCH --no-requeue
#SBATCH --comment=lane:llm-granite
#SBATCH --partition=all
#SBATCH --exclude=dl380p-dl380p-proxmox,gpuorangefs-5860-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-reference-%j.log
# Registered corpus: tests/llm/data/granite_480_prompts.txt SHA256=e14ac28d67b08146f5640c08a820620ceffcf0734527e1699d158245a58f85bf
set -euo pipefail
head_c=${1:?source commit}
model=${2:?model directory}
prompts=${3:?registered Granite 480-prompt file}
ref=${4:?output reference TSV path}
cpp_art=${5:-}
corpus_commit=${6:?registered corpus commit}
corpus_path=tests/llm/data/granite_480_prompts.txt
corpus_sha=e14ac28d67b08146f5640c08a820620ceffcf0734527e1699d158245a58f85bf
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$1" true </dev/null >/dev/null 2>&1 || true; }
work=/tmp/granite-reference-$SLURM_JOB_ID
mkdir -p "$work"
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
(cd "$model" && sha256sum -c --quiet SHA256SUMS)
[[ -s "$prompts" && ! -e "$ref" ]]
[[ $(sha256sum "$prompts" | cut -d" " -f1) == "$corpus_sha" ]]
[[ $(wc -l < "$prompts") -eq 480 ]]
kit="$(dirname "$model")/tooling/cpp-kit"
if ! command -v jq >/dev/null; then export PATH="$kit/bin:$PATH" LD_LIBRARY_PATH="$kit/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
command -v jq
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$head_c"
git -C src checkout -q --detach FETCH_HEAD
[[ $(git -C src rev-parse HEAD) == "$head_c" ]]
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$corpus_commit"
[[ $(git -C src rev-parse FETCH_HEAD) == "$corpus_commit" ]]
git -C src show "$corpus_commit:$corpus_path" > registered.txt
cmp -s registered.txt "$prompts"
rec "registered-corpus-sha-$corpus_sha"
if [[ -n "$cpp_art" ]]; then
  (cd "$cpp_art" && sha256sum -c --quiet SHA256SUMS)
  cp "$cpp_art/granite_prompt_ids" ids
else
  "${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror src/tools/sounio-llm/granite_prompt_ids.cpp -lpcre2-8 -o ids
fi
jq -er '.model.vocab|to_entries[]|[(.key|@base64),.value]|@tsv' "$model/tokenizer.json" > vocab.tsv
jq -er '.model.merges[]|[(.[0]|@base64),(.[1]|@base64)]|@tsv' "$model/tokenizer.json" > merges.tsv
jq -er '.added_tokens[]|[(.content|@base64),.id]|@tsv' "$model/tokenizer.json" > specials.tsv
./ids vocab.tsv merges.tsv specials.tsv "$prompts" > "$work/ref.tsv"
[[ $(wc -l < "$work/ref.tsv") -eq 480 ]]
cp "$work/ref.tsv" "$ref.tmp-$SLURM_JOB_ID"
mv "$ref.tmp-$SLURM_JOB_ID" "$ref"
cp "$work/ids" "$ref.ids.bin"
sha256sum "$model/tokenizer.json" "$prompts" "$ref" "$ref.ids.bin" > "$ref.sha256"
cat "$ref.sha256"
rec "oracle-sha-$(sha256sum "$ref" | cut -d" " -f1)"
rec GRANITE_REFERENCE_480_PASS
exit 0
__CA_BUNDLE__
