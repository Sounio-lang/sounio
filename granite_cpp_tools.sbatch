#!/usr/bin/env bash
#SBATCH --job-name=granite-cpp-tools
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-cpp-tools-%j.log
set -euo pipefail
work=/tmp/granite-cpp-tools-$SLURM_JOB_ID
stage=/orangefs/training/llm-granite-n10
kit="$stage/tooling/cpp-kit"
art="$stage/cpp-tools-$SLURM_JOB_ID"
mkdir -p "$work" "$art"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; rec "cpp-tools-exit-$rc"; cp "$work/"*.log "$art/" 2>/dev/null || true; if (( rc != 0 )); then for f in "$work/download.log" "$work/build-cmp.log" "$work/build-ids.log" "$work/ids-smoke.log" "$work/cmp-selftest.log"; do if [[ -s "$f" ]]; then rec "failure-log-$(basename "$f")"; tail -10 "$f" | sed -E "s@https?://[^ ]+@[URL]@g" | while IFS= read -r line; do rec "$line"; done; fi; done; fi' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
export PATH="$kit/bin:$PATH" LD_LIBRARY_PATH="$kit/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd "$kit" && sha256sum -c SHA256SUMS)
# Portable Clang-based C++ driver. No Zig source is used. Pin the official
# linux-x86_64 archive to the publisher's release manifest SHA256.
url=https://ziglang.org/download/0.15.2/zig-x86_64-linux-0.15.2.tar.xz
sha=02aa270f183da276e5b5920b1dac44a63f1a49e55050ebde3aecc9eb82f93239
zigdir="$stage/tooling/zig-x86_64-linux-0.15.2"
if [[ ! -x "$zigdir/zig" ]]; then
  # The original host curl binary cannot link to the worker libcurl.
  # The SHA-verified static BusyBox removes that shared-library dependency.
  SSL_CERT_FILE="$work/ca.crt" "$kit/bin/busybox" wget -O "$work/zig.tar.xz" "$url" > "$work/download.log" 2>&1
  printf '%s  %s\n' "$sha" "$work/zig.tar.xz" | sha256sum -c -
  "$kit/bin/busybox" xzcat "$work/zig.tar.xz" | tar -C "$work" -xf -
  mv "$work/zig-x86_64-linux-0.15.2" "$zigdir"
fi
export ZIG_GLOBAL_CACHE_DIR="$work/cache" ZIG_LOCAL_CACHE_DIR="$work/local-cache"
"$zigdir/zig" version
rec zig-cxx-ready
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
"$zigdir/zig" c++ -std=c++23 -O2 -Wall -Wextra -Werror src/tools/sounio-llm/cmp_61440.cpp -o "$art/cmp_61440" > "$work/build-cmp.log" 2>&1
"$art/cmp_61440" --selftest > "$work/cmp-selftest.log" 2>&1
rec cmp-selftest-pass
rec "cmp-sha-$(sha256sum "$art/cmp_61440" | cut -d' ' -f1)"
"$zigdir/zig" c++ -std=c++23 -O2 -Wall -Wextra -Werror -I"$kit/include" src/tools/sounio-llm/granite_prompt_ids.cpp "$kit/lib/libpcre2-8.a" -o "$art/granite_prompt_ids" > "$work/build-ids.log" 2>&1
rec "ids-sha-$(sha256sum "$art/granite_prompt_ids" | cut -d' ' -f1)"
model="$stage/granite-3.1-8b-instruct/tokenizer.json"
[[ -s "$model" && $(stat -c '%s' "$model") -gt 1000000 ]]
"$kit/bin/jq" -er '.model.vocab|to_entries[]|[(.key|@base64),.value]|@tsv' "$model" > "$work/vocab.tsv"
"$kit/bin/jq" -er '.model.merges[]|[(.[0]|@base64),(.[1]|@base64)]|@tsv' "$model" > "$work/merges.tsv"
"$kit/bin/jq" -er '.added_tokens[]|[(.content|@base64),.id]|@tsv' "$model" > "$work/specials.tsv"
"$art/granite_prompt_ids" "$work/vocab.tsv" "$work/merges.tsv" "$work/specials.tsv" --smoke > "$work/ids-smoke.log" 2>&1
rec GRANITE_CPP_TOKENIZER_SMOKE_PASS
sha256sum "$art/cmp_61440" "$art/granite_prompt_ids" "$model" > "$art/SHA256SUMS"
rec "cpp-artifacts-$art"
exit 0
__CA_BUNDLE__
