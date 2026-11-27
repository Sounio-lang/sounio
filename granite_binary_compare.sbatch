#!/usr/bin/env bash
#SBATCH --job-name=granite-binary-compare
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-binary-compare-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
first=${1:?D repro job}
second=${2:?R770 repro job}
work=/tmp/granite-binary-compare-$SLURM_JOB_ID
art=$stage/binary-compare-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
for j in "$first" "$second"; do
  src=$stage/binary-repro-$j
  (cd "$src" && sha256sum -c SHA256SUMS)
  ca=/tmp/granite-binary-repro-$j/source-A
  cb=/tmp/granite-binary-repro-$j/source-B
  grep -aobF "$ca" "$src/A1" > "$art/repro-$j-A1-path-offsets.log" || true
  grep -aobF "$cb" "$src/B1" > "$art/repro-$j-B1-path-offsets.log" || true
  rec "path-occurrences-$j-A1-$(wc -l < "$art/repro-$j-A1-path-offsets.log")"
  rec "path-occurrences-$j-B1-$(wc -l < "$art/repro-$j-B1-path-offsets.log")"
  LC_ALL=C sed -b "s@$ca@$cb@g" "$src/A1" > "$work/$j-path-normalized"
  if cmp -s "$work/$j-path-normalized" "$src/B1"; then rec "PATH_ONLY_BYTE_DIFF-$j"; else rec "REMAINING_BYTE_DIFF-$j"; cmp -l "$work/$j-path-normalized" "$src/B1" | head -20 > "$art/repro-$j-remaining.diff" || true; fi
  if [[ -s "$src/old-29553-build-diff.log" ]]; then cp "$src/old-29553-build-diff.log" "$art/repro-$j-old-L4-build.diff"; fi
done
p=$stage/binary-repro-$first/A1
q=$stage/binary-repro-$second/A1
LC_ALL=C sed -b "s@/tmp/granite-binary-repro-$first/source-A@/tmp/granite-binary-repro-$second/source-A@g" "$p" > "$work/cross-node-normalized"
if cmp -s "$work/cross-node-normalized" "$q"; then rec CROSS_NODE_PATH_ONLY_BYTE_DIFF; else rec CROSS_NODE_REMAINING_BYTE_DIFF; cmp -l "$work/cross-node-normalized" "$q" | head -20 > "$art/cross-node-remaining.diff" || true; fi
cmp -s "$stage/binary-repro-$first/madaros-A" "$stage/binary-repro-$second/madaros-A" && rec CROSS_NODE_MADAROS_IDENTICAL
# Preserve the raw hashes of all compared executables and the original L4 inputs.
sha256sum "$p" "$q" "$stage/binary-repro-$first/madaros-A" "$stage/binary-repro-$second/madaros-A" > "$art/compared.sha256"
cp "$stage/provenance-29553/inputs.sha256" "$art/L4-inputs.sha256"
rec GRANITE_BINARY_DIFF_CAPTURED
exit 0
__CA_BUNDLE__
