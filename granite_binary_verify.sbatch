#!/usr/bin/env bash
#SBATCH --job-name=granite-binary-verify
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-binary-verify-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
recon=$stage/binary-reconstruct-29648
orig=$stage/provenance-29553
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
(cd "$recon" && sha256sum -c SHA256SUMS)
sha256sum -c "$orig/inputs.sha256"
rec L4_ORIGINAL_INPUTS_SHA256_PASS
rec "old-L4-build-log-$(sha256sum "$orig/build-audit.log" | cut -d' ' -f1)"
rec "new-L4-build-log-$(sha256sum "$recon/build-29553.log" | cut -d' ' -f1)"
if cmp -s "$orig/build-audit.log" "$recon/build-29553.log"; then rec L4_BUILD_LOG_IDENTICAL; else
  rec L4_BUILD_LOG_DIFFERENT
  if [[ -s "$recon/old-29553-build.diff.log" ]]; then head -14 "$recon/old-29553-build.diff.log" | while IFS= read -r l; do rec "$l"; done; fi
fi
exit 0
__CA_BUNDLE__
