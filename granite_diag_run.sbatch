#!/usr/bin/env bash
#SBATCH --job-name=granite-diag-init
#SBATCH --partition=gpu-orangefs
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-diag-init-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
model=$stage/granite-3.1-8b-instruct
build=$stage/diag-build-${1:?build job id}
work=/tmp/granite-diag-init-$SLURM_JOB_ID
art=$stage/diag-run-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$work/"*.tsv "$work/"*.sha256 "$art/" 2>/dev/null || true; if [[ -s "$work/run.log" ]]; then grep -E "GRANITE_DIAG|AUDIT_61440|GRANITE_TOKEN_IDS_480|cuda-ptx" "$work/run.log" | while IFS= read -r l; do rec "$l"; done; fi; (cd "$art" && find . -maxdepth 1 -type f ! -name "SHA256SUMS*" -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS); rec "diag-exit-$rc"' EXIT
(cd "$build" && sha256sum -c SHA256SUMS)
(cd "$model" && sha256sum -c --quiet SHA256SUMS)
sha256sum "$stage/granite_480_prompts.txt" "$stage/granite_480_prompt_ids.tsv" > "$work/inputs.sha256"
[[ $(sha256sum "$stage/granite_480_prompts.txt" | cut -d' ' -f1) == e14ac28d67b08146f5640c08a820620ceffcf0734527e1699d158245a58f85bf ]]
[[ $(sha256sum "$stage/granite_480_prompt_ids.tsv" | cut -d' ' -f1) == 739880da2d40ff74b9a409f8a76d22d22ebc146b1afdb14007ae095512b8c22e ]]
cp "$build/audit" "$work/audit"
rec "diag-audit-sha-$(sha256sum "$work/audit" | cut -d' ' -f1)"
rec "diag-visible-devices-${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,memory.free,driver_version --format=csv,noheader | tee "$work/gpu-before.log" | while IFS= read -r l; do rec "$l"; done
ulimit -s unlimited
SOUNIO_AUDIT_BACKEND=cuda "$work/audit" --arch granite --model-dir "$model" --dtype bf16 --prompts 480 --tokens 128 --prompts-file "$stage/granite_480_prompts.txt" --prompt-ids "$stage/granite_480_prompt_ids.tsv" --out "$work/diagnostic.tsv" > "$work/run.log" 2>&1 &
pid=$!; rc=0; t0=$SECONDS; marker=0
while kill -0 "$pid" 2>/dev/null; do
  if grep -Eq '^AUDIT_61440_(INIT_)?FAIL$|^GRANITE_DIAG_FIRST_PROMPT_PASS ' "$work/run.log"; then marker=1; sleep 2; break; fi
  if (( SECONDS-t0 >= 300 )); then rec DIAG_DEADLINE; break; fi
  sleep 2
done
if kill -0 "$pid" 2>/dev/null; then
  grep -E '^(Name|State|Threads):' "/proc/$pid/status" > "$work/process-status.log" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
fi
wait "$pid" || rc=$?
rec "diag-process-rc-$rc"
if grep -Eq '^AUDIT_61440_(INIT_)?FAIL$' "$work/run.log"; then rec r-audit-fail; fi
# Diagnostic success means captured a terminal outcome, not numerical parity.
grep -Eq '^AUDIT_61440_(INIT_)?FAIL$|^GRANITE_DIAG_FIRST_PROMPT_PASS ' "$work/run.log"
rec GRANITE_DIAGNOSTIC_CAPTURED
exit 0
__CA_BUNDLE__
