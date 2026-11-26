#!/usr/bin/env bash
#SBATCH --job-name=llm-granite-parity
#SBATCH --no-requeue
#SBATCH --comment=lane:llm-granite
#SBATCH --partition=gpu-orangefs
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/llm-granite-parity-%j.log
#
# Second model family for ledger N-10: ibm-granite/granite-3.1-8b-instruct,
# BF16 greedy, 480 prompts x L=128, on ONE CUDA backend per submission.
# r740: submit with --gres=gpu:quadro_rtx_8000:1 (RTX 8000, sm_75).
# r770: --nodelist=gpuorangefs-r770-proxmox --gres=gpu:1 (L4, sm_89), when not drained.
#
# FILL BEFORE SUBMITTING (the job refuses to run with placeholders):
#   HEAD_C     the commit carrying the granite forward pass + tokenizer
#   MODEL_DIR  staged weights, with SHA256SUMS next to them
#   REF_TSV    reference TSV (another CUDA run, or the Metal M5 Max reference
#              from the Mac lane, history=UNCOMPARED), plus its sha256
#              Use REF_TSV=none REF_SHA=none for the first CUDA reference.
#   PROMPTS_FILE / PROMPT_IDS  original 480 prompts + independent oracle TSV.
#   PROMPTS_SHA / PROMPT_IDS_SHA  expected SHA256 from the oracle receipt.
#   The value "manifest" reads the expected digest from the completed
#   dependency's SHA256SUMS (or PROMPT_IDS.sha256), never the candidate file.
#   CPP_TOOLS_DIR may name the pinned CPU-built comparator artifact directory.
#
# Prediction, registered before the first submission (2026-11-26):
#   With the Option A float model unchanged ("binary32, RNE, FTZ+DAZ, fixed op
#   order"), Granite BF16 greedy on this backend is bit-identical to the
#   reference: token_mismatches=0, runner_up=0, top1/top2/margin bit = 0,
#   max_margin_drift=0, verdict=CROSS_BACKEND_IDENTICAL.
# Model: every operation Granite adds over Qwen3 (embedding/attention/residual/
#   logit scalar multipliers; no QK-norm) is a single contract primitive (mul or
#   div) in a fixed position, so the cross-backend argument of
#   2026-11-24-cross-backend-bit-identity-receipt.md carries over unchanged.
# Named risks (a mismatch localized to one of these refutes the "no new work"
#   part of the prediction, not the float model): (a) residual multiplier not
#   loaded as exact bits 0x3E6147AE; (b) argmax tie-break not "lowest index on
#   equal bits" everywhere; (c) RoPE table (theta 1e7) depends on host libm;
#   (d) the 3-element vocab tail (49155 mod 32 = 3) reduced differently.
# Results: r-node-..., r-gpu-..., r-build-<sha>, r-run-rc<rc>-s<sec>,
#   r-cmp-<verdict>-tok<n>-margin<n>-drift<x>, r-tokcols-<sha>, r-abort-... .

set -uo pipefail
HEAD_C="__FILL_HEAD_COMMIT__"
MODEL_DIR="__FILL_MODEL_DIR__"
REF_TSV="__FILL_REF_TSV__"
REF_SHA="__FILL_REF_SHA256__"
PROMPTS_FILE="__FILL_PROMPTS_FILE__"
PROMPT_IDS="__FILL_PROMPT_IDS__"
PROMPTS_SHA="__FILL_PROMPTS_SHA256__"
PROMPT_IDS_SHA="__FILL_PROMPT_IDS_SHA256__"
CPP_TOOLS_DIR="${CPP_TOOLS_DIR:-}"
SELF="$0"
REMOTE="https://github.com/Sounio-lang/sounio.git"
JOB="${SLURM_JOB_ID:-manual}"
WORK="/tmp/llm-granite-parity-$JOB"
CA="$WORK/ca.crt"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$1" true < /dev/null > /dev/null 2>&1 || true; }
die() { rec "r-abort-$(echo "$*" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-60)"; exit 2; }
sha() { [[ -s "$1" ]] && sha256sum "$1" | cut -c1-12 || echo none; }
manifest_hash() { awk -v p="$1" '$2==p { print $1; n++ } END { if (n!=1) exit 1 }' "$2"; }
case "$HEAD_C$MODEL_DIR$REF_TSV$REF_SHA$PROMPTS_FILE$PROMPT_IDS$PROMPTS_SHA$PROMPT_IDS_SHA" in *__FILL_*) die "placeholders-not-filled";; esac
mkdir -p "$WORK/home" || die mkdir
ART="$(dirname "$MODEL_DIR")/audit-$JOB"
mkdir -p "$ART" || die "artifact-directory"
# Save failed-build diagnostics and partial TSV under a distinct name.
trap 'rc=$?; if (( rc != 0 )); then cp "$WORK/"*.log "$ART/" 2>/dev/null || true; if [[ ! -s "$ART/cuda.tsv" && -s "$WORK/cuda.tsv" ]]; then cp "$WORK/cuda.tsv" "$ART/cuda.partial.tsv"; sha256sum "$ART/cuda.partial.tsv" > "$ART/partial.sha256"; fi; printf "%s\n" "$rc" > "$ART/exit-code"; fi' EXIT
export HOME="$WORK/home" XDG_CACHE_HOME="$WORK/home/.cache"
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$SELF" > "$CA"
grep -q 'BEGIN CERTIFICATE' "$CA" 2>/dev/null || cp /etc/ssl/certs/ca-certificates.crt "$CA" 2>/dev/null || die "no CA"
ulimit -s unlimited 2>/dev/null || true
rec "r-node-$(hostname | cut -c1-30)"
rec "r-gpu-$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-60)"

( cd "$MODEL_DIR" && sha256sum -c --quiet SHA256SUMS ) > "$WORK/weights.check" 2>&1 || die "weights-sha256"
if [[ "$REF_TSV" != none ]]; then
  if [[ "$REF_SHA" == manifest ]]; then REF_SHA=$(manifest_hash "$REF_TSV" "$(dirname "$REF_TSV")/SHA256SUMS") || die "missing-ref-manifest"; fi
  [[ "$(sha256sum "$REF_TSV" | cut -d' ' -f1)" == "$REF_SHA" ]] || die "ref-sha256"
else
  [[ "$REF_SHA" == none ]] || die "ref-sha256"
fi
[[ -s "$PROMPTS_FILE" && -s "$PROMPT_IDS" ]] || die "missing-prompt-oracle"
if [[ "$PROMPTS_SHA" == manifest ]]; then PROMPTS_SHA=$(manifest_hash "$PROMPTS_FILE" "$PROMPT_IDS.sha256") || die "missing-prompts-manifest"; fi
if [[ "$PROMPT_IDS_SHA" == manifest ]]; then PROMPT_IDS_SHA=$(manifest_hash "$PROMPT_IDS" "$PROMPT_IDS.sha256") || die "missing-ids-manifest"; fi
[[ "$(sha256sum "$PROMPTS_FILE" | cut -d' ' -f1)" == "$PROMPTS_SHA" ]] || die "prompts-sha256"
[[ "$(sha256sum "$PROMPT_IDS" | cut -d' ' -f1)" == "$PROMPT_IDS_SHA" ]] || die "ids-sha256"
sha256sum "$PROMPTS_FILE" "$PROMPT_IDS" > "$WORK/inputs.sha256"

D="$WORK/src"
git init -q "$D" || die init
git -C "$D" -c http.sslCAInfo="$CA" fetch -q --depth 1 --no-tags "$REMOTE" "$HEAD_C" > /dev/null 2>&1 || die fetch
git -C "$D" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD || die checkout
[[ "$(git -C "$D" rev-parse HEAD)" == "$HEAD_C" ]] || die "HEAD-mismatch"

( cd "$D" && bash scripts/ci/build_modular_madaros.sh "$WORK/madaros" > "$WORK/build-madaros.log" 2>&1 ) || die "build-madaros"
( cd "$D" && SOUNIO_STDLIB_PATH="$D/stdlib" "$WORK/madaros" build tests/llm/llm_v2_61440_audit.sio -o "$WORK/audit" > "$WORK/build-audit.log" 2>&1 ) || die "build-audit"
if [[ "$REF_TSV" != none ]]; then
  if [[ -n "$CPP_TOOLS_DIR" ]]; then
    (cd "$CPP_TOOLS_DIR" && sha256sum -c --quiet SHA256SUMS) || die "cmp-tool-manifest"
    cp "$CPP_TOOLS_DIR/cmp_61440" "$WORK/cmp" || die "cmp-tool-copy"
  else
    ( cd "$D" && "${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror tools/sounio-llm/cmp_61440.cpp -o "$WORK/cmp" > "$WORK/build-cmp.log" 2>&1 ) || die "build-cmp"
  fi
fi
chmod +x "$WORK/audit" || die "chmod-audit"
rec "r-build-audit-$(sha "$WORK/audit")-cmp-$(sha "$WORK/cmp")"

# The audit tool's flags are whatever the granite PR defines; adjust this line to them.
t0=$SECONDS; rc=0
srun --ntasks=1 --cpus-per-task=8 --mem=0 --job-name="granite-run" --export=ALL \
  env SOUNIO_STDLIB_PATH="$D/stdlib" SOUNIO_AUDIT_BACKEND=cuda "$WORK/audit" --arch granite --model-dir "$MODEL_DIR" \
      --dtype bf16 --prompts 480 --tokens 128 --prompts-file "$PROMPTS_FILE" --prompt-ids "$PROMPT_IDS" --out "$WORK/cuda.tsv" < /dev/null > "$WORK/run.log" 2>&1 || rc=$?
rec "r-run-rc$rc-s$((SECONDS - t0))"
[[ "$rc" -eq 0 && -s "$WORK/cuda.tsv" ]] || die "run-failed"

grep -q GRANITE_TOKEN_IDS_480_PASS "$WORK/run.log" || die "token-oracle-failed"
awk -F '\t' 'BEGIN{n=0} /^#/ {next} {if(NF!=5 || $1!=n || $3!=128) exit 1; n++} END{if(n!=480) exit 1}' "$WORK/cuda.tsv" || die "incomplete-audit"
cp "$WORK/audit" "$WORK/cuda.tsv" "$WORK/run.log" "$WORK/inputs.sha256" "$WORK/build-"*.log "$ART/" || die "artifact-copy"
sha256sum "$ART/audit" "$ART/cuda.tsv" "$PROMPTS_FILE" "$PROMPT_IDS" > "$ART/SHA256SUMS"
if [[ -x "$WORK/cmp" ]]; then cp "$WORK/cmp" "$ART/cmp" || die "comparator-copy"; sha256sum "$ART/cmp" >> "$ART/SHA256SUMS"; fi
rec "r-tsv-sha-$(sha256sum "$ART/cuda.tsv" | cut -d' ' -f1)"
if [[ "$REF_TSV" == none ]]; then
  rec "r-reference-UNCOMPARED"
  exit 0
fi
"$WORK/cmp" "$REF_TSV" "$WORK/cuda.tsv" > "$WORK/cmp.out" 2>&1; crc=$?
cp "$WORK/cmp.out" "$ART/cmp.out" || die "comparison-copy"
sha256sum "$REF_TSV" "$ART/cmp.out" >> "$ART/SHA256SUMS"
v=$(grep -o 'verdict=[A-Z_]*' "$WORK/cmp.out" | head -1 | cut -d= -f2)
tm=$(grep -o 'token_mismatches=[0-9]*' "$WORK/cmp.out" | head -1 | cut -d= -f2)
mm=$(grep -o 'margin_bit_mismatches=[0-9]*' "$WORK/cmp.out" | head -1 | cut -d= -f2)
dr=$(grep -o 'max_margin_drift=[^ ]*' "$WORK/cmp.out" | head -1 | cut -d= -f2)
rec "r-cmp-rc$crc-${v:-none}-tok${tm:-x}-margin${mm:-x}-drift${dr:-x}"
rec "r-tokcols-$(grep -o 'cand_token_cols_sha256=[0-9a-f]*' "$WORK/cmp.out" | cut -d= -f2 | cut -c1-16)"
grep -m 3 -E 'first_divergent_op|TOK_MISMATCH' "$WORK/cmp.out" | while read -r l; do
  rec "r-div-$(echo "$l" | tr -c 'A-Za-z0-9_.:=-' '_' | cut -c1-100)"; done
rec "r-done"
exit "$crc"
__CA_BUNDLE__
