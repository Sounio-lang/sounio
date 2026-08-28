#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT_DIR="$ROOT/results/associator_gum_variance"
mkdir -p "$OUT_DIR"
SRC="experiments/associator_gum_variance/associator_gum_variance.sio"
SOUC="${SOUC:-$ROOT/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
TS_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG="$OUT_DIR/RUNLOG.txt"
{
  echo "ts_utc=$TS_UTC"
  echo "git_sha=$GIT_SHA"
  echo "git_branch=$GIT_BRANCH"
  echo "cmd: $SOUC run $SRC"
  echo "----- stdout+stderr -----"
  "$SOUC" run "$SRC" 2>&1
} | tee "$LOG"

get() { grep -E "^$1" "$LOG" | tail -1 | sed "s/^$1//"; }
COMPLETE=$(grep -c "ASSOC_GUM_EXPERIMENT_COMPLETE" "$LOG" || true)
test "$COMPLETE" -ge 1
VERDICT=$(get "ASSOC_GUM_VERDICT=")
test -n "$VERDICT"

RECEIPT="$OUT_DIR/receipt.v1.json"
cat > "$RECEIPT" <<EOF
{
  "schema": "associator_gum_variance.receipt.v1",
  "ts_utc": "$TS_UTC",
  "git_sha": "$GIT_SHA",
  "git_branch": "$GIT_BRANCH",
  "protocol": "experiments/associator_gum_variance/PROTOCOL.md",
  "design": "docs/superpowers/specs/2026-07-25-associator-gum-variance-design.md",
  "source_analysis": "docs/research/variance_of_associator.md",
  "engine": "bin/souc",
  "fano": {
    "sigma": $(get "ASSOC_GUM_FANO_SIGMA="),
    "A0": $(get "ASSOC_GUM_FANO_A0="),
    "truth_fo_64sig2": $(get "ASSOC_GUM_FANO_TRUTH_FO="),
    "fd_var": $(get "ASSOC_GUM_FANO_FD_VAR="),
    "mc_var": $(get "ASSOC_GUM_FANO_MC_VAR="),
    "mc_mean": $(get "ASSOC_GUM_FANO_MC_MEAN="),
    "stepwise_var_32sig2": $(get "ASSOC_GUM_FANO_STEPWISE_VAR="),
    "note_printed_16sig2": $(get "ASSOC_GUM_FANO_NOTE_16SIG2="),
    "rel_mc": $(get "ASSOC_GUM_FANO_REL_MC="),
    "ratio_truth_over_stepwise": $(get "ASSOC_GUM_FANO_RATIO_TRUTH_OVER_STEPWISE="),
    "n_mc": $(get "ASSOC_GUM_FANO_N_MC=")
  },
  "quaternion": {
    "A0": $(get "ASSOC_GUM_QUAT_A0="),
    "fd_var": $(get "ASSOC_GUM_QUAT_FD_VAR="),
    "stepwise_proxy": $(get "ASSOC_GUM_QUAT_STEPWISE_PROXY=")
  },
  "verdict": "$VERDICT",
  "allowed_claim": "On the locked Fano-triple associator probe (a=e1,b=e2,c=e4, only a1 uncertain), first-order GUM truth 64σ² matches finite-difference GUM and Monte Carlo (rel err <10%). A covariance-blind stepwise model yields 32σ² (factor-2 underestimate; research note's 16σ² figure drops a factor of two in the last step). Quaternion subalgebra has A≡0 and FO variance ~0 while a blind independence proxy stays positive.",
  "forbidden_claims": [
    "compiler variance_of on Knowledge octonion chains is GUM-correct",
    "clinical or patient-level meaning"
  ]
}
EOF
python3 -m json.tool "$RECEIPT" > /dev/null
echo "wrote $RECEIPT verdict=$VERDICT"
