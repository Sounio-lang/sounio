#!/usr/bin/env bash
# od256 numeric gate on a DGX Spark (GB10 Blackwell, sm_121, aarch64) over SSH.
#
# The Spark is the CUDA runtime authority; this workspace stays the compiler/
# source authority (same split as scripts/dev/dgx_spark_epistemic_wmma_matmul_gate.sh).
# It manages no passwords — uses the ssh config's key (Host dgx-spark-{1,2}).
#
# Flow: gen fixtures locally → scp payload → build the runner ON the Spark
# (aarch64 gcc; the x86_64 prebuilt binary can't run there, so run_numeric.sh
# falls back to building from source) → JIT-run every kernel (driver JIT via
# cuModuleLoadDataEx; ptxas not required) → scp results back → --check vs mpmath.
#
# Usage (from the eisa worktree):
#   bash slurm-jobs/kaxi-ptxas-accept/submit-od256-numeric-spark.sh            # Spark #2, 1024 cases
#   CASES=4096 ROLE=experimental bash …/submit-od256-numeric-spark.sh          # heavier stress
#   ROLE=canonical bash …/submit-od256-numeric-spark.sh                        # Spark #1 (.43)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CASES="${CASES:-1024}"
MAX_THREADS="${MAX_THREADS:-128}"
ROLE="${ROLE:-experimental}"                       # experimental=#2(.24, may fail), canonical=#1(.43)
[[ "$ROLE" == canonical ]] && HOSTALIAS="dgx-spark-1" || HOSTALIAS="dgx-spark-2"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
[[ -f "$SSH_CONFIG" ]] || SSH_CONFIG=/workspace/.home/openvscode-server/.agents/claude-2/.ssh/config
REMOTE_DIR="${REMOTE_DIR:-/tmp/sounio-od256-numeric-$(id -u)}"
SSH=(ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=15)
SCP=(scp -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=15)

STAGE_LOCAL="${STAGE_LOCAL:-/tmp/od256_numeric_spark_stage}"
rm -rf "$STAGE_LOCAL"; mkdir -p "$STAGE_LOCAL/ptx" "$STAGE_LOCAL/fixtures"

# --- 1. stage payload -------------------------------------------------------
for p in od256_two_sum od256_two_prod od256_add od256_mul od256_div od256_sqrt; do
  g="tests/golden/kaxi_ptx/od256/${p}.ptx"
  [[ -f "$g" ]] || { echo "missing golden $g" >&2; exit 1; }
  cp -f "$g" "$STAGE_LOCAL/ptx/${p}.ptx"
done
python3 scripts/ci/od256_numeric_gate.py --gen "$STAGE_LOCAL/fixtures" --cases "$CASES" ${ADVERSARIAL:+--adversarial}
cp -f scripts/gpu/kaxi_ptx_runner.c "$STAGE_LOCAL/kaxi_ptx_runner.c"
cp -f slurm-jobs/kaxi-ptxas-accept/run_numeric.sh "$STAGE_LOCAL/run_numeric.sh"
chmod +x "$STAGE_LOCAL/run_numeric.sh"
tar -C "$STAGE_LOCAL" -czf "$STAGE_LOCAL/payload.tgz" ptx fixtures kaxi_ptx_runner.c run_numeric.sh

# --- 2. preflight + ship ----------------------------------------------------
echo "== Spark preflight ($HOSTALIAS) =="
"${SSH[@]}" "$HOSTALIAS" 'hostname; uname -m; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1' | sed 's/^/  /'
"${SSH[@]}" "$HOSTALIAS" "rm -rf '$REMOTE_DIR' && mkdir -p '$REMOTE_DIR/results'"
"${SCP[@]}" "$STAGE_LOCAL/payload.tgz" "$HOSTALIAS:$REMOTE_DIR/payload.tgz" >/dev/null
echo "shipped $CASES cases/kernel → $HOSTALIAS:$REMOTE_DIR"

# --- 3. run on the Spark ----------------------------------------------------
"${SSH[@]}" "$HOSTALIAS" "bash -lc '
  set -e
  cd \"$REMOTE_DIR\"
  tar -xzf payload.tgz
  chmod +x run_numeric.sh
  MAX_THREADS=$MAX_THREADS bash run_numeric.sh ./ptx ./fixtures ./results ./kaxi_ptx_runner.c
'" 2>&1 | sed 's/^/  /' || echo "  (run_numeric returned nonzero — check summary)"

# --- 4. fetch + verify ------------------------------------------------------
D="/tmp/od256-spark-${ROLE}.results"; rm -rf "$D"; mkdir -p "$D"
"${SSH[@]}" "$HOSTALIAS" "cd '$REMOTE_DIR/results' && tar -cz ." | tar -xz -C "$D"
echo "== Spark summary =="; cat "$D/summary.txt" 2>/dev/null | sed 's/^/  /'
echo "== GATE (real GB10 output vs mpmath) =="
python3 scripts/ci/od256_numeric_gate.py --check "$D" --min-bits 400
echo "gate exit=$?  (results in $D)"
