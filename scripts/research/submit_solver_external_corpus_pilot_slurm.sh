#!/usr/bin/env bash
# Submit the real external-corpus solver pilot to SLURM.
#
# Usage:
#   scripts/research/submit_solver_external_corpus_pilot_slurm.sh [out-dir]
#
# The job writes all artifacts outside the live repo tree. The Python runner
# records SLURM_JOB_ID, node list, tool root, and output paths in manifest.txt.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TOOL_ROOT="${SOUNIO_SOLVER_TOOL_ROOT:-$HOME/.cache/sounio/solver-tools}"

if [[ $# -gt 0 ]]; then
  OUT_DIR="$1"
elif [[ -n "${SOLVER_EXTERNAL_PILOT_OUT_DIR:-}" ]]; then
  OUT_DIR="$SOLVER_EXTERNAL_PILOT_OUT_DIR"
elif [[ -d /orangefs/training ]]; then
  OUT_DIR="/orangefs/training/sounio-solver-external-pilot-slurm-${STAMP}"
else
  OUT_DIR="/tmp/sounio-solver-external-pilot-slurm-${STAMP}"
fi

mkdir -p "$OUT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch not found on PATH; run from a SLURM login surface" >&2
  exit 127
fi

SBATCH_SCRIPT="$OUT_DIR/solver_external_corpus_pilot.sbatch"
cat >"$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=sounio-solver-pilot
#SBATCH --output=${OUT_DIR}/slurm-%j.out
#SBATCH --error=${OUT_DIR}/slurm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
cd "$ROOT_DIR"
export SOUNIO_SOLVER_TOOL_ROOT="$TOOL_ROOT"
export SOLVER_EXTERNAL_PILOT_OUT_DIR="$OUT_DIR"
export SOUNIO_SOLVER_ALLOW_UNVERIFIED_TLS="${SOUNIO_SOLVER_ALLOW_UNVERIFIED_TLS:-1}"
export PATH="\$SOUNIO_SOLVER_TOOL_ROOT/bin:\$PATH"

python3 scripts/research/run_solver_external_corpus_pilot.py \\
  --prepare-missing-tools \\
  --sat-count 12 \\
  --smt-count 12 \\
  --opb-count 12 \\
  --timeout-sec 30 \\
  --max-sat-candidates 400 \\
  --out-dir "$OUT_DIR"
EOF

sbatch "$SBATCH_SCRIPT"
echo "submitted_script=$SBATCH_SCRIPT"
echo "out_dir=$OUT_DIR"
