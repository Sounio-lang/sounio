#!/usr/bin/env bash
#SBATCH --job-name=souc-sat-portfolio
#SBATCH --array=1-20
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#
# examples/erdos/portfolio_slurm.sbatch — P1 cluster fan-out for souc-sat.
#
# SLURM array: each task is one diversified portfolio worker (seed =
# SLURM_ARRAY_TASK_ID). Results stage to a shared path (OrangeFS when present),
# so the controller can pick the first task that reaches UNSAT and verify its
# certificate with drat-trim.
#
# NOTE on this cluster (measured, see plan §4): SLURM CfgTRES currently throttles
# allocatable CPUs to ~20 (8+4+4+4); set --array width accordingly until P0 (the
# operator-side CfgTRES unlock) lands. sbcast/--bcast were unreliable in prior
# runs, so this script assumes the repo (or at least the prebuilt ELF) is visible
# on a shared FS; otherwise stage $ELF via `srun --chdir` to a shared dir first.
#
# Submit:   sbatch examples/erdos/portfolio_slurm.sbatch [CLIQUE_N]
# Collect:  the controller scans $STAGE/seed_*/worker.log for UNSAT, then runs
#           drat-trim on the winner's souc_sat_worker.{cnf,drat}.
set -u

CLIQUE_N="${1:-7}"
SEED="${SLURM_ARRAY_TASK_ID:-1}"

# Shared staging directory (prefer OrangeFS if mounted on the compute nodes).
if [[ -d /orangefs/training ]]; then
  STAGE="/orangefs/training/souc_sat_portfolio_${SLURM_ARRAY_JOB_ID:-local}"
else
  STAGE="${STAGE:-$PWD/souc_sat_portfolio}"
fi
mkdir -p "$STAGE/seed_$SEED"

# Resolve the engine. Prefer a prebuilt ELF on the shared FS; else build here.
ROOT="${SOUNIO_ROOT:-$PWD}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="$ROOT/examples/erdos/souc_sat.sio"
ELF="$STAGE/souc_sat.elf"
if [[ ! -x "$ELF" ]]; then
  # Build once (array tasks race-free via mkdir lock).
  if mkdir "$STAGE/.build.lock" 2>/dev/null; then
    "$SOUC" "$SRC" "$ELF" && chmod +x "$ELF"
    touch "$STAGE/.build.done"
  fi
  # Wait for the builder.
  for _ in $(seq 1 600); do [[ -f "$STAGE/.build.done" ]] && break; sleep 1; done
fi

cd "$STAGE/seed_$SEED"
echo "task seed=$SEED node=$(hostname) clique=K_${CLIQUE_N}/$((CLIQUE_N-1))"
"$ELF" "$SEED" "$CLIQUE_N"
echo "task seed=$SEED done exit=$?"
