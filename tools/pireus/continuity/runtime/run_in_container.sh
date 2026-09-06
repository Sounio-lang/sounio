#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_PROCID:?Slurm rank required}"
export LD_LIBRARY_PATH=/scratch/pireus/runtime/lib
export APPTAINER_CACHEDIR=/tmp/pireus-apptainer-cache
export APPTAINER_TMPDIR=/tmp
exec /scratch/pireus/runtime/apptainer-1.5.3/usr/bin/apptainer exec --nv \
  --bind /scratch/pireus:/scratch/pireus \
  --env LD_LIBRARY_PATH=/.singularity.d/libs \
  /scratch/pireus/images/inkling-spark.sif "$@"
