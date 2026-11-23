#!/usr/bin/env bash
# x86_64 container entry for the r740 pilot. Refuses an unpinned image.
# Does not use the Spark SIF, its dynamic loader, or the Spark library path.
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${PIREUS_R740_SIF_SHA256:?amd64 image hash is not pinned}"
if [[ ! "$PIREUS_R740_SIF_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "r740 image hash is not a pinned sha256" >&2
  exit 2
fi
sif=/scratch/pireus/images/pireus-r740-amd64.sif
echo "$PIREUS_R740_SIF_SHA256  $sif" | sha256sum -c -
prefix=/scratch/pireus/r740/apptainer-1.5.3/usr/bin/apptainer
test -x "$prefix"
exec "$prefix" exec --nv \
  --bind /scratch/pireus:/scratch/pireus \
  "$sif" "$@"
