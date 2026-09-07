#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_PROCID:?Slurm rank required}"
export LD_LIBRARY_PATH=/scratch/pireus/runtime/lib
export APPTAINER_CACHEDIR=/tmp/pireus-apptainer-cache
export APPTAINER_TMPDIR=/tmp
overlay_args=()
if [[ "${PIREUS_MARLIN_OVERLAY:-0}" == "1" ]]; then
  overlay=/scratch/pireus/cache/modelopt-26b999da4d72fd238c32331782f72b6aa110165adec96fb8885f4252a5a7099c.py
  echo "26b999da4d72fd238c32331782f72b6aa110165adec96fb8885f4252a5a7099c  $overlay" | sha256sum -c -
  repack_overlay=/scratch/pireus/cache/marlin-utils-2d3207f11b4c7a0023fa508666b447170ff8325e06248b4928f21df77b03fc0f.py
  echo "2d3207f11b4c7a0023fa508666b447170ff8325e06248b4928f21df77b03fc0f  $repack_overlay" | sha256sum -c -
  overlay_args=(--bind "$overlay:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:ro"
                --bind "$repack_overlay:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/marlin_utils_fp4.py:ro")
fi
exec /scratch/pireus/runtime/apptainer-1.5.3/usr/bin/apptainer exec --nv "${overlay_args[@]}" \
  --bind /scratch/pireus:/scratch/pireus \
  --env LD_LIBRARY_PATH=/.singularity.d/libs \
  /scratch/pireus/images/inkling-spark.sif "$@"
