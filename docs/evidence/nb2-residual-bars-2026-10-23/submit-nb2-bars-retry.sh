#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
run_root=/scratch/sounio-nb2-residual-bars-20261023/run2
repo=/workspace/sounio-sprint-stats-viz-main9-logarithmic-20261021
commit=13c825f7a7b3fbc78c7c25754d212d244b09a1d2
mkdir -- "$run_root"
mkdir "$run_root/examples" "$run_root/logs" "$run_root/artifacts" "$run_root/artifacts/native" "$run_root/tmp" "$run_root/cache"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" examples/viz_nb2_residual_bars.sio examples/viz_nb2_residual_bars.md | tar -xf - -C "$run_root"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" diff 99082d7bcc404ef308347c25b2e67954ddf4c793 "$commit" --exit-code -- bin scripts self-hosted stdlib > "$run_root/COMPILER_INPUT_DIFF.txt"
printf 'compiler_inputs_git_diff=empty\ncompiler_build_commit=99082d7bcc404ef308347c25b2e67954ddf4c793\nconsumer_commit=%s\ncorrection=horizontal_axis_intervals_only\n' "$commit" > "$run_root/SOURCE.txt"
cp /tmp/nb2-bars-retry-job.sh "$run_root/job.sh"
cp /tmp/nb2-bars-retry-dispatch.sh "$run_root/dispatch.sh"
(cd "$run_root" && sha256sum examples/* > CONSUMER_FILES.sha256)
job=$(kubectl -n slurm-pilot exec -i slurm-pilot-worker-cpuops-4pvxd -c slurmd -- runuser -u sounio -- env -i PATH=/usr/bin:/bin /usr/bin/sbatch --parsable < "$run_root/job.sh")
[[ "$job" =~ ^[0-9]+$ ]]
printf 'job_id=%s\nsubmission_surface=worker_stdin\n' "$job" | tee "$run_root/SUBMISSION.txt"
