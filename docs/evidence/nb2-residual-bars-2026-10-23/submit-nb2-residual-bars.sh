#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
commit=${1:?source commit required}
[[ "$commit" =~ ^[a-f0-9]{40}$ ]] || exit 64
run_root=/scratch/sounio-nb2-residual-bars-20261023/run1
repo=/workspace/sounio-sprint-stats-viz-main9-logarithmic-20261021
kubectl -n slurm-pilot exec slurm-pilot-worker-cpuops-4pvxd -c slurmd -- install -d -m 700 -o 1000 -g 1000 /scratch/sounio-nb2-residual-bars-20261023
mkdir -- "$run_root"
mkdir "$run_root/source" "$run_root/logs" "$run_root/artifacts" "$run_root/tmp" "$run_root/cache"
mkdir "$run_root/artifacts/checks" "$run_root/artifacts/native"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin/souc bin/madaros bin/souc-linux-x86_64 bin/souc-lean-single-x86_64 bin/souc-lean-single-x86_64.SeedReceipt.json bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 scripts self-hosted stdlib examples/viz_nb2_residual_bars.sio examples/viz_nb2_deviance.sio | tar -xf - -C "$run_root/source"
cp /tmp/nb2-residual-bars-job.sh "$run_root/job.sh"
cp /tmp/nb2-residual-bars-dispatch.sh "$run_root/dispatch.sh"
printf 'source_commit=%s\nmain_basis=94866e463d\nsource_overlay=none\ncontrol_scripts=outside_source\n' "$commit" > "$run_root/SOURCE.txt"
(cd "$run_root/source" && find bin scripts self-hosted stdlib examples -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$run_root/SOURCE_FILES.sha256" && sha256sum --check --status "$run_root/SOURCE_FILES.sha256")
tar -czf "$run_root/source.tar.gz" -C "$run_root/source" .
sha256sum "$run_root/source.tar.gz" > "$run_root/SNAPSHOT.sha256"
job=$(kubectl -n slurm-pilot exec -i slurm-pilot-worker-cpuops-4pvxd -c slurmd -- runuser -u sounio -- env -i PATH=/usr/bin:/bin /usr/bin/sbatch --parsable < "$run_root/job.sh")
[[ "$job" =~ ^[0-9]+$ ]]
printf 'job_id=%s\nsubmission_surface=worker_stdin\nsource_and_outputs=worker_scratch_ext4\nforge_used=false\nquota_changed=false\n' "$job" | tee "$run_root/SUBMISSION.txt"
