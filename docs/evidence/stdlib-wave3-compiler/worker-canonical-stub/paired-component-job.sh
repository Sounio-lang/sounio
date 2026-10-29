#!/bin/bash
#SBATCH --job-name=stdlib-worker-canonical-stub
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:08:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-canonical-stub01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-canonical-stub01/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -s 524288;ulimit -c 0
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/compiler-resumed/worker-canonical-stub01
D=$R/results
RAW=$B/root-integrated03/results/madaros
SEED=$B/root-integrated03/source/bin/souc-lean-single-x86_64
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
sha256sum "$RAW" "$SEED" "$B/root-integrated03/source.tar" > "$D/TOOLS.sha256"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
for mode in before after;do
 S=$R/$mode;mkdir -p "$S";tar -xf "$B/root-integrated03/source.tar" -C "$S"
 cat "$R/input/$mode.sio" "$R/input/ir_body.sio.in" > "$S/self-hosted/native/worker_proof.sio"
 cp "$R/input/driver.sio" "$S/driver.sio"
 cd "$S";export SOUNIO_STDLIB_PATH=$S/self-hosted
 find self-hosted stdlib driver.sio -type f -print0|sort -z|xargs -0 sha256sum > "$D/$mode.SOURCE.sha256"
 timeout 120 "$RAW" check driver.sio > "$D/$mode.check.stdout" 2>&1
 timeout 180 "$SEED" driver.sio "$D/$mode.elf" > "$D/$mode.compile.stdout" 2>&1
 ! grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$D/$mode.compile.stdout"
 chmod +x "$D/$mode.elf"
 rc=0;timeout 60 "$D/$mode.elf" > "$D/$mode.native.stdout" 2>&1 ||rc=$?
 if [[ $mode == before ]];then [[ "$rc" == 1 ]];grep -Fxq 'FRAME_ENVELOPE_REFUSED 141' "$D/$mode.native.stdout"
 else [[ "$rc" == 0 ]];grep -Fxq 'WORKER_CANONICAL_STUB_SCHEMA_OK param0_only=1 descriptor_mutations_refused=60 legacy_param2_refused=3' "$D/$mode.native.stdout";fi
 sha256sum -c "$D/$mode.SOURCE.sha256" > "$D/$mode.SOURCE.final.verify"
done
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
sha256sum "$D/before.elf" "$D/after.elf" > "$D/ELFS.sha256"
echo CANONICAL_PLACEHOLDER_PAIR_OK_ORIGINAL_CLI_STILL_141
