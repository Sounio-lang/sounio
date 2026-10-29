#!/bin/bash
#SBATCH --job-name=stdlib-public-pools-normal-raw04
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/public-pipeline57/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/public-pipeline57/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/services/public-pipeline57
S=$D/input
ORIGINAL=$B/services/normal-integrated01/sealed-services04
mkdir "$D/receipts"
finish(){ r=$?;trap - EXIT;for m in ORIGINAL INPUT TOOLS SDK;do if test -s "$D/receipts/$m.sha256";then sha256sum -c "$D/receipts/$m.sha256" >"$D/receipts/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
(cd "$ORIGINAL";sha256sum -c SEALED_SOURCE.sha256) >"$D/receipts/SEALED.before.verify"
sed "s@  @  $ORIGINAL/@" "$ORIGINAL/SEALED_SOURCE.sha256" >"$D/receipts/ORIGINAL.sha256"
mkdir "$D/receipts/preimage"
cp "$S/scripts/stdlib/wave3_native_pool_gate.sh" "$S/scripts/stdlib/wave4_normal_pipeline_gate.sh" "$D/receipts/preimage/"
cp "$D/overlay/scripts/stdlib/wave3_native_pool_gate.sh" "$D/overlay/scripts/stdlib/wave4_normal_pipeline_gate.sh" "$S/scripts/stdlib/"
find "$S" "$D/overlay" "$D/receipts/preimage" -type f -print0|sort -z|xargs -0 sha256sum >"$D/receipts/INPUT.sha256"
export MADAROS_RAW_BIN=$R/results/madaros
export ARROW_CPP_ROOT=$B/sdk/arrow26/root/usr
export TIME_BIN=$P/tools/time
export SOUNIO_LOOP_REGIONS=1 SOUNIO_LOOP_CHECKPOINT_ELISION=1
unset NORMAL_ORACLE_ONLY NORMAL_ABORT_ONLY
(find "$R/private-tools-v3" "$R/pthread-installed-v3" "$R/private-runtime" "$G" -type f -print0;printf '%s\0' "$R/tools-env-v3.sh" "$MADAROS_RAW_BIN" "$TIME_BIN" "$0")|sort -zu|xargs -0 sha256sum >"$D/receipts/TOOLS.sha256"
find "$ARROW_CPP_ROOT" -type f -print0|sort -z|xargs -0 sha256sum >"$D/receipts/SDK.sha256"
printf 'a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333  %s\n' "$MADAROS_RAW_BIN" >"$D/receipts/RAW.sha256"
sha256sum -c "$D/receipts/RAW.sha256" >"$D/receipts/RAW.verify"
printf 'gate\tstatus\n' >"$D/PHASES.tsv"
bad=0
for gate in pool normal;do
 script=$S/scripts/stdlib/wave3_native_pool_gate.sh
 if test "$gate" = normal;then script=$S/scripts/stdlib/wave4_normal_pipeline_gate.sh;fi
 rc=0
 bash "$script" "$D/$gate-results" >"$D/$gate.stdout" 2>"$D/$gate.stderr"||rc=$?
 printf '%s\t%s\n' "$gate" "$rc" >>"$D/PHASES.tsv"
 if test "$rc" != 0;then bad=1;fi
done
exit "$bad"
