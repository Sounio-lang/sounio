#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/scalar-shadow-frontier01
mountpoint -q /orangefs/training
test ! -e "$D"
mkdir -p "$D"
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/EXPORT_STATUS";exit "$r";};trap finish EXIT
findmnt -T /orangefs/training >"$D/MOUNT.txt"
sacct -X -j19729,19731,19737 --format=JobID,State,ExitCode,Elapsed,NodeList -P >"$D/JOBS.txt"
for p in root-wide-cleanup01 root-wide-opt-route01 root-scalar-shadow01;do
 sha256sum -c "$B/$p/INPUTS.sha256" >"$D/$p.before.verify"
done
sha256sum "$B/root-bidi-inputs07/results/native/libstdlib_platform.so" >"$D/RUNTIME_AT_EXPORT.sha256"
printf '%s\n' 'Runtime DSO hash captured at export, not added retroactively to pre-execution manifests. All original before/final input guards are retained unchanged.' >"$D/RUNTIME_SCOPE"
tar -czf "$D/scalar-shadow-frontier.tar.gz" -C "$B" root-wide-cleanup01 root-wide-cleanup01.sh root-wide-cleanup01.slurm.out root-wide-opt-route01 root-wide-opt-route01.sh root-wide-opt-route01.slurm.out root-scalar-shadow01 root-scalar-shadow01-input root-scalar-shadow01.sh root-scalar-shadow01.slurm.out compiler-resumed/wide-mustdef01 root-integrated04-capture/cases/wide_projection.sio root-integrated04/results/madaros root-integrated04/tools-env-v3.sh root-integrated04/private-tools-v3 root-integrated04/private-runtime root-integrated04/pthread-installed-v3 root-bidi-inputs07/results/native/libstdlib_platform.so services/bigframe-abi07/gcc13-private/root -C /scratch sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/node
sha256sum "$D/scalar-shadow-frontier.tar.gz" >"$D/ARCHIVE.sha256"
tar -tzf "$D/scalar-shadow-frontier.tar.gz" >"$D/ARCHIVE_CONTENTS.txt"
sha256sum -c "$D/ARCHIVE.sha256" >"$D/READBACK.verify"
stat -c '%s %n' "$D/scalar-shadow-frontier.tar.gz" >"$D/SIZE.txt"
for p in root-wide-cleanup01 root-wide-opt-route01 root-scalar-shadow01;do
 sha256sum -c "$B/$p/INPUTS.sha256" >"$D/$p.final.verify"
done
echo SCALAR_SHADOW_FRONTIER_CUSTODY_OK
