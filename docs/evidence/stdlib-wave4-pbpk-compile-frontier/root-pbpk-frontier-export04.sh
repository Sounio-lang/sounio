#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/pbpk-compile-frontier04
S=$B/root-bidi-inputs05/source
mountpoint -q /orangefs/training
test ! -e "$D"
mkdir -p "$D"
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/EXPORT_STATUS";exit "$r";};trap finish EXIT
findmnt -T /orangefs/training >"$D/MOUNT.txt"
sacct -X -j19687,19691 --format=JobID,State,ExitCode,Elapsed,NodeList -P >"$D/JOBS.txt"
for p in root-pbpk-compile04 root-pbpk-prints04;do
 (cd "$S";sha256sum -c "$B/$p/INPUTS.sha256") >"$D/$p.before.verify"
done
tar -czf "$D/pbpk-frontier.tar.gz" -C "$B" root-pbpk-compile04 root-pbpk-prints04 root-pbpk-compile04.sh root-pbpk-prints04.sh root-pbpk-compile04.slurm.out root-pbpk-prints04.slurm.out root-integrated04/results/madaros root-integrated04/tools-env-v3.sh root-integrated04/private-runtime -C "$S" stdlib tests/stdlib/chemistry/test_pbpk_clearance_ode.sio -C /scratch sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/time
sha256sum "$D/pbpk-frontier.tar.gz" >"$D/ARCHIVE.sha256"
tar -tzf "$D/pbpk-frontier.tar.gz" >"$D/ARCHIVE_CONTENTS.txt"
sha256sum -c "$D/ARCHIVE.sha256" >"$D/READBACK.verify"
stat -c '%s %n' "$D/pbpk-frontier.tar.gz" >"$D/SIZE.txt"
for p in root-pbpk-compile04 root-pbpk-prints04;do
 (cd "$S";sha256sum -c "$B/$p/INPUTS.sha256") >"$D/$p.final.verify"
done
echo PBPK_FRONTIER_CUSTODY_OK
