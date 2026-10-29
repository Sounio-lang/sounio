#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/root-integrated04
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw04-export02
mountpoint -q /orangefs/training
test ! -e "$D";mkdir -p "$D"
trap 'r=$?;echo "$r" >"$D/EXPORT_STATUS";exit "$r"' EXIT
findmnt -T /orangefs/training >"$D/MOUNT.txt"
sacct -X -j19655,19658,19661,19664,19665,19666,19667,19675,19676 --format=JobID,State,ExitCode,Elapsed,NodeList -P >"$D/JOBS.txt"
(cd "$R/source";sha256sum -c "$R/results/SOURCE.sha256") >"$D/SOURCE.before.verify"
sha256sum -c "$R/results/COMPILER.sha256" >"$D/COMPILER.before.verify"
sha256sum -c "$R/preflight-v4/PRIVATE_SDK.sha256" >"$D/SDK.before.verify"
sha256sum -c "$R/CAPTURE_INPUTS.sha256" >"$D/CAPTURE.before.verify"
tar -czf "$D/raw04-sources-seed-sdk-elfs-gates-v2.tar.gz" -C "$B" root-integrated04 root-integrated04-loader root-integrated04-loader02 root-integrated04-capture services/normal-integrated01/sealed-services04 services/bigframe-abi07/gcc13-private/root -C /scratch sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/time
sha256sum "$D/raw04-sources-seed-sdk-elfs-gates-v2.tar.gz" >"$D/ARCHIVE.sha256"
tar -tzf "$D/raw04-sources-seed-sdk-elfs-gates-v2.tar.gz" >"$D/ARCHIVE_CONTENTS.txt"
sha256sum -c "$D/ARCHIVE.sha256" >"$D/READBACK.verify"
(cd "$R/source";sha256sum -c "$R/results/SOURCE.sha256") >"$D/SOURCE.final.verify"
sha256sum -c "$R/results/COMPILER.sha256" >"$D/COMPILER.final.verify"
sha256sum -c "$R/preflight-v4/PRIVATE_SDK.sha256" >"$D/SDK.final.verify"
sha256sum -c "$R/CAPTURE_INPUTS.sha256" >"$D/CAPTURE.final.verify"
echo RAW04_ARCHIVE_READBACK_OK
