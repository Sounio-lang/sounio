#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/integrated02-export01
mkdir "$D"
cd "$B/stiff-api03/source"
sha256sum -c "$B/stiff-api03/results/ELFS.sha256" >"$B/stiff-api03/results/ELFS.postjob.verify"
sha256sum -c "$B/stiff-api03/results/GENERATED.sha256" >"$B/stiff-api03/results/GENERATED.postjob.verify"
cd "$B"
tar -czf "$D/integrated02.tar.gz" root-integrated02 root-integrated02-loader compiler-error-site03 stiff-api01 stiff-api02 stiff-api03 stiff-contract01 stiff-contract02
sha256sum "$D/integrated02.tar.gz" >"$D/SHA256SUMS"
sha256sum -c "$D/SHA256SUMS" >"$D/FINAL.verify"
tar -tzf "$D/integrated02.tar.gz" >/dev/null
stat -c '%s %n' "$D/integrated02.tar.gz" >"$D/SIZE"
echo INTEGRATED02_CUSTODY_OK
