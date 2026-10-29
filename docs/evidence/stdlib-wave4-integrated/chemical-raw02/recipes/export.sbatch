#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/chemical-raw02-export01
mkdir "$D"
cd "$B/compiler-error-site04/source"
sha256sum -c "$B/compiler-error-site04/results/SOURCE.sha256" >"$D/SOURCE04.postjob.verify"
sha256sum -c "$B/compiler-error-site04/results/COMPILER.sha256" >"$D/COMPILER04.postjob.verify"
cd "$B/compiler-error-site05/source"
sha256sum -c "$B/compiler-error-site05/results/SOURCE.sha256" >"$D/SOURCE05.postjob.verify"
sha256sum -c "$B/compiler-error-site05/results/COMPILER.sha256" >"$D/COMPILER05.postjob.verify"
cd "$B"
tar -czf "$D/chemical-raw02.tar.gz" compiler-error-site04 compiler-error-site05
sha256sum "$D/chemical-raw02.tar.gz" >"$D/SHA256SUMS"
sha256sum -c "$D/SHA256SUMS" >"$D/FINAL.verify"
tar -tzf "$D/chemical-raw02.tar.gz" >/dev/null
stat -c "%s %n" "$D/chemical-raw02.tar.gz" >"$D/SIZE"
echo CHEMICAL_RAW02_CUSTODY_OK
