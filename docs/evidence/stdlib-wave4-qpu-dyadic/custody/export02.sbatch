#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-qpu-dyadic01
D=$R/export02
mkdir "$D"
cd "$R/library"
sha256sum -c "$R/imported-results/SOURCE.sha256" >"$D/SOURCE.postjob.verify"
sha256sum -c "$R/imported-results/EXECUTABLES.sha256" >"$D/IMPORTED_ELFS.postjob.verify"
sha256sum -c "$R/gate/EXECUTABLES.sha256" >"$D/COMPONENT_ELFS.postjob.verify"
sha256sum -c "$R/receipts/SDK.sha256" >"$D/SDK.postjob.verify"
sha256sum -c "$R/receipts/RAW.sha256" >"$D/RAW.postjob.verify"
cd "$B"
tar -czf "$D/qpu-dyadic.tar.gz" root-qpu-dyadic01 root-integrated02/RAW_READY root-integrated02/results/madaros services/bigframe-abi07/gcc13-private/root services-tools-recovered/tools/node services/wide-counts33/input/dso
sha256sum "$D/qpu-dyadic.tar.gz" >"$D/SHA256SUMS"
sha256sum -c "$D/SHA256SUMS" >"$D/FINAL.verify"
tar -tzf "$D/qpu-dyadic.tar.gz" >/dev/null
stat -c '%s %n' "$D/qpu-dyadic.tar.gz" >"$D/SIZE"
echo QPU_DYADIC_EXPORT_OK
