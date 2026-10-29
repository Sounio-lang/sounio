#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/qpu-dyadic-export01
printf 'd97c9c37d66e52c86f6319d01d01973a483e61cda0d9eb7b7d06b2747738308b  %s\n' "$D/qpu-dyadic.tar.gz" >"$D/SHA256SUMS"
sha256sum -c "$D/SHA256SUMS" >"$D/FINAL.verify"
tar -tzf "$D/qpu-dyadic.tar.gz" >"$D/CONTENTS"
stat -c '%s %n' "$D/qpu-dyadic.tar.gz" >"$D/SIZE"
echo QPU_DYADIC_DURABLE_READBACK_OK
