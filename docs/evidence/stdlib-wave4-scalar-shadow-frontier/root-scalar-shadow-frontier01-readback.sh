#!/bin/bash
set -euo pipefail
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/scalar-shadow-frontier01
R=/scratch/sounio-stdlib-wave3-20261027/root-scalar-shadow-frontier01-readback
mkdir "$R"
trap 'r=$?;echo "$r" >"$R/STATUS";exit "$r"' EXIT
mountpoint -q /orangefs/training
sha256sum -c "$D/ARCHIVE.sha256" >"$R/READBACK.verify"
tar -tzf "$D/scalar-shadow-frontier.tar.gz" >"$R/ARCHIVE_CONTENTS.txt"
cmp "$D/ARCHIVE_CONTENTS.txt" "$R/ARCHIVE_CONTENTS.txt"
cp "$D/ARCHIVE.sha256" "$D/SIZE.txt" "$R/"
echo SCALAR_SHADOW_INDEPENDENT_ARCHIVE_READBACK_OK
