#!/bin/bash
set -euo pipefail
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/pbpk-compile-frontier04
R=/scratch/sounio-stdlib-wave3-20261027/root-pbpk-frontier04-readback
mkdir "$R"
trap 'r=$?;echo "$r" >"$R/STATUS";exit "$r"' EXIT
mountpoint -q /orangefs/training
sha256sum -c "$D/ARCHIVE.sha256" >"$R/READBACK.verify"
tar -tzf "$D/pbpk-frontier.tar.gz" >"$R/ARCHIVE_CONTENTS.txt"
cmp "$D/ARCHIVE_CONTENTS.txt" "$R/ARCHIVE_CONTENTS.txt"
cp "$D/ARCHIVE.sha256" "$D/SIZE.txt" "$R/"
echo PBPK_FRONTIER_INDEPENDENT_READBACK_OK
