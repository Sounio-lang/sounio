#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-graph-slice01
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/graph-slice-export01
mkdir "$D"
G=$(command -v g++)
C=$($G -print-prog-name=cc1plus)
sha256sum "$G" "$C" >"$D/TOOLS.sha256"
"$G" --version >"$D/CXX_VERSION.txt"
"$G" -std=c++23 -O2 -Wall -Wextra -Werror "$R/input/tests/fixtures/wave4_graph_slice/reference.cpp" -o "$D/reference.rebuild"
cmp "$D/reference.rebuild" "$R/results/reference"
"$D/reference.rebuild" "$R/results/support.0.native.stdout" >"$D/rebuild-oracle.stdout"
printf 'SIZE0=0\nSIZE1=1\nSIZE4=5\n' >"$D/wrong-count.stdout"
r=0;"$D/reference.rebuild" "$D/wrong-count.stdout" >"$D/wrong-count.oracle.stdout" 2>&1||r=$?
[[ "$r" == 1 ]]
printf '%s\n' "$r" >"$D/WRONG_COUNT_REFUSED"
sha256sum -c "$R/results/INPUTS.sha256" >"$D/INPUTS.postjob.verify"
sha256sum -c "$R/results/RAW.sha256" >"$D/RAW.postjob.verify"
sha256sum -c "$R/results/SOURCE.sha256" >"$D/SOURCE.postjob.verify"
sha256sum -c "$R/results/EXECUTABLES.sha256" >"$D/EXECUTABLES.postjob.verify"
sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.final.verify"
cd "$B"
tar -czf "$D/graph-slice.tar.gz" root-graph-slice01 root-integrated02/RAW_READY root-integrated02/results/madaros
sha256sum "$D/graph-slice.tar.gz" >"$D/SHA256SUMS"
sha256sum -c "$D/SHA256SUMS" >"$D/FINAL.verify"
tar -tzf "$D/graph-slice.tar.gz" >/dev/null
stat -c '%s %n' "$D/graph-slice.tar.gz" >"$D/SIZE"
echo GRAPH_SLICE_CUSTODY_AND_ORACLE_REBUILD_OK
