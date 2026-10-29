#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/pbpk-binding-stamp06
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
mkdir "$D/input-repo"
cp -a "$R/source/self-hosted" "$D/input-repo/self-hosted"
tar -xzf "$D/overlay.tar.gz" -C "$D/input-repo"
export MADAROS_RAW_BIN="$R/results/madaros"
export LEAN_SINGLE_BIN="$R/source/bin/souc-lean-single-x86_64"
export NODE_BIN="$P/tools/node"
"$D/input-repo/tests/fixtures/wave4_pbpk_tuple_print/binding_stamp_gate.sh" "$D/gate" > "$D/gate.stdout" 2> "$D/gate.stderr"
