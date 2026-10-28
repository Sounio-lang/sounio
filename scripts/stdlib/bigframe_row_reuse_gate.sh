#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -s 524288
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] || exit 64
: "${MADAROS_RAW_BIN:?rebuilt raw compiler required}"
B=$1
mkdir -p "$B/source"
cp -a "$ROOT/stdlib" "$B/source/"
cp "$ROOT/tests/fixtures/wave4_bigframe_rows/probe.sio.in" "$B/source/probe.sio"
cd "$B/source"
export SOUNIO_LOOP_REGIONS=0 SOUNIO_STDLIB_PATH="$PWD/stdlib"
find . -type f -print0|sort -z|xargs -0 sha256sum >"$B/INPUTS.sha256"
sha256sum "$MADAROS_RAW_BIN" >"$B/COMPILER.sha256"
timeout 300 "$MADAROS_RAW_BIN" --check probe.sio >"$B/probe.check" 2>&1
timeout 300 "$MADAROS_RAW_BIN" compile probe.sio -o "$B/probe.elf" >"$B/probe.compile" 2>&1
chmod +x "$B/probe.elf"
sha256sum "$B/probe.elf" >"$B/ELFS.sha256"
timeout 180 "$B/probe.elf" >"$B/probe.native" 2>&1
grep -qx BIGFRAME_ROW_REUSE_OK "$B/probe.native"
sha256sum -c "$B/INPUTS.sha256" >"$B/INPUTS.verify"
sha256sum -c "$B/COMPILER.sha256" >"$B/COMPILER.verify"
sha256sum -c "$B/ELFS.sha256" >"$B/ELFS.verify"
echo 0 >"$B/STATUS"
