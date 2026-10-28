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
cp "$ROOT/tests/stdlib/neuro/test_neuro.sio" "$B/source/test_neuro.sio"
cp "$ROOT/tests/fixtures/wave4_tensor_jordan/probe.sio.in" "$B/source/probe.sio"
cp "$ROOT/tests/reference/stdlib_tensor_jordan_reference.cpp" "$B/source/oracle.cpp"
cd "$B/source"
export SOUNIO_LOOP_REGIONS=1 SOUNIO_STDLIB_PATH="$PWD/stdlib"
find . -type f -print0|sort -z|xargs -0 sha256sum >"$B/INPUTS.sha256"
sha256sum "$MADAROS_RAW_BIN" >"$B/COMPILER.sha256"
for name in test_neuro probe;do
 timeout 300 "$MADAROS_RAW_BIN" --check "$name.sio" >"$B/$name.check" 2>&1
 timeout 300 "$MADAROS_RAW_BIN" compile "$name.sio" -o "$B/$name.elf" >"$B/$name.compile" 2>&1
 chmod +x "$B/$name.elf"
 sha256sum "$B/$name.elf" >>"$B/ELFS.sha256"
 timeout 180 "$B/$name.elf" >"$B/$name.native" 2>&1
done
grep -qx 'neuro tests passed=16 failed=0' "$B/test_neuro.native"
g++ -std=c++23 -O2 -Wall -Wextra -Werror oracle.cpp -o "$B/oracle"
sha256sum "$B/oracle" >>"$B/ELFS.sha256"
"$B/oracle" "$B/probe.native" >"$B/oracle.stdout" 2>"$B/oracle.stderr"
rc=0
"$B/oracle" "$B/probe.native" wrong >"$B/wrong.stdout" 2>"$B/wrong.stderr" || rc=$?
[[ "$rc" == 1 ]]
echo "$rc" >"$B/wrong.STATUS"
sha256sum -c "$B/INPUTS.sha256" >"$B/INPUTS.verify"
sha256sum -c "$B/COMPILER.sha256" >"$B/COMPILER.verify"
sha256sum -c "$B/ELFS.sha256" >"$B/ELFS.verify"
echo 0 >"$B/STATUS"
