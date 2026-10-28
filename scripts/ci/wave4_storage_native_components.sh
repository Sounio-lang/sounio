#!/usr/bin/env bash
# Compiler-component gate. This does not qualify safe HeapVec APIs or whole stdlib.
set -euo pipefail
export PATH=/usr/bin:/bin
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] || exit 64
: "${SEED_RAW_BIN:?legacy seed ELF required}"
: "${CHECK_RAW_BIN:?modern raw checker ELF required}"
[[ "$SEED_RAW_BIN" == /* && -x "$SEED_RAW_BIN" && "$CHECK_RAW_BIN" == /* && -x "$CHECK_RAW_BIN" ]] || exit 64
OUT=$1
mkdir -p "$OUT/source"
cp -a "$ROOT/self-hosted" "$ROOT/stdlib" "$OUT/source/"
cat "$ROOT/tests/fixtures/wave4_storage_native/emitter_probe.sio.in" >>"$OUT/source/self-hosted/native/codegen_x86_linux.sio"
cp "$ROOT/tests/fixtures/wave4_storage_native/emitter_driver.sio.in" "$OUT/source/emitter.sio"
cp "$ROOT/tests/fixtures/wave4_storage_native/registration.sio.in" "$OUT/source/registration.sio"
cp "$ROOT/tests/fixtures/wave4_storage_native/storage_barrier_oracle.cpp" "$OUT/source/"
cd "$OUT/source"
export SOUNIO_STDLIB_PATH="$PWD/self-hosted"
ulimit -s 524288
ulimit -c 0
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >"$OUT/INPUTS.sha256"
sha256sum "$SEED_RAW_BIN" "$CHECK_RAW_BIN" >"$OUT/COMPILERS.sha256"
for k in emitter registration;do
 timeout 1200 "$CHECK_RAW_BIN" --check "$k.sio" >"$OUT/$k.check" 2>&1
 printf '%s_check\t0\n' "$k" >>"$OUT/RESULTS.tsv"
 timeout 900 "$SEED_RAW_BIN" "$k.sio" "$OUT/$k.elf" >"$OUT/$k.compile" 2>&1
 printf '%s_compile\t0\n' "$k" >>"$OUT/RESULTS.tsv"
 chmod +x "$OUT/$k.elf"
 sha256sum "$OUT/$k.elf" >>"$OUT/ELFS.sha256"
 timeout 60 "$OUT/$k.elf" >"$OUT/$k.native"
 printf '%s_native\t0\n' "$k" >>"$OUT/RESULTS.tsv"
done
grep -qx 'storage registration and float call metadata: OK' "$OUT/registration.native"
"${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror storage_barrier_oracle.cpp -o "$OUT/oracle"
"$OUT/oracle" "$OUT/emitter.native" >"$OUT/oracle.native"
printf 'cpp23_transport_barrier\t0\n' >>"$OUT/RESULTS.tsv"
sha256sum -c "$OUT/INPUTS.sha256" >"$OUT/INPUTS.verify"
sha256sum -c "$OUT/COMPILERS.sha256" >"$OUT/COMPILERS.verify"
sha256sum -c "$OUT/ELFS.sha256" >"$OUT/ELFS.verify"
printf '0\n' >"$OUT/STATUS"
