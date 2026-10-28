#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:/bin
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] || exit 64
: "${SEED_RAW_BIN:?modern raw compiler ELF required}"
[[ "$SEED_RAW_BIN" == /* && -x "$SEED_RAW_BIN" ]] || exit 64
OUT=$1
mkdir -p "$OUT/input/compiler" "$OUT/input/io"
cp "$ROOT/self-hosted/compiler/trusted_origins.sio" "$OUT/input/compiler/"
cp "$ROOT/self-hosted/io/env.sio" "$OUT/input/io/"
cp "$ROOT/tests/fixtures/wave4_trusted_origins/positive.sio.in" "$OUT/input/positive.sio"
cp "$ROOT/tests/fixtures/wave4_trusted_origins/negative.sio.in" "$OUT/input/negative.sio"
cp "$ROOT/tests/fixtures/wave4_trusted_origins/stat64_layout.cpp" "$OUT/input/"
cd "$OUT/input"
mkdir -p fixtures/{ffi,mem} spoof/{ffi,mem}
printf 'worker authority\n' >fixtures/ffi/worker.sio
printf 'storage authority\n' >fixtures/mem/typed_storage.sio
cp fixtures/ffi/worker.sio spoof/ffi/worker.sio
cp fixtures/mem/typed_storage.sio spoof/mem/typed_storage.sio
ln -s fixtures/ffi/worker.sio worker-link.sio
ln fixtures/mem/typed_storage.sio storage-hardlink.sio
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >"$OUT/INPUTS.sha256"
sha256sum "$SEED_RAW_BIN" >"$OUT/SEED.sha256"
ulimit -s 524288
ulimit -c 0
export SOUNIO_STDLIB_PATH="$PWD"
for name in positive negative;do
 timeout 300 "$SEED_RAW_BIN" --check "$name.sio" >"$OUT/$name.check" 2>&1
 printf '%s_check\t0\n' "$name" >>"$OUT/RESULTS.tsv"
 timeout 300 "$SEED_RAW_BIN" compile "$name.sio" -o "$OUT/$name.elf" >"$OUT/$name.compile" 2>&1
 printf '%s_compile\t0\n' "$name" >>"$OUT/RESULTS.tsv"
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >>"$OUT/ELFS.sha256"
done
SOUNIO_STDLIB_PATH="$PWD/fixtures" timeout 30 "$OUT/positive.elf" >"$OUT/positive.native"
grep -qx 'trusted origins native: OK' "$OUT/positive.native"
printf 'positive_native\t0\n' >>"$OUT/RESULTS.tsv"
SOUNIO_STDLIB_PATH="$PWD/missing-authority" timeout 30 "$OUT/negative.elf" >"$OUT/negative.native"
grep -qx 'trusted origins fail closed: OK' "$OUT/negative.native"
printf 'negative_native\t0\n' >>"$OUT/RESULTS.tsv"
"${CXX:-g++}" -std=c++23 -Wall -Wextra -Werror stat64_layout.cpp -o "$OUT/layout"
"$OUT/layout" fixtures/ffi/worker.sio worker-link.sio >"$OUT/layout.native"
"$OUT/layout" fixtures/mem/typed_storage.sio storage-hardlink.sio >>"$OUT/layout.native"
printf 'cpp23_layout_identity\t0\n' >>"$OUT/RESULTS.tsv"
sha256sum -c "$OUT/INPUTS.sha256" >"$OUT/INPUTS.verify"
sha256sum -c "$OUT/SEED.sha256" >"$OUT/SEED.verify"
sha256sum -c "$OUT/ELFS.sha256" >"$OUT/ELFS.verify"
printf '0\n' >"$OUT/STATUS"
