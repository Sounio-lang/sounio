#!/bin/bash
set -uo pipefail
root=/scratch/sounio-stdlib-deep-20261023/final1
cd "$root/source"
export TMPDIR="$root/tmp" SOUNIO_BUILD_LOCK="$root/build.lock" SOUNIO_MADAROS_CACHE="$root/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_STDLIB_PATH="$root/source/stdlib" SOUNIO_SOUC_ENGINE=madaros
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-deep-20261023/compiler_guarded/artifacts/madaros.elf
export STDLIB_DEEP_OUT="$root/artifacts"
ulimit -s 524288
bash scripts/stdlib/deep_science_gate.sh
exit $?
