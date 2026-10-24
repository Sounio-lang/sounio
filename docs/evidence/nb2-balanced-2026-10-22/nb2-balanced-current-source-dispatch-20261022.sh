#!/usr/bin/env bash
# Packet-only Forge transport: build/check and two real demonstrations; no tests.
set -euo pipefail
[[ $# == 6 && "$1" == --filter && "$2" == basic_math && "$3" == --format && "$4" == junit && "$5" == --jobs && "$6" == 2 ]] || exit 78
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
printf '%s\n' 'purpose=shared-nb2-balanced-current-source-check-and-demonstrations' 'corpus_acceptance=false' 'numerical_acceptance=false' 'comparative_benchmark=false'
: "${SOUNIO_SCRATCH:?Forge scratch required}"
: "${SOUNIO_ARTIFACTS:?Forge artifacts required}"
test -x /usr/bin/time
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_REQUIRE_COMMITTED_MADAROS
export SOUNIO_STDLIB_PATH="$root/stdlib"
export SOUNIO_MADAROS_CACHE="$SOUNIO_SCRATCH/nb2-balanced-check-cache"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export TMPDIR="$SOUNIO_SCRATCH/nb2-balanced-check-tmp"
export SOUNIO_BUILD_LOCK="$SOUNIO_SCRATCH/nb2-balanced-check.lock"
mkdir -p "$TMPDIR" "$SOUNIO_MADAROS_CACHE" "$SOUNIO_ARTIFACTS/compiler"
ulimit -s 524288
compiler="$SOUNIO_SCRATCH/madaros-nb2-balanced-current"
printf 'CURRENT_SOURCE_BUILD_BEGIN\n'
bash scripts/ci/build_modular_madaros.sh "$compiler"
test -x "$compiler"
sha256sum "$compiler" > "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
cat "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
gzip -c "$compiler" > "$SOUNIO_ARTIFACTS/compiler/madaros-nb2-balanced-current.gz"
sha256sum "$SOUNIO_ARTIFACTS/compiler/madaros-nb2-balanced-current.gz"
export MADAROS_RAW_BIN="$compiler" SOUNIO_SOUC_ENGINE=madaros
for source in stdlib/io/sci_print.sio stdlib/math/logarithmic.sio stdlib/stats/regression/nb2_diagnostics.sio stdlib/stats/regression/nb2_deviance.sio examples/viz_nb2_deviance.sio; do
 printf 'TYPECHECK_BEGIN %s\n' "$source"
 "$root/bin/souc" check "$source"
 printf 'TYPECHECK_OK %s\n' "$source"
done
sha256sum --check "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
printf 'CURRENT_SOURCE_NB2_BALANCED_TYPECHECK_READY\n'
for demo in viz_nb2_deviance; do
 source="examples/$demo.sio"
 out="$SOUNIO_ARTIFACTS/$demo"
 mkdir -p "$out"
 program="$SOUNIO_SCRATCH/$demo.elf"
 printf 'NATIVE_COMPILE_BEGIN %s\n' "$demo"
 "$root/bin/souc" compile "$source" -o "$program" > "$out/compile.log" 2>&1
 sha256sum "$program" "$source" stdlib/math/logarithmic.sio stdlib/stats/regression/nb2_deviance.sio > "$out/PROGRAM_IDENTITIES.sha256"
 gzip -c "$program" > "$out/program.elf.gz"
 LC_ALL=C /usr/bin/time --version > "$out/TIME_VERSION.txt"
 printf 'NATIVE_BEGIN %s\n' "$demo"
 set +e
 LC_ALL=C /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/RESOURCE.txt" "$program" > "$out/native.stdout" 2> "$out/native-stderr.txt"
 program_status=$?
 set -e
 printf 'exit_code=%s\n' "$program_status" > "$out/EXECUTION.txt"
 cat "$out/RESOURCE.txt"
 sha256sum "$out/native.stdout"
 [[ "$program_status" == 0 ]] || exit "$program_status"
 printf 'NATIVE_DEMONSTRATION_COMPLETE %s\n' "$demo"
done
sha256sum --check "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
printf 'CURRENT_SOURCE_NB2_BALANCED_DEMONSTRATIONS_READY\n'
