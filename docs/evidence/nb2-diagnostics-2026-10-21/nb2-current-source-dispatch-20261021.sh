#!/usr/bin/env bash
# Packet-only Forge transport: typecheck and native demonstration, no tests.
set -euo pipefail
[[ $# == 6 && "$1" == --filter && "$2" == basic_math && "$3" == --format && "$4" == junit && "$5" == --jobs && "$6" == 2 ]] || exit 78
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
printf '%s\n' 'purpose=nb2-current-source-check-and-demonstration' 'corpus_acceptance=false' 'numerical_acceptance=false' 'figure_acceptance=false'
: "${SOUNIO_SCRATCH:?Forge scratch required}"
: "${SOUNIO_ARTIFACTS:?Forge artifacts required}"
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_REQUIRE_COMMITTED_MADAROS
export SOUNIO_STDLIB_PATH="$root/stdlib"
export SOUNIO_MADAROS_CACHE="$SOUNIO_SCRATCH/nb2-check-cache"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export TMPDIR="$SOUNIO_SCRATCH/nb2-check-tmp"
export SOUNIO_BUILD_LOCK="$SOUNIO_SCRATCH/nb2-check.lock"
mkdir -p "$TMPDIR" "$SOUNIO_MADAROS_CACHE" "$SOUNIO_ARTIFACTS/compiler" "$SOUNIO_ARTIFACTS/figure"
ulimit -s 524288
compiler="$SOUNIO_SCRATCH/madaros-nb2-current"
printf 'CURRENT_SOURCE_BUILD_BEGIN\n'
bash scripts/ci/build_modular_madaros.sh "$compiler"
test -x "$compiler"
sha256sum "$compiler" > "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
cat "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
# Preserve before consumers execute so a failed demo cannot lose its compiler.
gzip -c "$compiler" > "$SOUNIO_ARTIFACTS/compiler/madaros-nb2-current.gz"
sha256sum "$SOUNIO_ARTIFACTS/compiler/madaros-nb2-current.gz"
export MADAROS_RAW_BIN="$compiler" SOUNIO_SOUC_ENGINE=madaros
for source in stdlib/stats/regression/nb2_diagnostics.sio stdlib/stats/regression/nb_dispersion.sio stdlib/stats/regression/poisson.sio examples/viz_nb_dispersion_diagnostic.sio; do
 printf 'TYPECHECK_BEGIN %s\n' "$source"
 "$root/bin/souc" check "$source"
 printf 'TYPECHECK_OK %s\n' "$source"
done
sha256sum --check "$SOUNIO_ARTIFACTS/compiler/COMPILER.sha256"
printf 'CURRENT_SOURCE_NB2_TYPECHECK_READY\n'
figure="$SOUNIO_ARTIFACTS/figure"
program="$SOUNIO_SCRATCH/nb2-diagnostic.elf"
printf 'NATIVE_NB2_COMPILE_BEGIN\n'
"$root/bin/souc" compile examples/viz_nb_dispersion_diagnostic.sio -o "$program" > "$figure/compile.log" 2>&1
sha256sum "$program" examples/viz_nb_dispersion_diagnostic.sio stdlib/stats/regression/nb2_diagnostics.sio > "$figure/PROGRAM_IDENTITIES.sha256"
gzip -c "$program" > "$figure/nb2-diagnostic.elf.gz"
sha256sum "$figure/nb2-diagnostic.elf.gz"
printf 'NATIVE_NB2_BEGIN\n'
set +e
"$program" > "$figure/nb2-diagnostic.stdout" 2> "$figure/native-stderr.txt"
program_status=$?
set -e
printf 'exit_code=%s\n' "$program_status" > "$figure/EXECUTION.txt"
sha256sum "$figure/nb2-diagnostic.stdout"
[[ "$program_status" == 0 ]] || exit "$program_status"
printf 'NATIVE_NB2_GENERATED\n'
