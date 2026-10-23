#!/usr/bin/env bash
# Package-only Forge transport dispatcher: compiler build and typecheck, no tests.
set -euo pipefail
[[ $# == 6 && "$1" == --filter && "$2" == basic_math && "$3" == --format && "$4" == junit && "$5" == --jobs && "$6" == 2 ]] || exit 78
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
printf '%s\n' 'purpose=current-source-typecheck-only' 'corpus_acceptance=false' 'numerical_acceptance=false' 'figure_acceptance=false'
: "${SOUNIO_SCRATCH:?Forge scratch required}"
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_REQUIRE_COMMITTED_MADAROS
export SOUNIO_STDLIB_PATH="$root/stdlib"
export SOUNIO_MADAROS_CACHE="$SOUNIO_SCRATCH/nb-check-cache"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export TMPDIR="$SOUNIO_SCRATCH/nb-check-tmp"
export SOUNIO_BUILD_LOCK="$SOUNIO_SCRATCH/nb-check.lock"
mkdir -p "$TMPDIR" "$SOUNIO_MADAROS_CACHE"
ulimit -s 524288
compiler="$SOUNIO_SCRATCH/madaros-nb-current"
printf 'CURRENT_SOURCE_BUILD_BEGIN\n'
bash scripts/ci/build_modular_madaros.sh "$compiler"
test -x "$compiler"
sha256sum "$compiler" > "$SOUNIO_SCRATCH/compiler-before.sha256"
cat "$SOUNIO_SCRATCH/compiler-before.sha256"
export MADAROS_RAW_BIN="$compiler" SOUNIO_SOUC_ENGINE=madaros
for source in stdlib/stats/regression/nb_dispersion.sio examples/viz_nb_dispersion_diagnostic.sio; do
 printf 'TYPECHECK_BEGIN %s\n' "$source"
 "$root/bin/souc" check "$source"
 printf 'TYPECHECK_OK %s\n' "$source"
done
sha256sum --check "$SOUNIO_SCRATCH/compiler-before.sha256"
printf 'CURRENT_SOURCE_TYPECHECK_READY\n'
