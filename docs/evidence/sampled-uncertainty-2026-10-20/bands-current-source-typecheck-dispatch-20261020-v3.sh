#!/usr/bin/env bash
# Package-only Forge transport dispatcher: attested compiler reuse, typecheck and figure generation; no tests.
set -euo pipefail
[[ $# == 6 && "$1" == --filter && "$2" == basic_math && "$3" == --format && "$4" == junit && "$5" == --jobs && "$6" == 2 ]] || exit 78
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
printf '%s\n' 'purpose=sampled-uncertainty-current-source-check-and-figure' 'corpus_acceptance=false' 'numerical_acceptance=false' 'figure_acceptance=false'
: "${SOUNIO_SCRATCH:?Forge scratch required}"
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_REQUIRE_COMMITTED_MADAROS
export SOUNIO_STDLIB_PATH="$root/stdlib"
export SOUNIO_MADAROS_CACHE="$SOUNIO_SCRATCH/band-check-cache"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export TMPDIR="$SOUNIO_SCRATCH/band-check-tmp"
export SOUNIO_BUILD_LOCK="$SOUNIO_SCRATCH/band-check.lock"
mkdir -p "$TMPDIR" "$SOUNIO_MADAROS_CACHE"
ulimit -s 524288
compiler="$SOUNIO_SCRATCH/madaros-band-current"
printf 'ATTESTED_CURRENT_SOURCE_COMPILER_REUSE_BEGIN\n'
cat "$root/reused-compiler/ORIGIN.txt"
gzip -dc "$root/reused-compiler/madaros-band-current.gz" > "$compiler"
chmod +x "$compiler"
expected=$(cut -d ' ' -f 1 "$root/reused-compiler/COMPILER.sha256")
[[ "$expected" =~ ^[a-f0-9]{64}$ ]]
printf '%s  %s\n' "$expected" "$compiler" | sha256sum --check
test -x "$compiler"
sha256sum "$compiler" > "$SOUNIO_SCRATCH/compiler-before.sha256"
cat "$SOUNIO_SCRATCH/compiler-before.sha256"
export MADAROS_RAW_BIN="$compiler" SOUNIO_SOUC_ENGINE=madaros
for source in stdlib/viz/uncertainty.sio stdlib/viz/epiviz.sio stdlib/viz/ir.sio stdlib/viz/viz_canvas.sio stdlib/viz/viz_html.sio stdlib/viz/workbench.sio stdlib/viz/viz_app.sio stdlib/viz/viz_replay.sio stdlib/viz/viz_window.sio examples/viz_physchem_demo/main.sio examples/viz_lab/main.sio examples/viz_lab_window/main.sio examples/pbpk_dashboard/main.sio examples/viz_hello/main.sio examples/viz_html_export/main.sio examples/viz_workbench/main.sio examples/viz_nb_dispersion_diagnostic.sio examples/viz_sampled_uncertainty.sio; do
 printf 'TYPECHECK_BEGIN %s\n' "$source"
 "$root/bin/souc" check "$source"
 printf 'TYPECHECK_OK %s\n' "$source"
done
sha256sum --check "$SOUNIO_SCRATCH/compiler-before.sha256"
printf 'CURRENT_SOURCE_BAND_TYPECHECK_READY\n'
: "${SOUNIO_ARTIFACTS:?Forge artifacts required}"
figure="$SOUNIO_ARTIFACTS/figure"
mkdir -p "$figure"
program="$SOUNIO_SCRATCH/sampled-uncertainty.elf"
printf 'NATIVE_FIGURE_COMPILE_BEGIN\n'
"$root/bin/souc" compile examples/viz_sampled_uncertainty.sio -o "$program" > "$figure/compile.log" 2>&1
sha256sum "$program" examples/viz_sampled_uncertainty.sio > "$figure/PROGRAM_IDENTITIES.sha256"
printf 'NATIVE_FIGURE_BEGIN\n'
"$program" > "$figure/sampled-uncertainty.html" 2> "$figure/native-stderr.txt"
printf 'exit_code=0\n' > "$figure/EXECUTION.txt"
sha256sum "$figure/sampled-uncertainty.html"
gzip -c "$program" > "$figure/sampled-uncertainty.elf.gz"
sha256sum "$figure/sampled-uncertainty.elf.gz"
printf 'NATIVE_FIGURE_GENERATED\n'
