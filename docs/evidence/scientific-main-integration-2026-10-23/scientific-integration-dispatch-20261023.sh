#!/usr/bin/env bash
# Packet-only transport for current-source build, library checks, real examples.
set -euo pipefail
[[ $# == 6 && "$1" == --filter && "$2" == basic_math && "$3" == --format && "$4" == junit && "$5" == --jobs && "$6" == 2 ]] || exit 78
root=$(pwd -P)
shared=${SOUNIO_ARTIFACTS:?}
local_root=/scratch/sounio-scientific-integration-20261023/run1
umask 077
mkdir -- "$local_root"
mkdir "$local_root/tmp" "$local_root/cache" "$local_root/checks" "$local_root/native"
trap 'status=$?; printf "dispatch_exit_status=%s\n" "$status" > "$local_root/DISPATCH_STATUS.txt"; cp "$local_root/DISPATCH_STATUS.txt" "$shared/"' EXIT
printf '%s\n' 'purpose=current-source-scientific-integration' 'tests_run=false' 'numerical_concordance=false' 'cache_authority=false' 'large_outputs=scratch-ext4' > "$local_root/SCOPE.txt"
cp SOURCE.txt MANIFEST.sha256 "$local_root/"
sha256sum --check --status MANIFEST.sha256
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_REQUIRE_COMMITTED_MADAROS
export SOUNIO_STDLIB_PATH="$root/stdlib"
export SOUNIO_MADAROS_CACHE="$local_root/cache"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_BUILD_LOCK="$local_root/build.lock" TMPDIR="$local_root/tmp"
ulimit -s 524288
printf 'CURRENT_SOURCE_INTEGRATION_BUILD_BEGIN\n'
/usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$local_root/BUILD_RESOURCE.txt" bash scripts/ci/build_modular_madaros.sh "$local_root/madaros.elf" > "$local_root/build.stdout" 2> "$local_root/build.stderr"
sha256sum "$local_root/madaros.elf" > "$local_root/COMPILER.sha256"
gzip -n -c "$local_root/madaros.elf" > "$local_root/madaros.elf.gz"
export MADAROS_RAW_BIN="$local_root/madaros.elf" SOUNIO_SOUC_ENGINE=madaros
printf 'kind\tsource\texit_status\n' > "$local_root/RESULTS.tsv"
sources=(
 stdlib/math/logarithmic.sio stdlib/math/probability.sio stdlib/math/channels.sio
 stdlib/genomics/normalization.sio stdlib/physics/entropy.sio stdlib/physics/thermo.sio
 stdlib/stats/regression/linear.sio stdlib/stats/shapiro_wilk.sio
 stdlib/stats/regression/nb_dispersion.sio stdlib/stats/regression/poisson.sio
 stdlib/stats/regression/nb2_diagnostics.sio stdlib/stats/regression/nb2_deviance.sio
 stdlib/io/sci_print.sio stdlib/viz/ir.sio stdlib/viz/bar.sio
 stdlib/viz/uncertainty.sio stdlib/viz/viz_canvas.sio stdlib/viz/viz_html.sio
 examples/viz_nb2_deviance.sio examples/viz_nb_dispersion_diagnostic.sio
 examples/viz_logarithmic_primitives.sio examples/viz_genomics_entropy.sio
 examples/viz_sampled_uncertainty.sio
)
for source in "${sources[@]}"; do
 name=${source//\//_}
 set +e
 "$root/bin/souc" check "$source" > "$local_root/checks/$name.log" 2>&1
 status=$?
 set -e
 printf 'typecheck\t%s\t%s\n' "$source" "$status" >> "$local_root/RESULTS.tsv"
 printf 'TYPECHECK %s %s\n' "$status" "$source"
 [[ "$status" == 0 ]] || { cp "$local_root/RESULTS.tsv" "$shared/"; exit "$status"; }
done
for demo in viz_nb2_deviance viz_nb_dispersion_diagnostic viz_logarithmic_primitives viz_genomics_entropy viz_sampled_uncertainty; do
 out="$local_root/native/$demo"
 mkdir "$out"
 "$root/bin/souc" compile "examples/$demo.sio" -o "$out/program.elf" > "$out/compile.log" 2>&1
 sha256sum "$out/program.elf" "examples/$demo.sio" > "$out/IDENTITIES.sha256"
 set +e
 LC_ALL=C /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/RESOURCE.txt" "$out/program.elf" > "$out/native.stdout" 2> "$out/native.stderr"
 status=$?
 set -e
 printf 'native\texamples/%s.sio\t%s\n' "$demo" "$status" >> "$local_root/RESULTS.tsv"
 sha256sum "$out/native.stdout" >> "$out/IDENTITIES.sha256"
 gzip -n -c "$out/program.elf" > "$out/program.elf.gz"
 printf 'NATIVE %s %s\n' "$status" "$demo"
 [[ "$status" == 0 ]] || { cp "$local_root/RESULTS.tsv" "$shared/"; exit "$status"; }
done
sha256sum --check --status MANIFEST.sha256
printf 'source_endpoints_equal=true\ncompiler_build_exit=0\ntypechecks=23\nnative_examples=5\ntests_run=false\nnumerical_concordance=false\n' > "$local_root/RESULT.txt"
cp "$local_root/RESULT.txt" "$local_root/RESULTS.tsv" "$local_root/COMPILER.sha256" "$local_root/BUILD_RESOURCE.txt" "$shared/"
printf '%s\n' "$local_root" > "$shared/LOCAL_ARTIFACT_ROOT.txt"
printf 'CURRENT_SOURCE_SCIENTIFIC_INTEGRATION_COMPLETE\n'
