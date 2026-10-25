#!/usr/bin/env bash
# Actual production science -> Canvas/SVG. C++23 references; no external redraw.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
[[ $(uname -s) == Linux && $(id -u) == 1000 ]] || { echo 'Linux UID1000 worker required' >&2;exit 2; }
: "${MADAROS_RAW_BIN:?verified absolute source-built compiler required}"
: "${SCIENCE_NATIVE_OUTPUT:?new absolute output directory required}"
out=$SCIENCE_NATIVE_OUTPUT
[[ $MADAROS_RAW_BIN == /* && -x $MADAROS_RAW_BIN && $out == /* && ! -e $out && ! -L $out ]] || exit 2
mkdir -p "$out"
trap 'rc=$?;printf "%s\n" "$rc" > "$out/STATUS"' EXIT
ulimit -s 524288;ulimit -c 0
unset SOUNIO_SOUC_BIN SOUC_NATIVE_BIN
export SOUC_BIN="$MADAROS_RAW_BIN" SOUNIO_SOUC_ENGINE=madaros
export LC_ALL=C LANG=C SOUNIO_STDLIB_PATH="$repo/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_MADAROS_CACHE="$out/cache" SOUNIO_BUILD_LOCK="$out/build.lock"
engines=(madaros)
if [[ -n ${SCIENCE_NATIVE_SEED_BIN:-} ]];then
 [[ $SCIENCE_NATIVE_SEED_BIN == /* && -x $SCIENCE_NATIVE_SEED_BIN ]] || exit 2
 engines+=(seed)
fi
printf '%s\n' "${engines[@]}" > "$out/ENGINES.txt"
printf 'engine\tcase\tphase\texit\n' > "$out/RESULTS.tsv"
phase(){ local engine=$1 name=$2 label=$3;shift 3;local rc=0;"$@" > "$out/$engine.$name.$label.stdout" 2> "$out/$engine.$name.$label.stderr" || rc=$?;printf '%s\t%s\t%s\t%s\n' "$engine" "$name" "$label" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
find stdlib examples/science_native scripts/lib -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum bin/souc bin/madaros scripts/ci/souc-native-wrapper.sh scripts/stdlib/scientific_native_workflows_gate.sh tests/reference/scientific_native_workflows.cpp tests/reference/scientific_canvas_export.cpp tests/reference/canonical_palette_reference.cpp tests/reference/data/font8x8_basic.h tests/reference/data/canonical_palettes/BIDS-colormaps.py.txt >> "$out/INPUTS.sha256"
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILERS.sha256"
if [[ -n ${SCIENCE_NATIVE_SEED_BIN:-} ]];then sha256sum "$SCIENCE_NATIVE_SEED_BIN" >> "$out/COMPILERS.sha256";fi
cat > "$out/PRIMARY_DATA.sha256" <<'DATA'
49d8df366296b203ca3211bc0672cf2a762135bf12710735b6292756b19dffd5  tests/reference/data/font8x8_basic.h
b7478d10e2ccfdb22128867e137f16f55b687e9c225065e91bcbd6bb9e6c0352  tests/reference/data/canonical_palettes/BIDS-colormaps.py.txt
DATA
phase primary assets hash sha256sum --check "$out/PRIMARY_DATA.sha256"
phase cpp reference compile g++ -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror tests/reference/scientific_native_workflows.cpp -o "$out/reference"
# The existing exporter chooses zlib or its bounded stored-deflate fallback.
zlib_flags=()
if printf '#include <zlib.h>\n' | g++ -x c++ -E - >/dev/null 2>&1;then zlib_flags=(-lz);fi
phase cpp canvas-export compile g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/scientific_canvas_export.cpp "${zlib_flags[@]}" -o "$out/canvas-export"
primary=tests/reference/data/canonical_palettes/BIDS-colormaps.py.txt
for engine in "${engines[@]}";do
 for spec in 'poisson_field poisson SCIENTIFIC_POISSON_COMPLETE' 'normalization normalization SCIENTIFIC_NORMALIZATION_COMPLETE' 'survival survival SCIENTIFIC_SURVIVAL_COMPLETE' 'dp54_event dp54 SCIENTIFIC_DP54_COMPLETE';do
  read -r name mode marker <<< "$spec"
  source=examples/science_native/$name.sio
  if [[ $engine == madaros ]];then
   phase "$engine" "$name" check timeout 240 bin/souc check "$source"
   phase "$engine" "$name" compile timeout 300 bin/souc compile "$source" -o "$out/$engine.$name.elf"
  else
   export SOUC_NATIVE_BIN=$SCIENCE_NATIVE_SEED_BIN
   phase "$engine" "$name" check timeout 240 bash scripts/ci/souc-native-wrapper.sh check "$source"
   phase "$engine" "$name" compile timeout 300 "$SCIENCE_NATIVE_SEED_BIN" "$source" "$out/$engine.$name.elf"
  fi
  ! grep -Eiq 'error\[E[0-9]+\]|typecheck:.*failed|parse error' "$out/$engine.$name."{check,compile}.{stdout,stderr}
  [[ -s "$out/$engine.$name.elf" ]];chmod +x "$out/$engine.$name.elf"
  sha256sum "$out/$engine.$name.elf" > "$out/$engine.$name.ELF.sha256"
  phase "$engine" "$name" native /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$engine.$name.resource" timeout 180 "$out/$engine.$name.elf"
  grep -Fxq "<!-- $marker -->" "$out/$engine.$name.native.stdout"
  phase "$engine" "$name" concordance timeout 180 "$out/reference" "--$mode" "$primary" "$out/$engine.$name.native.stdout" "$out/$engine.$name.actual-canvas.ppm"
  phase "$engine" "$name" canvas-export "$out/canvas-export" "$out/$engine.$name.actual-canvas.ppm.rle" "$out/$engine.$name.exported.ppm" "$out/$engine.$name.actual-canvas.png"
  phase "$engine" "$name" raster-roundtrip cmp "$out/$engine.$name.actual-canvas.ppm" "$out/$engine.$name.exported.ppm"
  sed -n '1,/<\/svg>/p' "$out/$engine.$name.native.stdout" > "$out/$engine.$name.actual.svg"
  [[ -s "$out/$engine.$name.actual.svg" ]]
 done
 for spec in 'poisson_field poisson node manufactured_PDE_analytic_solution' 'poisson_field poisson segment SVG_consumes_same_prepared_snapshot' 'poisson_field poisson pixel actual_Canvas_heatmap_RGB/cell_geometry' 'normalization normalization factor independent_median-ratio_size_factor' 'survival survival ci-kind confidence_kind_and_level,_not_prediction' 'dp54_event dp54 dense independent_exponential_dense_value/derivative' 'normalization normalization caption-column normalization_caption_exceeds_its_panel_column' 'normalization normalization label-bounds actual_text_footprint/bounds' 'poisson_field poisson svg-color Poisson_SVG_heatmap_color' 'poisson_field poisson color-domain true_color-domain_metadata_including_tiny_nonzero_errors';do
  read -r name mode mutation expected <<< "$spec"
  input="$out/$engine.$name.native.stdout";mutant="$out/$engine.$name.mutant-$mutation.input"
  phase "$engine" "$name" "mutation-$mutation" "$out/reference" "--mutate-$mutation" "$input" "$mutant"
  [[ -s "$mutant" ]] && ! cmp -s "$input" "$mutant"
  rc=0
  phase "$engine" "$name" "rejection-$mutation" timeout 180 "$out/reference" "--$mode" "$primary" "$mutant" || rc=$?
  [[ $rc == 1 ]]
  grep -Fq "${expected//_/ }" "$out/$engine.$name.rejection-$mutation.stderr"
 done
done
phase integrity inputs unchanged sha256sum --check "$out/INPUTS.sha256"
phase integrity compilers unchanged sha256sum --check "$out/COMPILERS.sha256"
printf '%s\n' SCIENTIFIC_NATIVE_WORKFLOWS_GATE_PASS | tee "$out/FINAL.txt"
