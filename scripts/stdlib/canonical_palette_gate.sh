#!/usr/bin/env bash
# Native-only focused acceptance. Primary .py.txt files are read as data only.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?usage: canonical_palette_gate.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e $out ]] || { echo 'output exists' >&2;exit 64; }
mkdir -p "$out";out=$(cd "$out" && pwd)
cd "$repo"
: "${MADAROS_RAW_BIN:?explicit pinned source-built compiler required}"
[[ -x $MADAROS_RAW_BIN ]] || exit 64
export MADAROS_RAW_BIN
source scripts/lib/resolve_souc.sh;sounio_require_souc
export SOUNIO_STDLIB_PATH="$repo/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$out/cache" TMPDIR="$out/tmp"
mkdir -p "$TMPDIR" "$SOUNIO_MADAROS_CACHE"
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
sha256sum "$SOUC_BIN" > "$out/LAUNCHER.sha256"
find stdlib tests/reference/data/canonical_palettes tests/stdlib/canonical_palettes -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum scripts/stdlib/canonical_palette_gate.sh tests/run-pass/viz_canonical_palettes.sio tests/run-pass/stdlib_deep_viz_dynamic_fields.sio tests/reference/canonical_palette_reference.cpp tests/reference/deep_viz_dynamic_fields.cpp tests/reference/data/font8x8_basic.h tests/reference/scientific_canvas_export.cpp >> "$out/INPUTS.sha256"
printf 'phase\tsource\texit_status\n' > "$out/RESULTS.tsv"
phase(){ local label=$1;shift;local rc=0;"$@" > "$out/$label.stdout" 2> "$out/$label.stderr" || rc=$?;printf '%s\t%s\t%s\n' "$label" "$*" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
finish(){ local rc=$?;trap - EXIT;printf '%s\n' "$rc" > "$out/STATUS";exit "$rc"; };trap finish EXIT
primary=tests/reference/data/canonical_palettes
cat > "$out/PINNED_DATA.sha256" <<'HASHES'
b7478d10e2ccfdb22128867e137f16f55b687e9c225065e91bcbd6bb9e6c0352  tests/reference/data/canonical_palettes/BIDS-colormaps.py.txt
30419703377136667f2c8a426d396874a8dadf53699d277fc8ad759d05092799  tests/reference/data/canonical_palettes/BIDS-LICENSE.txt
5a1a81ea301728c8bba2933da832c0cd62229daf20893a024ab3d53244468dbc  tests/reference/data/canonical_palettes/Matplotlib-LICENSE.txt
679ea1847a643bc691970022f1c7945bff5fb57e7adb67fbd3f46d8efeba3d5b  tests/reference/data/canonical_palettes/Matplotlib-cm-listed.py.txt.gz
b163449e143d153539243b3932685764e5acad4e7de0fbf53092eaaca0da9ae6  tests/stdlib/canonical_palettes/legacy-approx-baseline.stdout.gz
HASHES
phase pinned-data sha256sum -c "$out/PINNED_DATA.sha256"
gzip -dc "$primary/Matplotlib-cm-listed.py.txt.gz" > "$out/Matplotlib-cm-listed.py.txt"
printf '%s\n' 'ddad3698f5129ceb1792a445371286c08bc9080298e657b3054aea19c9659ef9  Matplotlib-cm-listed.py.txt' > "$out/DECOMPRESSED.sha256"
phase decompressed bash -c 'cd "$1" && sha256sum -c DECOMPRESSED.sha256' _ "$out"
gzip -dc tests/stdlib/canonical_palettes/legacy-approx-baseline.stdout.gz > "$out/historical-approx.stdout"
phase reference.build g++ -std=c++23 -O2 -Wall -Wextra tests/reference/canonical_palette_reference.cpp -o "$out/reference"
phase primary-concordance "$out/reference" --sources "$primary/BIDS-colormaps.py.txt" "$out/Matplotlib-cm-listed.py.txt"
"$out/reference" --emit-lut "$primary/BIDS-colormaps.py.txt" > "$out/generated-data.sio"
phase generated-data cmp "$out/generated-data.sio" stdlib/viz/colormap_data.sio
for lib in colormap_data palette colormap colorbar dynamic_fields epiviz field_canvas viz_canvas viz_html;do phase "$lib.check" "$SOUC_BIN" check "stdlib/viz/$lib.sio";done
for src in tests/stdlib/canonical_palettes/legacy_palette_witness.sio tests/stdlib/canonical_palettes/approx_palette_witness.sio tests/run-pass/viz_canonical_palettes.sio tests/run-pass/stdlib_deep_viz_dynamic_fields.sio;do
 name=$(basename "$src" .sio)
 phase "$name.check" "$SOUC_BIN" check "$src"
 phase "$name.compile" "$SOUC_BIN" compile "$src" -o "$out/$name.elf"
 chmod +x "$out/$name.elf";sha256sum "$out/$name.elf" > "$out/$name.elf.sha256"
 phase "$name.native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 180 "$out/$name.elf"
done
phase canonical-concordance "$out/reference" --legacy "$primary/BIDS-colormaps.py.txt" "$out/legacy_palette_witness.native.stdout"
phase approximation-compatibility "$out/reference" --approx "$primary/BIDS-colormaps.py.txt" "$out/historical-approx.stdout" "$out/approx_palette_witness.native.stdout"
phase figure-concordance "$out/reference" --figure "$primary/BIDS-colormaps.py.txt" "$out/viz_canonical_palettes.native.stdout" "$out/actual-canvas.ppm"
# Existing lossless Canvas exporter encodes the verified actual RLE, never SVG.
zlib_flag=()
if printf '#include <zlib.h>\n' | g++ -x c++ -E - >/dev/null 2>&1;then zlib_flag=(-lz);fi
phase canvas-export.build g++ -std=c++23 -O2 -Wall -Wextra tests/reference/scientific_canvas_export.cpp "${zlib_flag[@]}" -o "$out/canvas-export"
phase canvas-export "$out/canvas-export" "$out/actual-canvas.ppm.rle" "$out/roundtrip-canvas.ppm" "$out/actual-canvas.png"
phase canvas-export.lossless cmp "$out/actual-canvas.ppm" "$out/roundtrip-canvas.ppm"
phase png-decoder.build g++ -std=c++23 -O2 -Wall -Wextra tests/stdlib/canonical_palettes/png_decode_reference.cpp -l:libz.so.1 -o "$out/png-decoder"
ldd "$out/png-decoder" > "$out/png-decoder.dependencies.txt"
phase png-native-primary "$out/png-decoder" "$primary/BIDS-colormaps.py.txt" tests/reference/data/font8x8_basic.h "$out/viz_canonical_palettes.native.stdout" "$out/actual-canvas.png" "$out/actual-canvas.ppm"
phase legacy-field-reference.build g++ -std=c++23 -O2 -Wall -Wextra tests/reference/deep_viz_dynamic_fields.cpp -o "$out/field-reference"
phase legacy-field-concordance "$out/field-reference" "$out/stdlib_deep_viz_dynamic_fields.native.stdout"
# Mutated copies are non-native controls, preserved separately from the source.
for mutant in rgb missing text;do
 case "$mutant" in
 rgb) sed '0,/fill="rgb(68,1,84)"/s//fill="rgb(69,1,84)"/' "$out/viz_canonical_palettes.native.stdout" > "$out/$mutant.input.stdout";;
 missing) sed '0,/data-color-class="3" data-palette-index="-1"/s//data-color-class="0" data-palette-index="0"/' "$out/viz_canonical_palettes.native.stdout" > "$out/$mutant.input.stdout";;
 text) sed '0,/<text x="36"/s//<text x="-1"/' "$out/viz_canonical_palettes.native.stdout" > "$out/$mutant.input.stdout";;esac
 [[ -s $out/$mutant.input.stdout ]]
 if cmp -s "$out/$mutant.input.stdout" "$out/viz_canonical_palettes.native.stdout";then echo "vacuous $mutant mutation" >&2;exit 1;fi
 if phase "$mutant.mutant" "$out/reference" --figure "$primary/BIDS-colormaps.py.txt" "$out/$mutant.input.stdout";then echo "$mutant mutant accepted" >&2;exit 1;fi
 case "$mutant" in rgb) reason='colorbar canonical colors/shared pixel-cell geometry';;missing)reason='heatmap canonical colors/missing/under/over/strided provenance';;text)reason='actual text cell bounds';;esac
 phase "$mutant.diagnostic" grep -Fxq "$reason" "$out/$mutant.mutant.stderr"
 sha256sum "$out/$mutant.input.stdout" > "$out/$mutant.input.sha256"
done
if [[ -n ${SOUNIO_SEED_COMPILER:-} ]];then
 phase strict-seed "$SOUNIO_SEED_COMPILER" tests/run-pass/viz_canonical_palettes.sio "$out/strict-seed.elf"
 sha256sum "$SOUNIO_SEED_COMPILER" > "$out/SEED_COMPILER.sha256"
fi
phase inputs-unchanged sha256sum -c "$out/INPUTS.sha256"
echo CANONICAL_PALETTE_GATE_PASS
