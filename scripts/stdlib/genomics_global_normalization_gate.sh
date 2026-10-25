#!/usr/bin/env bash
# Global-reference reuse; no out-of-core, inference-scale or calibration claim.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: genomics_global_normalization_gate.sh FRESH_ABSOLUTE_OUTPUT}"
[[ "$OUT" == /* && ! -e "$OUT" && ! -L "$OUT" ]] || exit 64
: "${MADAROS_RAW_BIN:?explicit source-built raw compiler required}"
MADAROS_RAW_BIN="$(readlink -f "$MADAROS_RAW_BIN")";export MADAROS_RAW_BIN
[[ -x "$MADAROS_RAW_BIN" && -z ${SOUNIO_SOUC_BIN:-} ]] || exit 64
mkdir -p "$(dirname "$OUT")";mkdir "$OUT"
trap 'code=$?;echo "$code" > "$OUT/STATUS"' EXIT
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
id > "$OUT/identity.txt"
inputs() {
 find stdlib scripts/lib -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
 sha256sum bin/{souc,madaros,souc-lean-single-x86_64} scripts/stdlib/genomics_{batch,global_normalization}_gate.sh tests/reference/genomics_global_normalization_reference.cpp tests/stdlib/genomics_batch/global_normalization.sio tests/compile-fail/genomics_normalization_private.sio
}
inputs > "$OUT/SOURCE.sha256"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$OUT/COMPILERS.sha256"
phase() { local label=$1;shift;local code=0;"$@" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr" || code=$?;printf '%s\t%s\n' "$label" "$code" >> "$OUT/RESULTS.tsv";[[ $code == 0 ]] || { cat "$OUT/$label.stdout" "$OUT/$label.stderr" >&2;return "$code"; }; }
no_diagnostics() { ! grep -Eq '^error:|error\[E' "$1" "$2"; }
phase cxx-build g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/genomics_global_normalization_reference.cpp -lquadmath -o "$OUT/reference.elf"
sha256sum "$OUT/reference.elf" > "$OUT/ELFS.sha256"
for engine in madaros lean_single;do
 for file in stdlib/genomics/{count_normalization,count_pipeline,count_pipeline_report}.sio tests/stdlib/genomics_batch/global_normalization.sio;do
  name=$(basename "$file")
  phase "$engine.$name.check" env SOUNIO_SOUC_ENGINE="$engine" bin/souc check "$file"
  no_diagnostics "$OUT/$engine.$name.check.stdout" "$OUT/$engine.$name.check.stderr"
 done
 phase "$engine.compile" env SOUNIO_SOUC_ENGINE="$engine" bin/souc compile tests/stdlib/genomics_batch/global_normalization.sio -o "$OUT/$engine.elf"
 no_diagnostics "$OUT/$engine.compile.stdout" "$OUT/$engine.compile.stderr"
 chmod +x "$OUT/$engine.elf";sha256sum "$OUT/$engine.elf" >> "$OUT/ELFS.sha256"
 phase "$engine.native" /usr/bin/time -v timeout 180 "$OUT/$engine.elf"
 phase "$engine.reference" "$OUT/reference.elf" "$OUT/$engine.native.stdout"
 code=0
 SOUNIO_SOUC_ENGINE="$engine" bin/souc compile tests/compile-fail/genomics_normalization_private.sio -o "$OUT/$engine.private.elf" > "$OUT/$engine.private.stdout" 2> "$OUT/$engine.private.stderr" || code=$?
 echo "$code" > "$OUT/$engine.private.exit"
 [[ $code == 1 && ! -s "$OUT/$engine.private.elf" ]] && grep -Eq 'error\[E259\]' "$OUT/$engine.private.stdout" "$OUT/$engine.private.stderr"
done
# Remove only equality assertions from the consumer, then require its clean
# production execution/oracle PASS. That identical consumer discriminates an
# actually compiled and executed local-recompute implementation below.
mkdir "$OUT/mutation"
cp -a stdlib "$OUT/mutation/stdlib"
node --input-type=module - tests/stdlib/genomics_batch/global_normalization.sio "$OUT/mutation/consumer.sio" "$OUT/mutation/stdlib/genomics/count_pipeline.sio" <<'NODE'
import fs from 'node:fs';
const [test,out,model]=process.argv.slice(2);let consumer=fs.readFileSync(test,'utf8');
for(const line of ['     assert(count_pipeline_factor(&w,j)==count_pipeline_factor(full,(sample+2)%4))\n','     assert(count_pipeline_offset(&w,j)==count_pipeline_offset(full,(sample+2)%4))\n']){if(!consumer.includes(line))throw Error('missing equality control');consumer=consumer.replace(line,'');}
fs.writeFileSync(out,consumer);
let s=fs.readFileSync(model,'utf8');const allocation='if code==0 && local_normalization';if(s.split(allocation).length!==3)throw Error('allocation census');s=s.replaceAll(allocation,'if code==0');
const a=' var i:i64=0\n while i<w.n {\n  let from=integer_buffer_at(&w.counts,i)\n  numeric_buffer_set(&!w.factors,i,count_normalization_factor(normalization,from))\n  numeric_buffer_set(&!w.norm_logs,i,count_normalization_log(normalization,from));i=i+1\n }';
if(!s.includes(a))throw Error('global copy source');
s=s.replace(a,' let recomputed=count_csr_size_factors_into(&w.csr,count_normalization_mode(normalization),&!w.factors,&!w.norm_logs,&!w.norm_work,&!w.norm_anchors,&!w.normalization)\n if recomputed!=0 {return 20+recomputed}');fs.writeFileSync(model,s);
NODE
mutation_inputs() {
 find "$OUT/mutation/stdlib" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
 sha256sum "$OUT/mutation/consumer.sio"
}
mutation_inputs > "$OUT/MUTATION_INPUTS.sha256"
phase mutation-source-initial sha256sum -c "$OUT/MUTATION_INPUTS.sha256"
phase control.compile env SOUNIO_SOUC_ENGINE=madaros bin/souc compile "$OUT/mutation/consumer.sio" -o "$OUT/control.elf"
no_diagnostics "$OUT/control.compile.stdout" "$OUT/control.compile.stderr"
chmod +x "$OUT/control.elf";sha256sum "$OUT/control.elf" >> "$OUT/ELFS.sha256"
phase control.native timeout 180 "$OUT/control.elf"
phase control.reference "$OUT/reference.elf" "$OUT/control.native.stdout"
phase mutant.compile env SOUNIO_STDLIB_PATH="$OUT/mutation/stdlib" SOUNIO_SOUC_ENGINE=madaros bin/souc compile "$OUT/mutation/consumer.sio" -o "$OUT/mutant.elf"
no_diagnostics "$OUT/mutant.compile.stdout" "$OUT/mutant.compile.stderr"
chmod +x "$OUT/mutant.elf";sha256sum "$OUT/mutant.elf" >> "$OUT/ELFS.sha256"
phase mutant.native timeout 180 "$OUT/mutant.elf"
code=0
"$OUT/reference.elf" "$OUT/mutant.native.stdout" > "$OUT/mutant.reference.stdout" 2> "$OUT/mutant.reference.stderr" || code=$?
echo "$code" > "$OUT/mutant.reference.exit"
[[ $code == 2 ]] && grep -Fxq 'GENOMICS_GLOBAL_NORMALIZATION_REFERENCE_REJECT global normalization mismatch: factor' "$OUT/mutant.reference.stderr"
# Regression replay of all six established fixtures, both engines + independent refs.
phase existing-six bash scripts/stdlib/genomics_batch_gate.sh "$OUT/existing-six"
inputs > "$OUT/SOURCE.final.sha256";cmp "$OUT/SOURCE.sha256" "$OUT/SOURCE.final.sha256"
mutation_inputs > "$OUT/MUTATION_INPUTS.final.sha256"
cmp "$OUT/MUTATION_INPUTS.sha256" "$OUT/MUTATION_INPUTS.final.sha256"
phase mutation-source-final sha256sum -c "$OUT/MUTATION_INPUTS.sha256"
phase source-final sha256sum -c "$OUT/SOURCE.sha256"
phase compilers-final sha256sum -c "$OUT/COMPILERS.sha256"
phase elfs-final sha256sum -c "$OUT/ELFS.sha256"
printf '%s\n' 'GENOMICS_GLOBAL_NORMALIZATION_GATE_PASS new_fixture_both_engines=true scenarios=16 point_means=8 existing_fixtures=6 real_local_recompute_mutant=1' > "$OUT/result.txt"
cat "$OUT/result.txt"
