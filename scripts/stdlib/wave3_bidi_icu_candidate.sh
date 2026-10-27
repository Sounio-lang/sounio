#!/usr/bin/env bash
# Supplemental diagnostic candidate ONLY. Never selects/replaces the pristine oracle.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
: "${SLURM_JOB_ID:?scheduled Slurm build required}"
: "${ICU_PRISTINE_ROOT:?absolute pristine ICU79.1 tooling root required}"
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 2
OUT="$(realpath -m "$1")"
PRISTINE="$(realpath -e "$ICU_PRISTINE_ROOT")"
[[ ! -e "$OUT" && "$OUT" != "$PRISTINE/"* && -s "$PRISTINE/icu4c-79.1-sources.tgz" ]] || exit 2
NODE="${NODE:-node}"
CXX="${CXX:-g++}"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT/source" "$OUT/build" "$OUT/upstream"
date -u +%FT%TZ > "$OUT/STARTED_UTC"
printf '%s\n' 'Diagnostic candidate only; NOT the pristine ICU79.1 oracle and NOT a stdlib acceptance waiver.' > "$OUT/SCOPE"
finish(){
 local status=$? changed=0
 trap - EXIT
 for kind in PRISTINE SOURCES SOURCE_TREE CANDIDATE EXECUTABLES;do
  if [[ -s "$OUT/$kind.sha256" ]];then sha256sum -c "$OUT/$kind.sha256" > "$OUT/$kind.final.verify" 2>&1 || changed=1;fi
 done
 [[ "$changed" == 0 ]] || status=1
 date -u +%FT%TZ > "$OUT/ENDED_UTC"
 printf '%s\n' "$status" > "$OUT/STATUS"
 exit "$status"
}
trap finish EXIT
find "$PRISTINE/install" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/PRISTINE.sha256"
sha256sum "$PRISTINE/icu4c-79.1-sources.tgz" "$PRISTINE/icu/source/common/ubidi.cpp" >> "$OUT/PRISTINE.sha256"
printf '%s  %s\n' 06fa63ecb521048f8082f12ad988f6b678d90cbd88aafce24cbc94902b191d99 "$PRISTINE/icu4c-79.1-sources.tgz" | sha256sum -c - > "$OUT/ARCHIVE.initial.verify"
sha256sum "$ROOT/scripts/stdlib/wave3_bidi_icu_candidate.sh" "$ROOT/tests/reference/wave3_bidi_cases.hpp" "$ROOT/tests/reference/wave3_bidi_reference.cpp" "$ROOT/tests/reference/wave3_bidi_diagnostic.cpp" "$ROOT/stdlib/text/data/ucd18-source.tar.gz" "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256" > "$OUT/SOURCES.sha256"
timeout 120 tar -xzf "$PRISTINE/icu4c-79.1-sources.tgz" -C "$OUT/source"
"$NODE" --input-type=module - "$OUT" <<'JS'
import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';
const out=process.argv[2],p=out+'/source/icu/source/common/ubidi.cpp',original=fs.readFileSync(p,'utf8'),sha=x=>crypto.createHash('sha256').update(x).digest('hex');
assert.equal(sha(original),'d6bfdaedb6cc9b3fafcb3c0fb0c13c58075b62eb9ff278dfb979bae245f32ebd');
const a=['            case LRI:','            case RLI:','                flags|=(DIRPROP_FLAG(ON)|DIRPROP_FLAG_LR(embeddingLevel));','                levels[i]=NO_OVERRIDE(embeddingLevel);'].join('\n');
const b=['                embeddingLevel = static_cast<UBiDiLevel>(stack[stackLast]) & ~ISOLATE;','                flags|=(DIRPROP_FLAG(ON)|DIRPROP_FLAG_LR(embeddingLevel));','                previousLevel=embeddingLevel;','                levels[i]=NO_OVERRIDE(embeddingLevel);'].join('\n');
let derived=original;const changes=[];
for(const before of [a,b]){
 assert.equal(derived.split(before).length,2,'nonunique source anchor');
 const after=before.replace('levels[i]=NO_OVERRIDE(embeddingLevel);','levels[i]=embeddingLevel;');
 derived=derived.replace(before,after);changes.push({before,after});
}
let reversed=derived;for(const c of [...changes].reverse())reversed=reversed.replace(c.after,c.before);assert.equal(reversed,original);
fs.writeFileSync(out+'/ubidi.original.cpp',original,{flag:'wx'});fs.writeFileSync(p,derived);
fs.writeFileSync(out+'/PATCH_CUSTODY.json',JSON.stringify({scope:'diagnostic candidate; not pristine oracle',original_sha256:sha(original),derived_sha256:sha(derived),changed_assignments:2,reverse_byte_equal:true,changes},null,2)+'\n',{flag:'wx'});
JS
status=0
diff -u --label a/source/common/ubidi.cpp --label b/source/common/ubidi.cpp "$OUT/ubidi.original.cpp" "$OUT/source/icu/source/common/ubidi.cpp" > "$OUT/isolate-override.patch" || status=$?
[[ "$status" == 1 ]]
# Exact source anchors and reverse byte equality above bind the derived source.
# The unified diff is a review artifact; no host patch utility is required.
find "$OUT/source/icu" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_TREE.sha256"
(cd "$OUT/build";CXX="$CXX" CXXFLAGS="-O2 -std=c++23 -fno-omit-frame-pointer" timeout 300 ../source/icu/source/configure --prefix="$OUT/install" --disable-tests --disable-samples --enable-shared --disable-static > configure.stdout 2> configure.stderr)
(cd "$OUT/build";timeout 2400 make -j4 > make.stdout 2> make.stderr)
(cd "$OUT/build";timeout 300 make install > install.stdout 2> install.stderr)
find "$OUT/install" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/CANDIDATE.sha256"
sha256sum "$OUT/isolate-override.patch" "$OUT/PATCH_CUSTODY.json" >> "$OUT/CANDIDATE.sha256"
export LD_LIBRARY_PATH="$OUT/install/lib"
"$OUT/install/bin/icuinfo" > "$OUT/ICUINFO" 2> "$OUT/ICUINFO.stderr"
timeout 120 tar -xzf "$ROOT/stdlib/text/data/ucd18-source.tar.gz" -C "$OUT/upstream"
(cd "$OUT/upstream";sha256sum -c "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256") > "$OUT/UCD.final.verify"
for name in reference diagnostic;do
 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -I"$OUT/install/include" "$ROOT/tests/reference/wave3_bidi_$name.cpp" -L"$OUT/install/lib" -licui18n -licuuc -licudata -o "$OUT/$name" > "$OUT/$name.build.stdout" 2> "$OUT/$name.build.stderr"
 ldd "$OUT/$name" > "$OUT/$name.BOUND_LIBRARIES"
 for library in libicuuc libicudata;do grep -F "$library.so.79 => $OUT/install/lib/" "$OUT/$name.BOUND_LIBRARIES";done
done
sha256sum "$OUT/reference" "$OUT/diagnostic" > "$OUT/EXECUTABLES.sha256"
oracle=0
timeout 600 "$OUT/reference" "$OUT/upstream/ucd-18.0.0" "$OUT/ICU_OPTIMIZED_LEVELS.tsv" > "$OUT/reference.stdout" 2> "$OUT/reference.stderr" || oracle=$?
printf '%s\n' "$oracle" > "$OUT/OFFICIAL_ORACLE_STATUS"
proof=0
timeout 120 "$OUT/diagnostic" "$OUT/upstream/ucd-18.0.0" > "$OUT/diagnostic.stdout" 2> "$OUT/diagnostic.stderr" || proof=$?
printf '%s\n' "$proof" > "$OUT/DIAGNOSTIC_STATUS"
grep -Fx BIDI_UAX9_LINE61_CPP_NORMATIVE_OK "$OUT/diagnostic.stdout"
grep -Fx "BIDI_UAX9_DUAL_NESTED_CPP_OK cases=3" "$OUT/diagnostic.stdout"
[[ "$proof" == 0 ]] || exit 1
grep -Fx BIDI_ICU_LINE61_AGREEMENT "$OUT/diagnostic.stdout"
[[ "$oracle" == 0 ]] || exit 1
grep -F BIDI_ICU_OFFICIAL_OK "$OUT/reference.stdout"
echo SUPPLEMENTAL_ICU_CANDIDATE_CONCORDANCE_ONLY
