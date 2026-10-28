#!/usr/bin/env bash
# Full diagnostic ICU candidate; the pristine acceptance oracle remains unchanged.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
: "${ICU_PRISTINE_ROOT:?pinned pristine archive/install root}"
: "${ICU_TWO_ASSIGN_PREFIX:?immutable two-assignment diagnostic install}"
: "${BIDI_POLICY_RAW:?explicit raw compiler for candidate-prefix refusal control}"
: "${BIDI_POLICY_PLATFORM:?pinned platform for refusal control}"
test "$#" = 1 || exit 64
case "$1" in /*) ;;*) exit 64;;esac
OUT=$(realpath -m "$1")
PRISTINE=$(realpath -e "$ICU_PRISTINE_ROOT")
BEFORE=$(realpath -e "$ICU_TWO_ASSIGN_PREFIX")
test ! -e "$OUT"
case "$OUT" in "$PRISTINE"|"$PRISTINE"/*|"$BEFORE"|"$BEFORE"/*) exit 64;;esac
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CXX=${CXX:-g++}
NODE=${NODE:-node}
MAKE=${MAKE:-make}
JOBS=${ICU_CANDIDATE_BUILD_JOBS:-4}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]]
test "$JOBS" -le "${SLURM_CPUS_PER_TASK:?}"
test -s "$PRISTINE/icu4c-79.1-sources.tgz"
test -s "$PRISTINE/install/lib/libicuuc.so.79.1"
test -s "$BEFORE/lib/libicuuc.so.79.1"
mkdir -p "$OUT/source" "$OUT/build" "$OUT/upstream"
printf '%s\n' 'Diagnostic candidate only; full corpus concordance is not pristine-oracle or whole-stdlib acceptance.' > "$OUT/SCOPE"
date -u > "$OUT/STARTED_UTC"
finish() {
 local status=$? changed=0
 trap - EXIT
 for kind in INPUTS PRISTINE BEFORE SOURCE_TREE CANDIDATE EXECUTABLES;do
  if test -s "$OUT/$kind.sha256";then
   sha256sum -c "$OUT/$kind.sha256" > "$OUT/$kind.final.verify" 2>&1 || changed=1
  fi
 done
 test "$changed" = 0 || status=1
 printf '%s\n' "$status" > "$OUT/STATUS"
 date -u > "$OUT/ENDED_UTC"
 exit "$status"
}
trap finish EXIT
sha256sum "$0" "$ROOT/tests/reference/wave4_bidi_icu_collect.cpp" "$ROOT/tests/reference/wave4_bidi_sdk_controls.cpp" \
 "$ROOT/tests/reference/wave3_bidi_reference.cpp" "$ROOT/tests/reference/wave3_bidi_cases.hpp" \
 "$ROOT/tests/reference/wave3_bidi_diagnostic.cpp" "$ROOT/scripts/stdlib/wave3_bidi_gate.sh" \
 "$ROOT/scripts/stdlib/bounded_command.mjs" "$ROOT/scripts/ci/souc-native-wrapper.sh" \
 "$ROOT/stdlib/text/data/ucd18-source.tar.gz" "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256" \
 "$(command -v "$CXX")" "$(command -v "$NODE")" "$(command -v "$MAKE")" "$BIDI_POLICY_RAW" "$BIDI_POLICY_PLATFORM" > "$OUT/INPUTS.sha256"
find "$PRISTINE/install" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/PRISTINE.sha256"
sha256sum "$PRISTINE/icu4c-79.1-sources.tgz" >> "$OUT/PRISTINE.sha256"
find "$BEFORE" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/BEFORE.sha256"
printf '06fa63ecb521048f8082f12ad988f6b678d90cbd88aafce24cbc94902b191d99  %s\n' "$PRISTINE/icu4c-79.1-sources.tgz" | sha256sum -c - > "$OUT/ARCHIVE.verify"
printf '7ea44ea43be9fd2f420b4d692d202512998c2c03a464be9f70547eb8abae2e67  %s\n' "$BEFORE/../source/icu/source/common/ubidi.cpp" | sha256sum -c - > "$OUT/BEFORE_SOURCE.verify"
find "$PRISTINE/install" "$BEFORE" -type l -printf '%p\t%l\n'|sort > "$OUT/SDK_LINKS.before"
"$CXX" --version > "$OUT/CXX_VERSION"
printf 'item\tphase\texpected\tactual\n' > "$OUT/RESULTS.tsv"
: > "$OUT/EXECUTABLES.sha256"
phase() {
 local item=$1 name=$2 expected=$3 limit=$4 status=0
 shift 4
 /usr/bin/time -v timeout "$limit" "$@" > "$OUT/$item.$name.stdout" 2> "$OUT/$item.$name.stderr" || status=$?
 printf '%s\t%s\t%s\t%s\n' "$item" "$name" "$expected" "$status" >> "$OUT/RESULTS.tsv"
 test "$status" = "$expected"
}
phase sdk extract 0 120 tar -xzf "$PRISTINE/icu4c-79.1-sources.tgz" -C "$OUT/source"
"$NODE" --input-type=module - "$OUT" <<'JS'
import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';
const out=process.argv[2],p=out+'/source/icu/source/common/ubidi.cpp';
const original=fs.readFileSync(p,'utf8'),sha=s=>crypto.createHash('sha256').update(s).digest('hex');
assert.equal(sha(original),'d6bfdaedb6cc9b3fafcb3c0fb0c13c58075b62eb9ff278dfb979bae245f32ebd');
const a=['            case LRI:','            case RLI:','                flags|=(DIRPROP_FLAG(ON)|DIRPROP_FLAG_LR(embeddingLevel));','                levels[i]=NO_OVERRIDE(embeddingLevel);'].join('\n');
const b=['                embeddingLevel = static_cast<UBiDiLevel>(stack[stackLast]) & ~ISOLATE;','                flags|=(DIRPROP_FLAG(ON)|DIRPROP_FLAG_LR(embeddingLevel));','                previousLevel=embeddingLevel;','                levels[i]=NO_OVERRIDE(embeddingLevel);'].join('\n');
const c=['    dirProps=bd->pBiDi->dirProps;','    dirProp=dirProps[position];','    if(dirProp==ON) {'].join('\n');
let derived=original;const changes=[];
for(const before of [a,b,c]) {
 assert.equal(derived.split(before).length,2,'nonunique exact source anchor');
 const after=before===c
  ?before.replace('    if(dirProp==ON) {','    // UAX9 rev52 X6 precedes BD14/BD15/N0: overridden ON is not a bracket.\n    if(dirProp==ON && !(bd->pBiDi->levels[position]&UBIDI_LEVEL_OVERRIDE)) {')
  :before.replace('levels[i]=NO_OVERRIDE(embeddingLevel);','levels[i]=embeddingLevel;');
 derived=derived.replace(before,after);changes.push({before,after});
}

const hp=out+'/source/icu/source/common/ubidiimp.h';
const header=fs.readFileSync(hp,'utf8');
assert.equal(sha(header),'ee6416a915822a0ea43cbef7916c1fbcc7dafdf518c75b7e50549c68db8f828f');
function change(before,after,count=1) {
 assert.equal(derived.split(before).length-1,count,'source anchor '+before.slice(0,90));
 derived=derived.split(before).join(after);changes.push({before,after,count});
}
change(['        /* if the last real char was ON, change NSM to ON so that it',
'           will stay ON even if the last real char is a bracket which',
'           may be changed to L or R */','        newProp=pLastIsoRun->lastBase;',
'        if(newProp==ON)','            dirProps[position]=newProp;'].join('\n'),
['        /* N0: leave original NSMs intact so W1 inherits a bracket after',
'           its final N0 resolution, including across ignored BN characters. */',
'        newProp=pLastIsoRun->lastBase;'].join('\n'));
const helpers=String.raw`static void
bracketStartSequence(BracketData *bd, IsoRun *run) {
    run->bd16Id=bd->sequenceCounter++;
    run->bd16Depth=0;
}

/* BD16 preflight tracks canonical closing values, not ICU's pending N0
   records (which may include completed pairs and duplicate synonyms). */
static void
bracketObserveBD16(BracketData *bd, int32_t position) {
    IsoRun *run=&bd->isoRuns[bd->isoRunLast];
    if(!bd->discover || bd->suppressed[run->bd16Id])
        return;
    UChar32 c=bd->pBiDi->text[position];
    UBidiPairedBracketType type=ubidi_getPairedBracketType(c);
    if(type==U_BPT_OPEN) {
        if(run->bd16Depth==63) {
            bd->suppressed[run->bd16Id]=true;
            run->bd16Depth=0;
            return;
        }
        UChar32 close=u_getBidiPairedBracket(c);
        run->bd16Stack[run->bd16Depth++]=close==0x232A ? 0x3009 : close;
    } else if(type==U_BPT_CLOSE) {
        if(c==0x232A) c=0x3009;
        for(int32_t j=run->bd16Depth-1;j>=0;--j) {
            if(run->bd16Stack[j]==c) {
                run->bd16Depth=j;
                break;
            }
        }
    }
}

`;
change('static void\nbracketInit(UBiDi *pBiDi, BracketData *bd) {',
helpers+'static void\nbracketInit(UBiDi *pBiDi, BracketData *bd, UBool *suppressed, UBool discover) {');
change('    bd->pBiDi=pBiDi;\n    bd->isoRunLast=0;',
'    bd->pBiDi=pBiDi;\n    bd->suppressed=suppressed;\n    bd->discover=discover;\n    bd->sequenceCounter=0;\n    bd->isoRunLast=0;\n    bracketStartSequence(bd, &bd->isoRuns[0]);');
change('    bd->isoRuns[0].limit=0;\n    bd->isoRuns[0].level=level;',
'    bd->isoRuns[0].limit=0;\n    bracketStartSequence(bd, &bd->isoRuns[0]);\n    bd->isoRuns[0].level=level;');
change('    pLastIsoRun->limit=pLastIsoRun->start;\n    pLastIsoRun->level=embeddingLevel;',
'    pLastIsoRun->limit=pLastIsoRun->start;\n    bracketStartSequence(bd, pLastIsoRun);\n    pLastIsoRun->level=embeddingLevel;');
change('    pLastIsoRun->start=pLastIsoRun->limit=lastLimit;\n    pLastIsoRun->level=level;',
'    pLastIsoRun->start=pLastIsoRun->limit=lastLimit;\n    bracketStartSequence(bd, pLastIsoRun);\n    pLastIsoRun->level=level;');
change('    if(dirProp==ON && !(bd->pBiDi->levels[position]&UBIDI_LEVEL_OVERRIDE)) {',
'    if(dirProp==ON && !(bd->pBiDi->levels[position]&UBIDI_LEVEL_OVERRIDE))\n        bracketObserveBD16(bd, position);\n    if(dirProp==ON && !(bd->pBiDi->levels[position]&UBIDI_LEVEL_OVERRIDE)\n       && !bd->suppressed[pLastIsoRun->bd16Id]) {');
change('resolveExplicitLevels(UBiDi *pBiDi, UErrorCode *pErrorCode) {',
'resolveExplicitLevelsPass(UBiDi *pBiDi, UErrorCode *pErrorCode,\n                          UBool *suppressed, UBool discover) {');
change('        bracketInit(pBiDi, &bracketData);',
'        bracketInit(pBiDi, &bracketData, suppressed, discover);',2);
const wrapper=String.raw`/* BD16 overflow returns an empty list for the whole isolating run sequence.
   Restore original properties and replay ICU's own explicit resolver so pairs
   completed before the overflow are also removed. Other sequences stay active.
   Sequence identities follow the same explicit/isolate transitions in both
   passes; there is at most one boundary plus one isolate push per code unit. */
static UBiDiDirection
resolveExplicitLevels(UBiDi *pBiDi, UErrorCode *pErrorCode) {
    if(U_FAILURE(*pErrorCode)) return UBIDI_LTR;
    const int32_t length=pBiDi->length;
    if(length>(INT32_MAX-2)/2) {
        *pErrorCode=U_MEMORY_ALLOCATION_ERROR;
        return UBIDI_LTR;
    }
    const int32_t capacity=length*2+2;
    icu::LocalMemory<DirProp> original(static_cast<DirProp*>(uprv_malloc(length>0 ? length : 1)));
    icu::LocalMemory<UBool> suppressed(static_cast<UBool*>(uprv_malloc(capacity*sizeof(UBool))));
    if(original.isNull() || suppressed.isNull()) {
        *pErrorCode=U_MEMORY_ALLOCATION_ERROR;
        return UBIDI_LTR;
    }
    uprv_memcpy(original.getAlias(),pBiDi->dirProps,length);
    uprv_memset(suppressed.getAlias(),0,capacity*sizeof(UBool));
    const Flags originalFlags=pBiDi->flags;
    UBiDiDirection direction=resolveExplicitLevelsPass(pBiDi,pErrorCode,suppressed.getAlias(),true);
    if(U_FAILURE(*pErrorCode) || direction==static_cast<UBiDiDirection>(-1)) return direction;
    UBool overflow=false;
    for(int32_t i=0;i<capacity;++i) overflow|=suppressed[i];
    if(!overflow) return direction;
    uprv_memcpy(pBiDi->dirProps,original.getAlias(),length);
    pBiDi->flags=originalFlags;
    return resolveExplicitLevelsPass(pBiDi,pErrorCode,suppressed.getAlias(),false);
}

`;
change('/*\n * Use a pre-specified embedding levels array:',wrapper+'/*\n * Use a pre-specified embedding levels array:');
let changedHeader=header;
function hchange(before,after) {
 assert.equal(changedHeader.split(before).length,2,'header anchor');
 changedHeader=changedHeader.replace(before,after);
}
hchange('    UBiDiDirection contextDir;          /* L or R to use as context for following openings */\n} IsoRun;',
'    UBiDiDirection contextDir;          /* L or R to use as context for following openings */\n    int32_t bd16Id;\n    int32_t bd16Depth;\n    UChar32 bd16Stack[63];\n} IsoRun;');
hchange('    UBool isNumbersSpecial;             /* reordering mode for NUMBERS_SPECIAL */',
'    UBool *suppressed;                  /* BD16 overflow by stable sequence identity */\n    UBool discover;\n    int32_t sequenceCounter;\n    UBool isNumbersSpecial;             /* reordering mode for NUMBERS_SPECIAL */');
fs.writeFileSync(out+'/ubidiimp.pristine.h',header,{flag:'wx'});
fs.writeFileSync(hp,changedHeader);
fs.writeFileSync(out+'/HEADER_CUSTODY.json',JSON.stringify({
 original_sha256:sha(header),derived_sha256:sha(changedHeader),
 scope:'internal sequence state for normative BD16 preflight; no public ABI change'
},null,2)+'\n',{flag:'wx'});

let reversed=derived;
for(const change of [...changes].reverse())reversed=reversed.split(change.after).join(change.before);
assert.equal(reversed,original);
fs.writeFileSync(out+'/ubidi.pristine.cpp',original,{flag:'wx'});
fs.writeFileSync(p,derived);
fs.writeFileSync(out+'/PATCH_CUSTODY.json',JSON.stringify({
 scope:'diagnostic full-corpus candidate, not pristine oracle',
 normative:'https://www.unicode.org/reports/tr9/tr9-52.html#N0',
 original_sha256:sha(original),derived_sha256:sha(derived),
 isolate_override_assignments:2,current_type_guard:true,nsm_preserved:true,bd16_empty_list_replay:true,reverse_byte_equal:true,changes
},null,2)+'\n',{flag:'wx'});
JS
diff_status=0
diff -u --label a/source/common/ubidi.cpp --label b/source/common/ubidi.cpp \
 "$OUT/ubidi.pristine.cpp" "$OUT/source/icu/source/common/ubidi.cpp" > "$OUT/override-current-type.patch" || diff_status=$?
test "$diff_status" = 1
diff_status=0
diff -u --label a/source/common/ubidiimp.h --label b/source/common/ubidiimp.h "$OUT/ubidiimp.pristine.h" "$OUT/source/icu/source/common/ubidiimp.h" >> "$OUT/override-current-type.patch" || diff_status=$?
test "$diff_status" = 1
find "$OUT/source/icu" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/SOURCE_TREE.sha256"
phase corpus extract 0 120 tar -xzf "$ROOT/stdlib/text/data/ucd18-source.tar.gz" -C "$OUT/upstream"
(cd "$OUT/upstream";sha256sum -c "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256") > "$OUT/UCD.verify"
run_collect() {
 local label=$1 prefix=$2 expected=$3
 export LD_LIBRARY_PATH="$prefix/lib${BIDI_TOOLCHAIN_LD:+:$BIDI_TOOLCHAIN_LD}"
 phase "$label" compile-collect 0 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$prefix/include" "$ROOT/tests/reference/wave4_bidi_icu_collect.cpp" -L"$prefix/lib" \
  -licui18n -licuuc -licudata -o "$OUT/$label-collect"
 phase "$label" compile-locked 0 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$prefix/include" "$ROOT/tests/reference/wave3_bidi_reference.cpp" -L"$prefix/lib" \
  -licui18n -licuuc -licudata -o "$OUT/$label-locked"
 phase "$label" compile-sdk-controls 0 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$prefix/include" "$ROOT/tests/reference/wave4_bidi_sdk_controls.cpp" -L"$prefix/lib" \
  -licuuc -licudata -o "$OUT/$label-sdk-controls"
 phase "$label" sdk-controls "$expected" 120 "$OUT/$label-sdk-controls"
 sha256sum "$OUT/$label-sdk-controls" >> "$OUT/EXECUTABLES.sha256"
 ldd "$OUT/$label-collect" > "$OUT/$label.BOUND_LIBRARIES"
 for library in libicuuc libicudata;do grep -F "$library.so.79 => $prefix/lib/" "$OUT/$label.BOUND_LIBRARIES";done
 sha256sum "$OUT/$label-collect" "$OUT/$label-locked" >> "$OUT/EXECUTABLES.sha256"
 phase "$label" full-collect "$expected" 600 "$OUT/$label-collect" \
  "$OUT/upstream/ucd-18.0.0" "$OUT/$label.mismatches.tsv" none
 grep -Fx 'BIDI_COLLECT_COMPLETE cases=861948 class_cases=770241 character_cases=91707 read_cases=861948 executed_cases=861948' "$OUT/$label.full-collect.stdout"
 phase "$label" locked-reference "$expected" 600 "$OUT/$label-locked" \
  "$OUT/upstream/ucd-18.0.0" "$OUT/$label.optimized.tsv"
}
run_collect pristine "$PRISTINE/install" 1
grep -F 'mixed level disagreement kind=1 line=61' "$OUT/pristine.locked-reference.stderr"
run_collect isolate2 "$BEFORE" 1
grep -F 'mixed level disagreement kind=1 line=67' "$OUT/isolate2.locked-reference.stderr"
cd "$OUT/build"
export CXX
export CXXFLAGS="-O2 -std=c++23 -fno-omit-frame-pointer"
export LD_LIBRARY_PATH="${BIDI_TOOLCHAIN_LD:-}"
phase sdk configure 0 300 ../source/icu/source/configure --prefix="$OUT/install" \
 --disable-tests --disable-samples --enable-shared --disable-static
phase sdk make 0 2400 "$MAKE" -j"$JOBS"
phase sdk install 0 300 "$MAKE" install
find "$OUT/install" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/CANDIDATE.sha256"
sha256sum "$OUT/override-current-type.patch" "$OUT/PATCH_CUSTODY.json" >> "$OUT/CANDIDATE.sha256"
run_collect candidate "$OUT/install" 0
grep -Fx 'BIDI_COLLECT_BAD bases=0 levels=0 orders=0 counts=0' "$OUT/candidate.full-collect.stdout"
grep -F 'BIDI_ICU_OFFICIAL_OK cases=861948 class_cases=770241 character_cases=91707 ' "$OUT/candidate.locked-reference.stdout"
for mode in level base order missing duplicate;do
 phase "mutant-$mode" full-collect 1 600 "$OUT/candidate-collect" \
  "$OUT/upstream/ucd-18.0.0" "$OUT/mutant-$mode.tsv" "$mode"
 grep -Fx "BIDI_COLLECT_MUTATION mode=$mode hits=1" "$OUT/mutant-$mode.full-collect.stdout"
 grep -Fx 'BIDI_COLLECT_VERDICT FAIL' "$OUT/mutant-$mode.full-collect.stdout"
 case "$mode" in
  level) wanted='BIDI_COLLECT_BAD bases=0 levels=1 orders=0 counts=0';;
  base) wanted='BIDI_COLLECT_BAD bases=1 levels=0 orders=0 counts=0';;
  order) wanted='BIDI_COLLECT_BAD bases=0 levels=0 orders=1 counts=0';;
  missing|duplicate) wanted='BIDI_COLLECT_BAD bases=0 levels=0 orders=0 counts=1';;
 esac
 grep -Fx "$wanted" "$OUT/mutant-$mode.full-collect.stdout"
done
export ICU_PREFIX="$OUT/install" MADAROS_RAW_BIN="$BIDI_POLICY_RAW" STDLIB_PLATFORM_LIBRARY="$BIDI_POLICY_PLATFORM"
phase policy candidate-as-primary 2 30 bash "$ROOT/scripts/stdlib/wave3_bidi_gate.sh" "$OUT/forbidden-primary"
grep -F 'diagnostic ICU candidate cannot serve as the locked acceptance oracle' "$OUT/policy.candidate-as-primary.stderr"
test ! -e "$OUT/forbidden-primary"
find "$PRISTINE/install" "$BEFORE" -type l -printf '%p\t%l\n'|sort > "$OUT/SDK_LINKS.after"
cmp "$OUT/SDK_LINKS.before" "$OUT/SDK_LINKS.after"
(cd "$OUT/upstream";sha256sum -c "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256") > "$OUT/UCD.final.verify"
printf '%s\n' 'CANDIDATE_FULL861948_CONCORDANCE_ONLY' > "$OUT/VERDICT"
echo CANDIDATE_FULL861948_CONCORDANCE_ONLY
