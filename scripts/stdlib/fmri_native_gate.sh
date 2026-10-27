#!/usr/bin/env bash
# Bounded native input-dependent fMRI QC and independent C++23 acceptance.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
out=$(realpath -m "$1")
[[ ! -e "$out" ]] || exit 64
case "$out/" in "$root/stdlib/"*|"$root/tests/"*|"$root/scripts/"*) exit 64;;esac
if [[ -n "${SLURM_JOB_ID:-}" ]];then authority="slurm:$SLURM_JOB_ID"
elif [[ "${GITHUB_ACTIONS:-}" == true && "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ && "${GITHUB_JOB:-}" =~ ^[A-Za-z0-9_-]+$ ]];then authority="github:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled worker or official CI required" >&2;exit 64;fi
raw=$(realpath -e "${MADAROS_RAW_BIN:?qualified raw compiler required}")
platform=$(realpath -e "${STDLIB_PLATFORM_LIBRARY:?qualified native runtime required}")
[[ -x "$raw" && -s "$platform" && "${platform##*/}" == libstdlib_platform.so ]] || exit 64
node=${NODE:-node}
cxx=${CXX:-g++}
mkdir -p "$out"
printf '%s\n' "$authority" > "$out/EXECUTION_AUTHORITY"
export SOUNIO_SOUC_BIN="$raw" SOUC_NATIVE_BIN="$raw" SOUNIO_SOUC_RAW_MODE=modular
export SOUNIO_STDLIB_PATH="$root/stdlib"
export LD_LIBRARY_PATH="$(dirname "$platform")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288
ulimit -c 0
cd "$root"
catalog(){
 find stdlib tests/fixtures/fmri tests/stdlib/fmri -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
 sha256sum tests/reference/stdlib_fmri_native_reference.cpp scripts/stdlib/fmri_native_gate.sh scripts/stdlib/bounded_command.mjs scripts/ci/souc-native-wrapper.sh
}
catalog > "$out/INPUTS.sha256"
sha256sum "$raw" "$platform" > "$out/COMPILER_RUNTIME.sha256"
printf 'case\tphase\tstatus\n' > "$out/RESULTS.tsv"
: > "$out/EXECUTABLES.sha256"
finish(){
 local r=$? bad=0
 trap - EXIT
 catalog > "$out/INPUTS.final.sha256" || bad=1
 cmp "$out/INPUTS.sha256" "$out/INPUTS.final.sha256" > "$out/INPUTS.catalog.verify" 2>&1 || bad=1
 for name in INPUTS COMPILER_RUNTIME EXECUTABLES;do
  sha256sum -c "$out/$name.sha256" > "$out/$name.final.verify" 2>&1 || bad=1
 done
 find "$out" -type f \( -name '*.sio' -o -name '*.nii' -o -name '*.tsv' -o -name '*.log' \) -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$out/ARTIFACTS.sha256" || bad=1
 sha256sum "$out/DERIVATIONS.json" >> "$out/ARTIFACTS.sha256" || bad=1
 sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify" 2>&1 || bad=1
 [[ "$bad" == 0 ]] || r=1
 echo "$r" > "$out/STATUS"
 if [[ "$r" == 0 ]];then echo FMRI_NATIVE_GATE_OK;fi
 exit "$r"
}
trap finish EXIT
phase(){
 local name=$1 kind=$2 seconds=$3 r=0;shift 3
 "$node" "$root/scripts/stdlib/bounded_command.mjs" "$seconds" 16777216 -- "$@" > "$out/$name.$kind.log" 2>&1 || r=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$r" >> "$out/RESULTS.tsv"
 return "$r"
}
phase reference compile 180 "$cxx" -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror tests/reference/stdlib_fmri_native_reference.cpp -o "$out/reference"
sha256sum "$out/reference" >> "$out/EXECUTABLES.sha256"
"$cxx" --version > "$out/CXX_VERSION.txt"
ldd "$out/reference" > "$out/BOUND_LIBRARIES.txt"
! grep -q 'not found' "$out/BOUND_LIBRARIES.txt"
awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' "$out/BOUND_LIBRARIES.txt" | LC_ALL=C sort -u | xargs -r sha256sum > "$out/BOUND_LIBRARIES.sha256"
fixture="$root/tests/fixtures/fmri"
phase reference mutants 30 "$out/reference" --mutants "$fixture/tiny_real_slice_nifti1.nii" "$fixture/tiny_real_slice_nifti2.nii" "$fixture/tiny_real_motion.tsv" "$fixture/tiny_atlas_labels_nifti1.nii" "$fixture/tiny_index_config.v1.tsv" "$out/cases"
while IFS=$'\t' read -r name expected;do
 phase "$name" config 30 "$out/reference" --sounio-config "$out/cases/$name/config.tsv"
done < "$out/cases/PLAN.tsv"
"$node" --input-type=module - "$root" "$out" <<'JS'
import fs from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';
const [root,out]=process.argv.slice(2),test=root+'/tests/stdlib/fmri/test_pipeline_real_e2e.sio';
const original=fs.readFileSync(test),text=new TextDecoder('utf-8',{fatal:true}).decode(original);
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
fs.writeFileSync(out+'/ORIGINAL_TEST.sio',original,{flag:'wx'});
const records=[];
for(const line of fs.readFileSync(out+'/cases/PLAN.tsv','utf8').trimEnd().split('\n')){
 const [name,rc,...extra]=line.split('\t');assert.equal(extra.length,0);assert.match(name,/^[a-z-]+$/);assert.match(rc,/^[0-9]+$/);
 let derived=text;const substitutions=[];
 for(const [from,to]of [['tests/fixtures/fmri/tiny_real_slice_nifti1.nii',out+'/cases/'+name+'/a.nii'],['tests/fixtures/fmri/tiny_real_slice_nifti2.nii',out+'/cases/'+name+'/b.nii'],['tests/fixtures/fmri/tiny_real_motion.tsv',out+'/cases/'+name+'/motion.tsv'],['tests/fixtures/fmri/tiny_atlas_labels_nifti1.nii',out+'/cases/'+name+'/atlas.nii']]){
  const a='"'+from+'"',b='"'+to+'"';assert.equal(derived.split(a).length-1,1);assert(!derived.includes(b));derived=derived.replace(a,b);substitutions.push({from:a,to:b});
 }
 const begin='// FMRI_CONFIG_BEGIN\n',end='// FMRI_CONFIG_END\n';
 assert.equal(derived.split(begin).length,2);assert.equal(derived.split(end).length,2);
 const from=derived.slice(derived.indexOf(begin)+begin.length,derived.indexOf(end));
 const expression=fs.readFileSync(out+'/'+name+'.config.log','utf8').trim();
 assert.match(expression,/^fmri_index_config_new\([^\r\n;{}]+\)$/);
 const to='    let config = '+expression+'\n';derived=derived.replace(from,to);substitutions.push({from,to});
 let reverse=derived;for(const {from,to}of substitutions.toReversed()){reverse=reverse.replace(to,from);}
 assert(Buffer.from(reverse).equals(original));
 const target=out+'/cases/'+name+'/test.sio';fs.writeFileSync(target,derived,{flag:'wx'});
 records.push({name,expected_native_exit:Number(rc),source:target,original_sha256:hash(original),derived_sha256:hash(derived),reverse_byte_equal:true,substitutions});
}
assert(records.length>=10);fs.writeFileSync(out+'/DERIVATIONS.json',JSON.stringify(records,null,2)+'\n',{flag:'wx'});
JS
count=0
while IFS=$'\t' read -r name expected;do
 case_dir="$out/cases/$name"
 phase "$name" check 300 bash scripts/ci/souc-native-wrapper.sh check "$case_dir/test.sio"
 phase "$name" compile 300 bash scripts/ci/souc-native-wrapper.sh compile "$case_dir/test.sio" -o "$case_dir/program.elf"
 chmod +x "$case_dir/program.elf"
 sha256sum "$case_dir/program.elf" >> "$out/EXECUTABLES.sha256"
 actual=0;phase "$name" native 120 "$case_dir/program.elf" || actual=$?
 [[ "$actual" == "$expected" ]] || { echo "$name expected=$expected actual=$actual" >&2;exit 1; }
 if [[ "$expected" == 0 ]];then
  phase "$name" cpp 30 "$out/reference" --check "$case_dir/a.nii" "$case_dir/b.nii" "$case_dir/motion.tsv" "$case_dir/atlas.nii" "$case_dir/config.tsv" "$out/$name.native.log"
 else
  ! grep -q SCIENCE_FMRI_OK "$out/$name.native.log"
  rejected=0;phase "$name" reference-refusal 30 "$out/reference" --validate "$case_dir/a.nii" "$case_dir/b.nii" "$case_dir/motion.tsv" "$case_dir/atlas.nii" "$case_dir/config.tsv" || rejected=$?
  [[ "$rejected" == 2 ]]
 fi
 count=$((count+1))
done < "$out/cases/PLAN.tsv"
[[ "$count" -ge 10 ]]
phase oracle mutate 30 "$out/reference" --wrong-output "$out/baseline.native.log" "$out/wrong-output.log"
r=0;phase oracle reject-wrong-value 30 "$out/reference" --check "$out/cases/baseline/a.nii" "$out/cases/baseline/b.nii" "$out/cases/baseline/motion.tsv" "$out/cases/baseline/atlas.nii" "$out/cases/baseline/config.tsv" "$out/wrong-output.log" || r=$?
[[ "$r" == 1 ]] && grep -Fq 'numerical mismatch' "$out/oracle.reject-wrong-value.log"
# Resource ownership and reused result availability are separate regressions.
for test in test_native_binary_reader test_native_metrics_reuse;do
 phase "$test" check 300 bash scripts/ci/souc-native-wrapper.sh check "$root/tests/stdlib/fmri/$test.sio"
 phase "$test" compile 300 bash scripts/ci/souc-native-wrapper.sh compile "$root/tests/stdlib/fmri/$test.sio" -o "$out/$test.elf"
 chmod +x "$out/$test.elf"
 sha256sum "$out/$test.elf" >> "$out/EXECUTABLES.sha256"
 phase "$test" native 120 "$out/$test.elf"
 marker=NIFTI_NATIVE_BINARY_READ_OK
 if [[ "$test" == test_native_metrics_reuse ]];then marker=FMRI_NATIVE_REUSE_OK;fi
 [[ "$(grep -Fxc "$marker" "$out/$test.native.log")" == 1 ]]
done
"$node" --input-type=module - "$root" "$out" <<'JS'
import fs from 'node:fs';import assert from 'node:assert/strict';
const [root,out]=process.argv.slice(2),s=fs.readFileSync(root+'/tests/stdlib/fmri/test_native_binary_reader.sio','utf8');
const anchor='with IO,Alloc,Mut,Div,Panic';assert.equal(s.split(anchor).length-1,1);
fs.writeFileSync(out+'/missing-alloc.sio',s.replace(anchor,'with IO,Mut,Div,Panic'),{flag:'wx'});
JS
r=0;phase missing-alloc check 300 bash scripts/ci/souc-native-wrapper.sh check "$out/missing-alloc.sio" || r=$?
[[ "$r" == 1 ]] && grep -Fq E035 "$out/missing-alloc.check.log"
sha256sum -c "$out/BOUND_LIBRARIES.sha256" > "$out/BOUND_LIBRARIES.final.verify"
printf '%s\n' "$count" > "$out/CASE_COUNT"
