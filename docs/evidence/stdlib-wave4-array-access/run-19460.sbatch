#!/bin/bash
#SBATCH --job-name=bigframe-local-array-candidate
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/bigframe-local-arrays11/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/bigframe-local-arrays11/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;U=$B/root-integrated01;R=$B/services/bigframe-local-arrays11;D=$R/results
RAW=$U/results/madaros;NODE=$B/services-tools-recovered/tools/node;TIME=$B/services/bigframe-integrated01/tools/time
mkdir "$D"
trap 'r=$?;echo "$r" > "$D/COLLECTOR_STATUS";exit "$r"' EXIT
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
mkdir "$R/before" "$R/after"
cp -a "$U/source/stdlib" "$R/before/";cp -a "$U/source/stdlib" "$R/after/"
cp "$R/input/after/stdlib/data/"*.sio "$R/after/stdlib/data/"
"$NODE" - "$R" "$U/source" > "$D/LEDGER.verify" <<'JS'
const fs=require('fs'),crypto=require('crypto'),r=process.argv[2],s=process.argv[3],j=JSON.parse(fs.readFileSync(r+'/input/LOCAL_ARRAY_LEDGER.json','utf8'));
const hash=v=>crypto.createHash('sha256').update(v).digest('hex');
if(j.sites.length!==42||j.files.length!==3)throw Error('empty/changed selection');
for(const f of j.files){
 const old=fs.readFileSync(s+'/'+f.file,'utf8'),a=fs.readFileSync(r+'/after/'+f.file,'utf8'),b=fs.readFileSync(r+'/before/'+f.file,'utf8');
 if(hash(old)!==f.source_before_sha256||b!==old||hash(a)!==f.source_after_sha256)throw Error('source preimage mismatch');
 const restored=a.split('\n');
 for(const x of j.sites.filter(x=>x.file===f.file)){if(restored[x.access_line-1]!==x.after)throw Error('replacement mismatch');restored[x.access_line-1]=x.before;}
 if(restored.join('\n')!==old)throw Error('nonlocal edit');
}
console.log('42_LOCAL_ARRAY_READS_BYTE_EXACT_REVERSE_OK');
JS
find "$R/before" "$R/after" -type f -print0|sort -z|xargs -0 sha256sum > "$D/LIBRARIES.sha256"
sha256sum "$U/source/tests/stdlib/data/test_bigframe_ops_stdlib.sio" "$U/source/tests/stdlib/data/test_bigframe_stats.sio" "$U/source/tests/stdlib/data/test_bigframe_ml.sio" "$NODE" "$TIME" > "$D/WITNESS_TOOLS.sha256"
printf '%s  %s\n' 6f95285b3f9c239382a7f44616644b7cdeeca1740daa51e177edf838c3b48a23 "$U/source/tests/stdlib/data/test_bigframe_ops_stdlib.sio" > "$D/ORIGINAL.sha256"
sha256sum -c "$D/ORIGINAL.sha256" > "$D/ORIGINAL.verify"
cp "$B/services/bigframe-portable09/results/reference" "$D/reference";sha256sum "$D/reference" > "$D/REFERENCE.sha256"
"$D/reference" --self-test > "$D/ORACLE_CONTROLS.stdout"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
phase(){ local label=$1 seconds=$2;shift 2;local rc=0;timeout "$seconds" "$@" > "$D/$label.stdout" 2>&1 ||rc=$?;printf '%s\t%s\n' "$label" "$rc" >> "$D/PHASES.tsv";return "$rc"; }
for variant in before after;do
 export SOUNIO_STDLIB_PATH="$R/$variant/stdlib" SOUNIO_LOOP_REGIONS=0
 for lib in bigframe_ops bigframe_stats bigframe_ml;do
  phase "$variant.$lib.library-check" 600 "$RAW" --check "$R/$variant/stdlib/data/$lib.sio" ||true
 done
 # Scientific sample and ABI outputs are compared using the unchanged C++ oracle.
 for mode in 0 1;do
  export SOUNIO_LOOP_REGIONS="$mode"
  for name in word6 word7 word_wrap5 mixed6 mixed7 mixed_wrap5 frame6 frame7 frame_wrap5 scaled_raw;do
   source="$B/services/bigframe-abi07/input02/$name.sio"
   sha256sum "$source" >> "$D/MATRIX_INPUTS.sha256"
   phase "$variant.$name.$mode.check" 300 "$RAW" --check "$source" ||true
   rc=0;phase "$variant.$name.$mode.compile" 600 "$RAW" compile "$source" -o "$D/$variant.$name.$mode.elf" ||rc=$?
   if [[ "$rc" == 0 && -s "$D/$variant.$name.$mode.elf" ]];then
    chmod +x "$D/$variant.$name.$mode.elf";sha256sum "$D/$variant.$name.$mode.elf" >> "$D/EXECUTABLES.sha256"
    phase "$variant.$name.$mode.native" 30 "$D/$variant.$name.$mode.elf" ||true
    phase "$variant.$name.$mode.oracle" 30 "$D/reference" --verify "$name" "$D/$variant.$name.$mode.native.stdout" ||true
   fi
  done
 done
 # Entire original programs, unchanged assertions and120-second native deadlines.
 for mode in 0 1;do
  export SOUNIO_LOOP_REGIONS="$mode"
  for program in ops stats ml;do
   source="$U/source/tests/stdlib/data/test_bigframe_$program.sio"
   marker=BIGFRAME_${program^^}_GATE_OK
   if [[ "$program" == ops ]];then source="$U/source/tests/stdlib/data/test_bigframe_ops_stdlib.sio";fi
   phase "$variant.original-$program.$mode.check" 300 "$RAW" --check "$source" ||true
   rc=0;phase "$variant.original-$program.$mode.compile" 1200 "$TIME" -v -o "$D/$variant.original-$program.$mode.compile.resource" "$RAW" compile "$source" -o "$D/$variant.original-$program.$mode.elf" ||rc=$?
   if [[ "$rc" == 0 && -s "$D/$variant.original-$program.$mode.elf" ]];then
    chmod +x "$D/$variant.original-$program.$mode.elf";sha256sum "$D/$variant.original-$program.$mode.elf" >> "$D/EXECUTABLES.sha256"
    phase "$variant.original-$program.$mode.native" 120 "$TIME" -v -o "$D/$variant.original-$program.$mode.native.resource" "$D/$variant.original-$program.$mode.elf" ||true
    phase "$variant.original-$program.$mode.marker" 5 grep -Fx "$marker" "$D/$variant.original-$program.$mode.native.stdout" ||true
   fi
  done
 done
done
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum -c "$D/LIBRARIES.sha256" > "$D/LIBRARIES.final.verify"
sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.final.verify"
sha256sum -c "$D/WITNESS_TOOLS.sha256" > "$D/WITNESS_TOOLS.final.verify"
sha256sum -c "$D/MATRIX_INPUTS.sha256" > "$D/MATRIX_INPUTS.final.verify"
sha256sum -c "$D/REFERENCE.sha256" > "$D/REFERENCE.final.verify"
echo 'CAUSAL_DIAGNOSTIC_ONLY: actual before/after phase statuses authoritative; default arena181 remains separate from typed-array-access repair.' > "$D/SCOPE"
echo BIGFRAME_LOCAL_ARRAY_CANDIDATE_RECORDED
