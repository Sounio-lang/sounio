#!/bin/bash
#SBATCH --job-name=wide-counts-hoist-discriminator
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/wide-counts34/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/wide-counts34/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/wide-counts34;D=$R/results;U=$B/root-integrated02
mkdir "$D"
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER EXECUTABLES SDK GENERATED LIBRARIES IR;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" > "$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" > "$D/STATUS";exit "$r";};trap finish EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
find "$B/services/quantum-pair-library28/after/stdlib" -type f -print0|sort -z|xargs -0 sha256sum > "$D/LIBRARIES.sha256"
T=$B/services/bigframe-abi07/gcc13-private/root
find "$T" -type f -print0|sort -z|xargs -0 sha256sum > "$D/SDK.sha256"
export PATH="$T/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/deps"
export COMPILER_PATH="$T/usr/libexec/gcc/x86_64-linux-gnu/13:$T/usr/bin"
export LIBRARY_PATH="$T/usr/lib/gcc/x86_64-linux-gnu/13:$T/usr/lib/x86_64-linux-gnu:$T/lib/x86_64-linux-gnu"
"$T/usr/bin/x86_64-linux-gnu-g++-13" --sysroot="$T" -B"$T/usr/libexec/gcc/x86_64-linux-gnu/13/" -B"$T/usr/bin/" -isystem "$T/usr/include/c++/13" -isystem "$T/usr/include/x86_64-linux-gnu/c++/13" -std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc "$R/input/tests/fixtures/wave4_quantum_pair/wide_counts_reference.cpp" -o "$D/reference"
"$D/reference" --self-test > "$D/ORACLE_CONTROLS.stdout"
sha256sum "$D/reference" > "$D/EXECUTABLES.sha256"

"$D/reference" --generate "$D/generated" > "$D/GENERATE.stdout"
NODE=$B/services-tools-recovered/tools/node
"$NODE" - "$R" "$D" <<'JS'
const fs=require('fs'),r=process.argv[2],d=process.argv[3],t=fs.readFileSync(r+'/input/tests/fixtures/wave4_quantum_pair/wide_counts.sio.in','utf8'),calls=fs.readFileSync(d+'/generated/CALLS.sio','utf8');
if(t.split('@CALLS@').length!==2)throw Error('template marker');
fs.writeFileSync(d+'/driver.sio',t.replace('@CALLS@',calls));
JS
find "$D/generated" -type f -print0|sort -z|xargs -0 sha256sum > "$D/GENERATED.sha256";sha256sum "$D/driver.sio" >> "$D/GENERATED.sha256"
export SOUNIO_STDLIB_PATH="$B/services/quantum-pair-library28/after/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_LOOP_REGIONS=0 SOUNIO_NV2_IR_TRACE=1
export LD_LIBRARY_PATH="$R/input/dso:$T/deps"
ulimit -s 524288;ulimit -c 0
phase(){ local label=$1 seconds=$2;shift 2;local r=0;timeout "$seconds" "$@" > "$D/$label.stdout" 2>&1||r=$?;printf '%s\t%s\n' "$label" "$r" >> "$D/PHASES.tsv";return "$r"; }
phase check 300 "$U/results/madaros" --check "$D/driver.sio"
phase compile 600 "$U/results/madaros" compile "$D/driver.sio" -o "$D/driver.elf"
chmod +x "$D/driver.elf";sha256sum "$D/driver.elf" >> "$D/EXECUTABLES.sha256"
ir=0
"$NODE" - "$D" > "$D/IR_GUARD.stdout" 2>&1 <<'JS' ||ir=$?
const fs=require('fs'),d=process.argv[2],s=fs.readFileSync(d+'/compile.stdout','utf8');
const modules=s.split('NV2_IR module fn_count=').slice(1),out=[];
for(const m of modules){const lines=m.split('\n'),by=new Map();for(const line of lines){let a=/^NV2_IR function fn=(\d+) name=(\S+)/.exec(line);if(a)by.set(a[1],{name:a[2],fields:[]});a=/^NV2_IR fn=(\d+) i=(\d+) op=field_get .*field_idx=(-?\d+)/.exec(line);if(a&&by.has(a[1]))by.get(a[1]).fields.push({instruction:Number(a[2]),field:Number(a[3])});}const a=[...by.values()].find(x=>x.name.endsWith('wide_core_literal')),b=[...by.values()].find(x=>x.name.endsWith('wide_core_hoist'));if(a&&b)out.push({literal:a,hoist:b});}
if(out.length===0)throw Error('no IR with both kernels');const last=out.at(-1);fs.writeFileSync(d+'/IR_CENSUS.json',JSON.stringify({modules_with_kernels:out.length,last},null,2)+'\n');
if(last.literal.fields.length!==4||last.hoist.fields.length!==3||last.literal.fields.filter(x=>x.field===0).length!==2||last.hoist.fields.filter(x=>x.field===0).length!==1)throw Error('field projection census differs');
console.log('IR_LITERAL4_HOIST3_P00_2_TO1_OK');
JS
printf '%s\n' "$ir" > "$D/IR_GUARD_STATUS"
sha256sum "$D/compile.stdout" "$D/IR_GUARD.stdout" > "$D/IR.sha256"
if [[ -s "$D/IR_CENSUS.json" ]];then sha256sum "$D/IR_CENSUS.json" >> "$D/IR.sha256";fi
phase native 600 "$D/driver.elf"
rc=0;phase oracle 120 "$D/reference" "$D/native.stdout"||rc=$?
echo 'HOT LOOP ONLY, supplied BasisProb, exact source loop preserved; no complete slow acceptance or QPU-hardware claim. Performance is reported for all three alternating-order seeds.' > "$D/SCOPE"
[[ "$ir" == 0 && "$rc" == 0 ]]
echo WIDE_COUNTS_CAUSAL_DISCRIMINATOR_OK
