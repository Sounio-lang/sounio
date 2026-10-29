#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/storage-closed-audit05
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288
ulimit -c 0
mkdir "$D/source" "$D/results"
tar -xf "$B/compiler-resumed/storage-type-foundation07/inputs/base.tar" -C "$D/source"
tar -xzf "$D/input/foundation.tar.gz" -C "$D/source"
cp "$D/input/declarations.sio" "$D/source/declarations.sio"
cp "$D/source/self-hosted/check/storage_types.sio" "$D/original.storage_types.sio"
"$P/tools/node" - "$D" <<'JS'
const fs=require('fs'),crypto=require('crypto');const d=process.argv[2];let s=fs.readFileSync(d+'/original.storage_types.sio','utf8');
const old='if index>=0 {\n      if STORAGE_SOURCE_COUNT>=4352 {return ctx}return ctx};index=i';
if(!s.includes(old))throw Error('expected misplaced original guard absent');s=s.replace(old,'if index>=0 {return ctx};index=i');
const append='     if index>=0 {\n      STORAGE_SOURCE_DECLS=';if(!s.includes(append))throw Error('expected original append absent');s=s.replace(append,'     if index>=0 {\n      if STORAGE_SOURCE_COUNT>=4352 {return ctx}\n      STORAGE_SOURCE_DECLS=');
fs.writeFileSync(d+'/candidate.storage_types.sio',s);
const ep=fs.readFileSync(d+'/source/self-hosted/check/epistemic.sio','utf8');
const names=['provenance_new','provenance_is_trusted','validity_always','validity_duration','validity_until','validity_while_pred','validity_unsatisfiable','validity_from_type_entry','knowledge_meta_default','knowledge_meta_from_ty','knowledge_meta_validity','knowledge_meta_provenance','provenance_subsumes','validity_subsumes'];
const pieces=[],hashes=[];
for(const name of ['Provenance','Validity','KnowledgeMeta',...names]){
 const re=new RegExp('^pub (?:fn|struct) '+name+'[ (\\{]','m');const m=re.exec(ep);if(!m)throw Error('missing declaration '+name);
 const end=ep.indexOf('\n}\n',m.index);if(end<0)throw Error('missing declaration end '+name);const body=ep.slice(m.index,end+3);pieces.push(body);hashes.push(crypto.createHash('sha256').update(body).digest('hex')+'  '+name);
}
const constants=ep.match(/^const (?:PROVENANCE|VALIDITY)_KIND_[^\n]+/gm);if(!constants||constants.length!==12)throw Error('constant census');
fs.writeFileSync(d+'/epistemic-deps.sio',constants.join('\n')+'\n'+pieces.join('\n'));
fs.writeFileSync(d+'/EXTRACTED_FUNCTIONS.sha256',hashes.join('\n')+'\n');
fs.writeFileSync(d+'/source/driver.sio','use check::storage_types::{storage_audit_component}\nfn main()->i32 with Mut,Panic,Div,Alloc,IO {storage_audit_component("'+d+'/source/declarations.sio")}\n');
fs.writeFileSync(d+'/source/check_driver.sio','use check::storage_types::{storage_types_reset}\nfn main()->i32 with Mut {storage_types_reset();0}\n');
JS
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$D/input/reference.cpp" -o "$D/results/reference"
sha256sum "$R/results/madaros" "$R/source/bin/souc-lean-single-x86_64" "$CXX" "$D/results/reference" > "$D/TOOLS.sha256"
find "$D/input" -type f -print0 | sort -z | xargs -0 sha256sum > "$D/INPUTS.sha256"
cd "$D/source"
export SOUNIO_STDLIB_PATH="$D/source/self-hosted"
finish(){ r=$?; trap - EXIT; set +e; sha256sum -c "$D/TOOLS.sha256" > "$D/results/tools.verify" || r=1; sha256sum -c "$D/INPUTS.sha256" > "$D/results/inputs.verify" || r=1; printf '%s\n' "$r" > "$D/results/STATUS"; exit "$r"; }
trap finish EXIT
for variant in original candidate; do
 cp "$D/$variant.storage_types.sio" self-hosted/check/storage_types.sio
 timeout 1200 "$R/results/madaros" check check_driver.sio > "$D/results/$variant.check.stdout" 2> "$D/results/$variant.check.stderr"
 cat "$D/$variant.storage_types.sio" "$D/epistemic-deps.sio" "$D/input/probe.sio.in" > self-hosted/check/storage_types.sio
 find self-hosted stdlib driver.sio check_driver.sio declarations.sio -type f -print0 | sort -z | xargs -0 sha256sum > "$D/results/$variant.SOURCE.sha256"
 timeout 1200 "$R/source/bin/souc-lean-single-x86_64" driver.sio "$D/results/$variant.elf" > "$D/results/$variant.compile.stdout" 2> "$D/results/$variant.compile.stderr"
 if grep -E 'error\[E|parse error|Error:|would have dropped|cannot compile an assignment' "$D/results/$variant.compile.stdout" "$D/results/$variant.compile.stderr"; then exit 65; fi
 chmod u+x "$D/results/$variant.elf"
 timeout 90 "$D/results/$variant.elf" > "$D/results/$variant.native.stdout" 2> "$D/results/$variant.native.stderr"
 "$D/results/reference" "$D/results/$variant.native.stdout" "$variant" > "$D/results/$variant.oracle.stdout"
 sha256sum -c "$D/results/$variant.SOURCE.sha256" > "$D/results/$variant.source.verify"
 cp self-hosted/check/storage_types.sio "$D/results/$variant.injected.storage_types.sio"
done
sha256sum "$D/results/original.elf" "$D/results/candidate.elf" > "$D/results/ELFS.sha256"
printf 'STORAGE_CLOSED_AUDIT_PAIRED_COMPONENT_OK\n'
