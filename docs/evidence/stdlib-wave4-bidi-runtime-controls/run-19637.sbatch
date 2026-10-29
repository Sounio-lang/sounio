#!/bin/bash
#SBATCH --job-name=bidi-golden214-byte-concordance
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/bidi-protocol50/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/bidi-protocol50/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/bidi-protocol50
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
trap 'r=$?;echo "$r" >"$R/results/GOLDEN_STATUS";exit "$r"' EXIT
A=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01/focused-source-elf-argv-receipts-19269-19270.tar.gz
printf '19b5e632c3adb31942c181e22941c86e3258ec9d37de58548ce4e6182ba15442  %s\n' "$A" >"$R/results/OLD_CUSTODY.sha256"
sha256sum -c "$R/results/OLD_CUSTODY.sha256" >"$R/results/OLD_CUSTODY.verify"
node - "$B" "$R" >"$R/results/GOLDEN.stdout" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),b=process.argv[2],r=process.argv[3],run=b+'/root-bidi-inputs02',a=run+'/fixture-run/corpus',other=b+'/core/coherent-text-focused01/bidi-corpus';
const sha=x=>crypto.createHash('sha256').update(x).digest('hex'),read=p=>fs.readFileSync(p),id=(base,p)=>{const v=read(path.join(base,p));return {path:p,bytes:v.length,sha256:sha(v)};};
if(read(run+'/fixture-run/STATUS').toString().trim()!=='0'||read(run+'/results/STATUS').toString().trim()!=='0')throw Error('19618 not qualified');
for(const dir of [a,other])if(!fs.lstatSync(dir).isDirectory()||fs.lstatSync(dir).isSymbolicLink())throw Error('nonregular corpus directory');
const names=fs.readdirSync(a).sort(),oldNames=fs.readdirSync(other).sort();if(names.length!==214||JSON.stringify(names)!==JSON.stringify(oldNames))throw Error('corpus census differs');
const files=[];for(const file of names){const p=path.join(a,file),q=path.join(other,file);for(const f of [p,q])if(!fs.lstatSync(f).isFile()||fs.lstatSync(f).isSymbolicLink())throw Error('corpus symlink/nonfile');const x=read(p),y=read(q);if(!x.equals(y))throw Error('CORPUS_BYTE_MISMATCH '+file);files.push({file,bytes:x.length,sha256:sha(x)});}
const counts=JSON.parse(read(a+'/COUNTS.json'));if(counts.cases!==861948||counts.chunks!==211)throw Error('official counts');
const sourceFiles=['stdlib/text/data/ucd18-source.tar.gz','stdlib/text/data/UCD18-SOURCES.sha256','tests/reference/wave3_bidi_generator.cpp','tests/reference/wave3_bidi_cases.hpp'].map(p=>id(run+'/source',p));
const oldInputs=JSON.parse(read(run+'/fixture-run/INPUTS.json'));for(const source of sourceFiles){const original=oldInputs.find(i=>i.path===run+'/source/'+source.path);if(!original||original.sha256!==source.sha256)throw Error('source differs from qualified19618 '+source.path);}
const sourceManifest=read(run+'/source/stdlib/text/data/UCD18-SOURCES.sha256').toString();
const upstream=sourceManifest.trimEnd().split('\n').map(line=>{const m=/^([a-f0-9]{64})  (.+)$/.exec(line);if(!m)throw Error('bad UCD source record');const bytes=read(run+'/fixture-run/upstream/'+m[2]);if(sha(bytes)!==m[1])throw Error('UCD source changed');return {path:m[2],bytes:bytes.length,sha256:m[1]};});
const program=id(run+'/source','tests/stdlib/wave3-bidi/conformance.sio');
const golden={schema:'sounio.unicode18.bidi.official-corpus.v1',files,counts,source_files:sourceFiles,upstream_files:upstream,provenance:{qualification_job:'19618',qualification_status:'COMPLETED0',qualification_scope:'same original Sounio program over all211 official chunks/861948 cases; no ICU or case changes',raw_sha256:read(run+'/results/RAW.sha256').toString().split(' ')[0],source_head:read(run+'/results/SOURCE_HEAD').toString().trim(),source_scope:'qualified immutable snapshot plus recorded WIP; not a clean revision claim',program_source:program,current_corpus:a,crosschecked_corpus:other,files_byte_identical:true,crosschecked_files:files.length,old_custody_archive:'/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01/focused-source-elf-argv-receipts-19269-19270.tar.gz',old_custody_sha256:'19b5e632c3adb31942c181e22941c86e3258ec9d37de58548ce4e6182ba15442'}};
fs.writeFileSync(r+'/results/official-corpus.candidate.json',JSON.stringify(golden,null,2)+'\n');
fs.writeFileSync(r+'/results/ORIGINAL_CORPUS.json',JSON.stringify(files,null,2)+'\n');
console.log('BIDI_GOLDEN_BYTE_IDENTICAL files=214 chunks=211 cases=861948');
JS
sha256sum "$R/results/official-corpus.candidate.json" >"$R/results/GOLDEN.sha256"
echo BIDI_GOLDEN214_SOURCE_BOUND_OK
