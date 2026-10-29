#!/bin/bash
#SBATCH --job-name=pireus-five-staging-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/pireus-custody48/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/pireus-custody48/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;P=$B/services/pireus-staging43;R=$B/services/pireus-custody48;D=$R/metadata
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
mkdir "$R/archive" "$D"
trap 'r=$?;echo "$r" >"$D/STATUS";exit "$r"' EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$D/SNAPSHOT_TIME"
sacct -j19611 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,NodeList -P >"$D/ACCOUNTING"
node - "$P" >"$D/RESULT.json" <<'JS'
const fs=require('fs'),p=process.argv[2],names=['multiprobe_block_certification','operator_autogenesis','operator_discovery_engine','operator_novelty_frontier','operator_seed_kernel'];
const read=f=>{try{return fs.readFileSync(f,'utf8').trim();}catch{return null;}};
const results=names.map(name=>({name,before:read(p+'/results/'+name+'.before.status'),after:read(p+'/results/'+name+'.after.status')}));
const guards=['INPUTS','TOOLS','ELFS','WITNESSES'].map(name=>({name,complete:read(p+'/results/'+name+'.final.verify')!==null,all_ok:read(p+'/results/'+name+'.final.verify')?.split('\n').every(line=>line.endsWith(': OK'))===true}));
console.log(JSON.stringify({schema:'sounio.pireus.input-closure-result.v1',producer_status:read(p+'/results/STATUS'),results,guards,scope:'Same retained ELFs19392; only cwd data closure differs. Custody success does not convert native failures/timeouts into pass.'},null,2));
JS
for n in INPUTS TOOLS ELFS WITNESSES;do if [[ -s "$P/results/$n.sha256" ]];then sha256sum -c "$P/results/$n.sha256" >"$D/$n.verify";fi;done
cp "$B/whole-integrated01/SOURCE_GIT_SHA" "$D/SOURCE_GIT_SHA"
cp "$B/whole-integrated01/results/full/RESULT.json" "$D/ORIGINAL_WHOLE_RESULT.json"
cd "$B"
tar -czf "$R/archive/pireus-input-closure.tar.gz" services/pireus-staging43 services/pireus-custody48/metadata whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_multiprobe_block_certification.sio.elf whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_multiprobe_block_certification.sio.elf.sha256 whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_autogenesis.sio.elf whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_autogenesis.sio.elf.sha256 whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_discovery_engine.sio.elf whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_discovery_engine.sio.elf.sha256 whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_novelty_frontier.sio.elf whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_novelty_frontier.sio.elf.sha256 whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_seed_kernel.sio.elf whole-integrated01/results/retained/tests/stdlib/hardware/test_pireus_operator_seed_kernel.sio.elf.sha256 whole-integrated01/results/native/libstdlib_platform.so root-integrated01/results/madaros root-integrated01/RAW_READY services-tools-recovered/tools/node
sha256sum "$R/archive/pireus-input-closure.tar.gz" >"$D/ARCHIVE.sha256"
stat -c%s "$R/archive/pireus-input-closure.tar.gz" >"$D/ARCHIVE.bytes"
tar -tzf "$R/archive/pireus-input-closure.tar.gz" >"$D/CONTENTS.list"
echo PIREUS_INPUT_CLOSURE_CUSTODY_OK
