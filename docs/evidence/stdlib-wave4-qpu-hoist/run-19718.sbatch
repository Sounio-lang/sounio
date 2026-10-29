#!/bin/bash
#SBATCH --job-name=qpu-source-hoist-placement-controls
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/qpu-source-validator63/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/qpu-source-validator63/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/services/qpu-source-validator63
S=$B/services/qpu-dyadic-compat58/input
NODE=$B/services-tools-recovered/tools/node
find "$D/input" "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
sha256sum "$NODE" "$0" >"$D/TOOLS.sha256"
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
"$NODE" - "$D" "$S" <<'JS'
const fs=require('fs'),d=process.argv[2],s=process.argv[3];
for(const name of ['before','after']){
 const file=name==='before'?d+'/input/docs/evidence/stdlib-wave4-qpu-hoist/source-validator-7b84.preimage.sh':d+'/input/scripts/stdlib/wave4_qpu_dyadic_gate.sh';
 const shell=fs.readFileSync(file,'utf8'),block=shell.split("<<'JS'\n")[1].split('\nJS')[0];
 const prefix=block.split("if(t.split('@CALLS@')")[0];
 const probe=prefix+"\nconst moved=hoisted.replace('    let p00 = p.p00\\n','').replace('    while k < shots {\\n','    while k < shots {\\n    let p00 = p.p00\\n');let accepted=false;try{sourceKind(moved);accepted=true;}catch{};console.log('MOVED_DECLARATION_ACCEPTED '+accepted);if(accepted!=="+(name==='before'?'true':'false')+")throw Error('unexpected placement result');";
 console.log('VALIDATOR '+name);
 new Function('require','process',probe)(require,{argv:['node','-',s,d,s+'/stdlib']});
}
console.log('EXACT_HOIST_PLACEMENT_BEFORE_AFTER_OK');
JS
