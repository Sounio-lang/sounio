#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/root-pbpk-prints04
S=$B/root-bidi-inputs05/source
RAW=$B/root-integrated04/results/madaros
mkdir "$D"
cd "$S"
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
sha256sum "$RAW" >"$D/INPUTS.sha256"
find stdlib -type f -print0|sort -z|xargs -0 sha256sum >>"$D/INPUTS.sha256"
sha256sum tests/stdlib/chemistry/test_pbpk_clearance_ode.sio >>"$D/INPUTS.sha256"
node - "$D" <<'JS'
const fs=require('fs'),d=process.argv[2],s=fs.readFileSync('tests/stdlib/chemistry/test_pbpk_clearance_ode.sio','utf8');
const names=['st_inf','st_neg','s_nuk'];let t=s;for(const n of names){const from='print('+n+')';if(t.split(from).length!==2)throw Error('expected unique '+n);t=t.replace(from,'print_int('+n+')');}fs.writeFileSync(d+'/diagnostic.sio',t);fs.writeFileSync(d+'/DIAGNOSTIC_SCOPE','Only three status print dispatch calls changed; original regression remains unchanged. This is localization, not a production fix.\n');
JS
sha256sum "$D/diagnostic.sio" >>"$D/INPUTS.sha256"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 for phase in check compile;do
  rc=0
  args=("$phase" "$D/diagnostic.sio")
  if [[ "$phase" == compile ]];then args+=(-o "$D/program-$mode.elf");fi
  "$P/tools/time" -v -o "$D/$phase-$mode.resource" timeout 1200 "$RAW" "${args[@]}" >"$D/$phase-$mode.stdout" 2>"$D/$phase-$mode.stderr"||rc=$?
  echo "$rc" >"$D/$phase-$mode.STATUS"
 done
 if [[ -f "$D/program-$mode.elf" ]];then sha256sum "$D/program-$mode.elf" >>"$D/ELFS.sha256";fi
done
