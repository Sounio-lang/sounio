// A closed launcher binds standalone and compound invocation to one raw ELF.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
export const fileDigest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
export function createCompilerBinding(raw,output){
 if(!raw||!path.isAbsolute(raw))throw Error('compound mode requires absolute MADAROS_RAW_BIN');
 raw=fs.realpathSync(raw);fs.accessSync(raw,fs.constants.X_OK);
 const header=Buffer.alloc(4),fd=fs.openSync(raw,'r');try{fs.readSync(fd,header,0,4,0);}finally{fs.closeSync(fd);}
 if(!fs.statSync(raw).isFile()||header.toString('hex')!=='7f454c46')throw Error('compound compiler must be an executable raw ELF');
 const retain=path.join(output,'retained'),launcher=path.join(output,'compiler-launcher.sh');
 fs.mkdirSync(retain);
 const body='#!/bin/bash\nset -euo pipefail\nRAW='+quote(raw)+'\nRETAIN='+quote(retain)+'\n'+
 'rc=0\n"$RAW" "$@" || rc=$?\n'+
 'if [[ $rc == 0 && ${1:-} == compile ]];then\n source=${2:?};shift 2;output=""\n'+
 ' while [[ $# -gt 0 ]];do if [[ $1 == -o ]];then output=${2:?};shift 2;else shift;fi;done\n'+
 ' case "$source" in */tests/stdlib/*.sio)\n relative=${source#*/tests/stdlib/}\n'+
 ' [[ "$relative" != /* && "$relative" != *../* && -n "$output" && -s "$output" ]] || exit 65\n'+
 ' dest="$RETAIN/$relative.elf";mkdir -p "$(dirname "$dest")"\n'+
 ' [[ ! -e "$dest" && ! -e "$dest.sha256" ]] || exit 65\n'+
 ' cp -- "$output" "$dest";cmp -- "$output" "$dest";chmod 700 "$dest";sha256sum "$dest" >"$dest.sha256"\n'+
 ' ;; esac\nfi\nexit "$rc"\n';
 fs.writeFileSync(launcher,body,{flag:'wx',mode:0o700});
 const binding={path:raw,sha256:fileDigest(raw),launcher,launcher_sha256:fileDigest(launcher),retained_dir:retain};
 fs.writeFileSync(path.join(output,'COMPILER_BINDING.json'),JSON.stringify(binding,null,2)+'\n',{flag:'wx'});
 return binding;
}
export function compilerEnvironment(binding,base=process.env){
 return {...base,MADAROS_RAW_BIN:binding.path,SOUC_BIN:binding.launcher,SOUC_NATIVE_BIN:binding.path,SOUNIO_SOUC_BIN:binding.path,SOUNIO_SOUC_RAW_MODE:'modular'};
}
export function verifyCompilerBinding(binding){
 if(fileDigest(binding.path)!==binding.sha256||fileDigest(binding.launcher)!==binding.launcher_sha256)throw Error('bound compiler or launcher changed');
}
