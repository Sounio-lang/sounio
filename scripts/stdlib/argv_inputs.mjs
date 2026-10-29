#!/usr/bin/env node
// Literal file-path sidecars, shared by the canonical runner and full-suite guards.
// Node performs process/file orchestration only; this module does no scientific calculation.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const inside=(p,root)=>p===root||p.startsWith(root+path.sep);
const portable=(p,root)=>inside(p,root)?path.relative(root,p):p;
function input(root,p,absolute=false) {
  const requested=absolute?p:portable(p,root);
  try {
    const real=fs.realpathSync(p),st=fs.statSync(real);
    if(!st.isFile())return {path:requested,status:'not-file',real_path:absolute?real:portable(real,root)};
    const bytes=fs.readFileSync(real);
    return {path:requested,status:'file',real_path:absolute?real:portable(real,root),bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex')};
  } catch(e) {
    if(e.code==='ENOENT'||e.code==='ENOTDIR'){
      try{fs.lstatSync(p);return {path:requested,status:'unreadable',error:'dangling link or inaccessible target'};}catch(missing){if(missing.code!=='ENOENT'&&missing.code!=='ENOTDIR')return {path:requested,status:'unreadable',error:missing.code||missing.message};}
      return {path:requested,status:'missing'};
    }
    return {path:requested,status:'unreadable',error:e.code||e.message};
  }
}
export function argvSnapshot(root,test) {
  root=path.resolve(root);
  const sidecarPath=path.resolve(root,test)+'.args';
  const sidecar=input(root,sidecarPath);
  const result={test_path:test,sidecar,argv:[],dependencies:[],error:null};
  if(sidecar.status==='missing')return result;
  if(sidecar.status!=='file'){result.error='sidecar is not a readable regular file';return result;}
  try {
    const raw=fs.readFileSync(sidecarPath);
    const text=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(raw);
    if(text.includes('\0'))throw Error('NUL in sidecar');
    // LF is the delimiter. No shell syntax, quoting, trimming, CR removal or word splitting.
    // Empty lines are ignored, matching pireus_argv_tests_gate.sh; an unterminated last line is valid.
    result.argv=text.split('\n').filter(line=>line.length!==0);
    result.dependencies=result.argv.map(arg=>input(root,path.resolve(root,arg),path.isAbsolute(arg)));
    if(result.dependencies.some(x=>x.status!=='file'))result.error='argv file missing, unreadable or not regular';
  } catch(e) {result.error='malformed sidecar: '+e.message;}
  return result;
}
export function verifyArgvSnapshot(root,before) {
  const after=argvSnapshot(root,before.test_path);
  if(JSON.stringify(after)!==JSON.stringify(before))throw Error('argv input changed: '+before.test_path);
  return after;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const [op,root,test,executable,...extra]=process.argv.slice(2);
  if(op!=='run'||!root||!test||!executable||extra.length){console.error('usage: argv_inputs.mjs run ROOT TEST ELF');process.exitCode=65;}
  else {
    const before=argvSnapshot(root,test);
    let status=65;
    try {
      if(before.error)throw Error(before.error+': '+test);
      verifyArgvSnapshot(root,before);
      const child=spawnSync(executable,before.argv,{cwd:root,stdio:'inherit'});
      if(child.error)throw child.error;
      status=child.signal?128+(os.constants.signals[child.signal]||0):child.status;
      verifyArgvSnapshot(root,before);
    } catch(e) {console.error('error: argv inputs: '+e.message);status=65;}
    process.exitCode=Number.isInteger(status)?status:65;
  }
}
